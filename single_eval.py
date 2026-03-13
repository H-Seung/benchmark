"""
- Evaluate ONE model file at ONE imgsz (test dataset)
- Metrics: mAP50-95, AP_small, recall, precision
- Latency:
  - core_latency (pure inference): predictor.inference() for ALL formats
  - pipeline split: io (cv2.imread) / pre / infer / post / total
- VRAM:
  - Prefer NVML per-process usedGpuMemory
  - Fallback to NVML total device used (WDDM)
- Append one row to CSV

Usage:
  python single_eval.py --model "models/.../yolo11l_cessna_640.engine" --imgsz 640
"""

import argparse
import csv
import datetime
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import psutil

# =============================
# CONFIG
# =============================
DEVICE = 0
DATA_YAML = "cfg/cessna_fhd.yaml"
TEST_TXT = "data/cessna_fhd/test.txt"
GT_JSON = r"d:/SW-test/model-poc-cessna/data/coco/raw_fhd/annotations/instances_test.json"

CORE_WARMUP = 100
CORE_ITERS = 300

PIPE_WARMUP = 20
PIPE_ITERS = 200

CSV_PATH = "evaluation_results.csv"

# Assets dummy image (predictor init용)
ASSET_DUMMY_IMAGE = str(Path("ultralytics") / "assets" / "bus.jpg")
if not Path(ASSET_DUMMY_IMAGE).exists():
    ASSET_DUMMY_IMAGE = None

# =============================
# Set single-thread for CPU operations to reduce variability
# (안하면 opencv, torch가 cpu thread 많이 사용해서 thread 스케줄링에 따른 latency 변동 심해짐)
# =============================
cv2.setNumThreads(1)
torch.set_num_threads(1)

# =============================
# NVML
# =============================
try:
    import pynvml

    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


def init_nvml() -> bool:
    global _NVML_AVAILABLE
    if not _NVML_AVAILABLE:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        _NVML_AVAILABLE = False
        return False


def get_gpu_total_mem_mb(device_index: int = 0) -> float:
    """Total device used (WDDM friendly, but system-wide)."""
    if not _NVML_AVAILABLE:
        return float("nan")
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return float(mem.used) / (1024**2)


def get_gpu_process_mem_mb(device_index: int = 0) -> float:
    """Per-process VRAM. WDDM에서는 안 잡힐 수 있어 NaN/0 나올 수 있음."""
    if not _NVML_AVAILABLE:
        return float("nan")
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        pid = os.getpid()

        # try v3 first (newer)
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
        except Exception:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

        for p in procs:
            if int(p.pid) == int(pid):
                return float(p.usedGpuMemory) / (1024**2)

        # WDDM에서 compute list에 안 뜨는 경우가 있어 NaN 반환
        return float("nan")
    except Exception:
        return float("nan")


def get_gpu_utilization(device_index: int = 0) -> float:
    """GPU compute utilization (%)"""
    if not _NVML_AVAILABLE:
        return float("nan")
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return float("nan")


# =============================
# Data helpers
# =============================
def load_test_images():
    with open(TEST_TXT, "r", encoding="utf-8") as f:
        imgs = [line.strip() for line in f if line.strip()]
    return imgs[: PIPE_WARMUP + PIPE_ITERS]


def yolo_predict_to_coco(gt_json, pred_json, out_json):
    with open(gt_json, "r", encoding="utf-8") as f:
        gt = json.load(f)
    assert len(gt["categories"]) == 1, "Single-class GT only supported"
    gt_cat_id = gt["categories"][0]["id"]

    filename_to_id = {Path(img["file_name"]).name: img["id"] for img in gt["images"]}

    with open(pred_json, "r", encoding="utf-8") as f:
        yolo_preds = json.load(f)

    coco_dets = []
    skipped = 0
    for p in yolo_preds:
        file_name = Path(p["file_name"]).name
        if file_name not in filename_to_id:
            skipped += 1
            continue
        coco_dets.append(
            {
                "image_id": int(filename_to_id[file_name]),
                "category_id": int(gt_cat_id),
                "bbox": [float(b) for b in p["bbox"]],
                "score": float(p["score"]),
            }
        )

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco_dets, f, indent=2)

    print(f"[COCO] Converted: {len(coco_dets)}, Skipped: {skipped}")


def calculate_ap_small(pred_coco_path):
    coco_gt = COCO(GT_JSON)
    coco_dt = coco_gt.loadRes(pred_coco_path)

    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.evaluate()
    e.accumulate()
    e.summarize()

    # stats index:
    # 0=mAP, 1=AP50, 2=AP75, 3=AP_small, 4=AP_medium, 5=AP_large
    return float(e.stats[3])


# =============================
# Accuracy
# =============================
def evaluate_accuracy(model: YOLO, model_name: str, model_format: str, imgsz: int):
    save_dir = Path(f"benchmark_results/{model_name}.{model_format}_i{imgsz}").absolute()
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        save_json=True,
        imgsz=imgsz,
        batch=16,
        conf=0.001,
        iou=0.65,
        device=DEVICE,
        half=False,  # FP32 fixed (your choice)
        project=save_dir.parent,
        name=save_dir.name,
        verbose=False,
    )

    results = metrics.results_dict
    pred_json = save_dir / "predictions.json"
    pred_coco_json = save_dir / "predictions_coco.json"
    if not pred_json.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_json}")

    yolo_predict_to_coco(GT_JSON, pred_json, pred_coco_json)
    ap_small = calculate_ap_small(str(pred_coco_json))

    return {
        "mAP50-95": float(results["metrics/mAP50-95(B)"]),
        "recall": float(results["metrics/recall(B)"]),
        "precision": float(results["metrics/precision(B)"]),
        "AP_small": float(ap_small),
    }


# =============================
# Predictor helpers (core + pipeline)
# =============================
def _ensure_predictor(model: YOLO, imgsz: int, warmup_source: str):
    """
    Ensure model.predictor exists for all formats.
    Uses model.predict once to init predictor.
    """
    model.predict(source=warmup_source, imgsz=imgsz, device=DEVICE, verbose=False)
    pred = getattr(model, "predictor", None)
    if pred is None:
        raise RuntimeError("model.predictor not initialized.")
    pred.args.imgsz = imgsz
    pred.args.device = DEVICE
    pred.args.verbose = False
    return pred


def measure_core_latency_universal(model: YOLO, imgsz: int, predictor):
    """
    Pure inference time measured via predictor.inference(im)
    - pre-created tensor (H2D 제외 목적)
    - 모든 포맷에서 동작하는 공정 측정
    """
    # predictor.preprocess는 list[np.ndarray]를 받아서 torch tensor를 만들지만,
    # core는 '순수 inference'만 보려는 목적이라 torch tensor를 직접 만듦.
    im = torch.randn(1, 3, imgsz, imgsz, device=f"cuda:{DEVICE}", dtype=torch.float32)

    # 엔진이 FP16일 수도 있으니 "실제 predictor가 기대하는 dtype"에 맞추기
    # (안 맞추면 내부에서 캐스팅/복사가 발생할 수 있음)
    try:
        # pt의 경우 model.model.parameters 가능
        if hasattr(model, "model") and hasattr(model.model, "parameters"):
            p = next(model.model.parameters(), None)
            if p is not None and p.dtype == torch.float16:
                im = im.half()
    except Exception:
        # engine/onnx/torchscript은 여기서 안전하게 무시
        pass

    # Warmup
    for _ in range(CORE_WARMUP):
        _ = predictor.inference(im)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(CORE_ITERS):
            t0 = time.perf_counter()
            _ = predictor.inference(im)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    arr = np.array(times, dtype=np.float64) * 1000.0
    return float(arr.mean()), float(arr.std())


def measure_pipeline_latency_split(model: YOLO, predictor, images, imgsz: int):
    """
    Pipeline latency split (std 제거 버전)
    - io: cv2.imread
    - pre: predictor.preprocess
    - infer: predictor.inference
    - post: predictor.postprocess
    - total: sum
    """
    # Warmup: end-to-end
    for img in images[:PIPE_WARMUP]:
        model(img, imgsz=imgsz, device=DEVICE, verbose=False)
    torch.cuda.synchronize()

    t_io, t_pre, t_inf, t_post, t_total = [], [], [], [], []
    cpu_samples = []
    gpu_samples = []
    process = psutil.Process(os.getpid())
    process.cpu_percent(None)  # reset

    with torch.no_grad():
        for img_path in images[PIPE_WARMUP:]:
            # 1) IO
            t0 = time.perf_counter()
            im0 = cv2.imread(img_path)
            t1 = time.perf_counter()
            if im0 is None:
                raise FileNotFoundError(f"cv2.imread failed: {img_path}")

            # 2) Pre
            im0s = [im0]
            tp0 = time.perf_counter()
            im = predictor.preprocess(im0s)
            tp1 = time.perf_counter()

            # 3) Infer
            ti0 = time.perf_counter()
            preds = predictor.inference(im)
            torch.cuda.synchronize()
            ti1 = time.perf_counter()

            # 4) Post
            tpo0 = time.perf_counter()
            _ = predictor.postprocess(preds, im, im0s)
            torch.cuda.synchronize()
            tpo1 = time.perf_counter()

            io_ms = (t1 - t0) * 1000
            pre_ms = (tp1 - tp0) * 1000
            inf_ms = (ti1 - ti0) * 1000
            post_ms = (tpo1 - tpo0) * 1000
            total_ms = io_ms + pre_ms + inf_ms + post_ms

            t_io.append(io_ms)
            t_pre.append(pre_ms)
            t_inf.append(inf_ms)
            t_post.append(post_ms)
            t_total.append(total_ms)

            cpu_samples.append(process.cpu_percent(None)) # cpu 샘플 리스트
            gpu_samples.append(get_gpu_utilization(DEVICE)) # gpu 샘플 리스트

    def _mean(a):
        return float(np.mean(np.array(a, dtype=np.float64)))

    return {
        "io_latency_ms": _mean(t_io),
        "pre_latency_ms": _mean(t_pre),
        "infer_latency_ms": _mean(t_inf),
        "post_latency_ms": _mean(t_post),
        "pipeline_latency_ms": _mean(t_total),
        "fps_avg": 1000.0 / _mean(t_total),
        "cpu_percent": _mean(cpu_samples),
        "gpu_util_percent": _mean(gpu_samples),
    }


# =============================
# CSV append
# =============================
def append_csv(row: dict):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model (.pt/.onnx/.engine/.torchscript)")
    parser.add_argument("--imgsz", type=int, required=True, help="Image size to evaluate")
    parser.add_argument("--train_imgsz", type=int, default=-1, help="Training imgsz (for logging only)")
    args = parser.parse_args()

    model_path = Path(args.model)
    model_format = model_path.suffix.replace(".", "")
    model_name = model_path.stem
    imgsz = int(args.imgsz)

    # NVML init
    init_nvml()

    images = load_test_images()
    warmup_source = ASSET_DUMMY_IMAGE if ASSET_DUMMY_IMAGE else images[0]

    # ---- VRAM baseline (process preferred) ----
    vram_before_proc = get_gpu_process_mem_mb(DEVICE)
    vram_before_total = get_gpu_total_mem_mb(DEVICE)

    # Load model
    model = YOLO(str(model_path))

    # Predictor init (first predict)
    predictor = _ensure_predictor(model, imgsz, warmup_source)

    torch.cuda.synchronize()
    time.sleep(0.05)

    # ---- VRAM service (after predictor init) ----
    vram_service_proc = get_gpu_process_mem_mb(DEVICE)
    vram_service_total = get_gpu_total_mem_mb(DEVICE)

    # Choose VRAM metric:
    # - If per-process is valid (not NaN), use it
    # - Else fallback to total device used (but subprocess separation makes it still meaningful)
    def _pick_vram(before_proc, after_proc, before_total, after_total):
        if not np.isnan(after_proc) and after_proc > 0:
            return before_proc if not np.isnan(before_proc) else 0.0, after_proc
        return before_total, after_total

    system_vram_before_mb, service_vram_mb = _pick_vram(
        vram_before_proc, vram_service_proc, vram_before_total, vram_service_total
    )
    vram_delta_mb = service_vram_mb - system_vram_before_mb

    same_res = "O" if args.train_imgsz == imgsz and args.train_imgsz > 0 else ""

    print(f"\n=== {model_name} ({model_format}) @ {imgsz} ===")
    print(f"[VRAM] before={system_vram_before_mb:.1f} MB  service={service_vram_mb:.1f} MB  delta={vram_delta_mb:.1f} MB")

    # 1) Accuracy (요청대로 포함)
    print("- Evaluate Accuracy...")
    acc = evaluate_accuracy(model, model_name, model_format, imgsz)

    # 2) Core latency (universal)
    print("- Measure Core latency (universal predictor.inference)...")
    core_avg, core_std = measure_core_latency_universal(model, imgsz, predictor)

    # 3) Pipeline split (mean only)
    print("- Measure Pipeline latency (io/pre/infer/post)...")
    pipe = measure_pipeline_latency_split(model, predictor, images, imgsz)

    row = {
        "model_name": model_name,
        "format": model_format,
        "train_imgsz": (args.train_imgsz if args.train_imgsz > 0 else ""),
        "test_imgsz": imgsz,
        "mAP50-95": round(acc["mAP50-95"], 4),
        "AP_small": round(acc["AP_small"], 4),
        "recall": round(acc["recall"], 4),
        "precision": round(acc["precision"], 4),

        "core_latency_avg_ms": round(core_avg, 3),
        "core_latency_std_ms": round(core_std, 3),

        "io_latency_ms": round(pipe["io_latency_ms"], 3),
        "pre_latency_ms": round(pipe["pre_latency_ms"], 3),
        "infer_latency_ms": round(pipe["infer_latency_ms"], 3),
        "post_latency_ms": round(pipe["post_latency_ms"], 3),
        "pipeline_latency_ms": round(pipe["pipeline_latency_ms"], 3),
        "fps_avg": round(pipe["fps_avg"], 2),

        "cpu_%": round(pipe["cpu_percent"], 2), # 전체 cpu 중 이 프로세스가 사용한 평균 비율
        "gpu_util_%": round(pipe["gpu_util_percent"], 2),

        "system_vram_before_mb": round(system_vram_before_mb, 1),
        "service_vram_mb": round(service_vram_mb, 1),
        "vram_delta_mb": round(vram_delta_mb, 1),

        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "same_res": same_res,
    }

    append_csv(row)
    print("[CSV] appended.")

    # Clean exit (process ends -> VRAM released)
    # No need to del/empty_cache; process termination is the cleanup.


if __name__ == "__main__":
    main()