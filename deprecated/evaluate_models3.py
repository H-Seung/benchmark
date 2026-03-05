"""
Model Evaluation Script
- Accuracy
- Core Latency : pt만 가능, 다른 포맷은 dummy forward 불가하므로 np.nan 처리
- Pipeline Latency : image loading, io + preprocess + postprocess
- VRAM usage
- CSV export

Before running:
1. Fix GPU clock (optional but recommended)
2. Close other GPU workloads
"""

import os
import time
import csv
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import datetime
import json
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# =============================
# CONFIG
# =============================

DEVICE = 0
DATA_YAML = "cfg/cessna_fhd.yaml"
TEST_TXT = "data/cessna_fhd/test.txt"
GT_JSON = "d:/SW-test/model-poc-cessna/data/coco/raw_fhd/annotations/instances_test.json"

CORE_WARMUP = 100
CORE_ITERS = 300

PIPE_WARMUP = 20
PIPE_ITERS = 200

CSV_PATH = "../evaluation_results.csv"

IMG_SIZES = [640, 960, 1088]
MODELS_ROOT = Path("../models")
TARGET_MODEL_FOLDERS = [
    "yolo11l_p2_cessna_640",
    "yolo11l_p2_cessna_960",
    "yolo11l_p2_cessna_1088",
    "yolo11l_cessna_640",
    "yolo11l_cessna_960",
    "yolo11l_cessna_1088",
    "yolo26l_cessna_640",
    "yolo26l_cessna_960",
    "yolo26l_cessna_1088",
]
VALID_SUFFIXES = [
    ".pt",
    ".onnx",
    ".engine",
    ".torchscript"
]
# =============================
# --- NVML (real GPU memory) ---
try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


def init_nvml():
    """Initialize NVML once. Safe to call multiple times."""
    global _NVML_AVAILABLE
    if not _NVML_AVAILABLE:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        _NVML_AVAILABLE = False
        return False


def get_gpu_total_mem_mb(device_index: int = 0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem.used / (1024 ** 2)


def get_model_paths():
    model_files = []
    for folder_name in TARGET_MODEL_FOLDERS:
        folder = MODELS_ROOT / folder_name  # folder : models\yolo11l_p2_cessna_960
        if not folder.exists():
            print(f"[WARNING] Folder not found: {folder}")
            continue

        for file in folder.iterdir():
            if file.suffix in VALID_SUFFIXES:
                model_files.append(file)
    return model_files


def load_test_images():
    with open(TEST_TXT) as f:
        imgs = [line.strip() for line in f if line.strip()]
    return imgs[:PIPE_WARMUP + PIPE_ITERS]


def yolo_predict_to_coco(gt_json, pred_json, out_json):

    with open(gt_json, "r") as f:
        gt = json.load(f)

    assert len(gt["categories"]) == 1, "Single-class GT only supported"

    gt_cat_id = gt["categories"][0]["id"]

    filename_to_id = {
        Path(img["file_name"]).name: img["id"]
        for img in gt["images"]
    }

    with open(pred_json, "r") as f:
        yolo_preds = json.load(f)

    coco_dets = []
    skipped = 0

    for p in yolo_preds:

        file_name = Path(p["file_name"]).name

        if file_name not in filename_to_id:
            skipped += 1
            continue

        coco_dets.append({
            "image_id": int(filename_to_id[file_name]),
            "category_id": int(gt_cat_id),
            "bbox": [float(b) for b in p["bbox"]],
            "score": float(p["score"]),
        })

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(coco_dets, f, indent=2)

    print(f"[DONE] COCO detections saved: {out_json}")
    print(f"[COCO] Converted: {len(coco_dets)}, Skipped: {skipped}")


def calculate_ap_small(pred_coco_path):

    coco_gt = COCO(GT_JSON)
    coco_dt = coco_gt.loadRes(pred_coco_path)

    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.evaluate()
    e.accumulate()
    e.summarize()

    # stats index
    # 0 = mAP
    # 1 = AP50
    # 2 = AP75
    # 3 = AP_small
    # 4 = AP_medium
    # 5 = AP_large
    stats = e.stats

    return float(stats[3])


def evaluate_accuracy(model, model_name, model_format, imgsz):

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
        half=False,  # FP32
        project=save_dir.parent,
        name=save_dir.name,
        verbose=False
    )

    results = metrics.results_dict

    pred_json = save_dir / "predictions.json"
    pred_coco_json = save_dir / "predictions_coco.json"
    if not pred_json.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_json}")
    yolo_predict_to_coco(GT_JSON, pred_json, pred_coco_json)  # YOLO 결과 → COCO 변환
    ap_small = calculate_ap_small(str(pred_coco_json))

    return {
        "mAP50-95": float(results["metrics/mAP50-95(B)"]),
        "recall": float(results["metrics/recall(B)"]),
        "precision": float(results["metrics/precision(B)"]),
        "AP_small": ap_small
    }


def measure_core_latency(model, imgsz):
    dummy = torch.randn(1, 3, imgsz, imgsz).to("cuda")

    model.model.eval()

    # Warmup
    for _ in range(CORE_WARMUP):
        model.model(dummy)

    torch.cuda.synchronize()

    times = []

    with torch.no_grad():
        for _ in range(CORE_ITERS):
            start = time.time()
            model.model(dummy)
            torch.cuda.synchronize()
            times.append(time.time() - start)

    times = np.array(times)

    return times.mean()*1000, times.std()*1000


def _ensure_predictor(model: YOLO, imgsz: int):
    """
    Make sure model.predictor exists and is configured for current imgsz/device.
    We call model.predict once (fast) to initialize predictor consistently across formats.
    """
    # This triggers predictor creation internally (ultralytics behavior)
    model.predict(source=ASSET_DUMMY_IMAGE, imgsz=imgsz, device=DEVICE, verbose=False)
    pred = getattr(model, "predictor", None)
    if pred is None:
        raise RuntimeError("model.predictor not initialized. Ultralytics version mismatch?")
    # Ensure args are set
    pred.args.imgsz = imgsz
    pred.args.device = DEVICE
    pred.args.verbose = False
    return pred


# use any small existing image path for predictor init
ASSET_DUMMY_IMAGE = str(Path("ultralytics") / "assets" / "bus.jpg")  # fallback
if not Path(ASSET_DUMMY_IMAGE).exists():
    # if ultralytics assets not present in your runtime, just use first test image later
    ASSET_DUMMY_IMAGE = None


def measure_pipeline_latency_split(model: YOLO, images, imgsz: int):
    """
    Measure pipeline latency split into:
    - io_decode: cv2.imread (disk + decode)
    - preprocess: predictor.preprocess (includes resize/letterbox + tensor conversion + H2D)
    - inference: predictor.inference
    - postprocess: predictor.postprocess (includes decode + NMS + results creation)
    - total: sum of above

    VRAM (Windows/WDDM friendly):
    - baseline0: predictor init 전 total GPU used
    - baseline1: predictor init 후 total GPU used (workspace/buffers 반영)
    - peak_total: 측정 루프 중 최대 total GPU used
    - delta_from_baseline0 = peak_total - baseline0
    - delta_after_predictor = peak_total - baseline1
    """

    # -------------------------
    # Warmup (기존 end-to-end)
    # -------------------------
    for img in images[:PIPE_WARMUP]:
        model(img, imgsz=imgsz, device=DEVICE, verbose=False)
    torch.cuda.synchronize()

    # Torch allocator peak reset (참고용)
    torch.cuda.reset_peak_memory_stats()

    # -------------------------
    # 0) baseline0 (predictor init 전)
    # -------------------------
    baseline0 = get_gpu_total_mem_mb(DEVICE)
    # Windows에서 NVML 갱신이 늦는 경우가 있어 약간 대기
    time.sleep(0.05)
    baseline0 = get_gpu_total_mem_mb(DEVICE)

    # -------------------------
    # Predictor init
    # -------------------------
    global ASSET_DUMMY_IMAGE
    if ASSET_DUMMY_IMAGE is None:
        ASSET_DUMMY_IMAGE = images[0]
    predictor = _ensure_predictor(model, imgsz)

    # predictor init이 GPU 메모리 할당을 유발할 수 있으니 sync + sleep
    torch.cuda.synchronize()
    time.sleep(0.05)

    # -------------------------
    # 1) baseline1 (predictor init 후)
    # -------------------------
    baseline1 = get_gpu_total_mem_mb(DEVICE)
    time.sleep(0.05)
    baseline1 = get_gpu_total_mem_mb(DEVICE)

    # -------------------------
    # Timings
    # -------------------------
    t_io, t_pre, t_inf, t_post, t_total = [], [], [], [], []

    # VRAM peak(total device used)
    peak_total = max(baseline0, baseline1)

    # -------------------------
    # Measurement loop
    # -------------------------
    with torch.no_grad():
        for idx, img_path in enumerate(images[PIPE_WARMUP:], start=1):

            # 1) IO + decode
            t0 = time.perf_counter()
            im0 = cv2.imread(img_path)  # BGR uint8
            t1 = time.perf_counter()
            if im0 is None:
                raise FileNotFoundError(f"cv2.imread failed: {img_path}")

            # 2) Preprocess (includes H2D)
            im0s = [im0]
            tp0 = time.perf_counter()
            im = predictor.preprocess(im0s)
            tp1 = time.perf_counter()

            # 3) Inference
            ti0 = time.perf_counter()
            preds = predictor.inference(im)
            torch.cuda.synchronize()
            ti1 = time.perf_counter()

            # 4) Postprocess (NMS/scale boxes/create Results)
            tpo0 = time.perf_counter()
            _ = predictor.postprocess(preds, im, im0s)
            torch.cuda.synchronize()
            tpo1 = time.perf_counter()

            # Total
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

            # -------------------------
            # VRAM sampling (너무 자주 읽으면 의미 없을 때가 있어서 5프레임마다)
            # -------------------------
            if idx % 5 == 0:
                # sync + small sleep로 NVML 갱신 반영
                torch.cuda.synchronize()
                time.sleep(0.01)
                cur = get_gpu_total_mem_mb(DEVICE)
                if cur > peak_total:
                    peak_total = cur

    # 루프 끝나고 한 번 더 peak 갱신(마지막 할당 반영)
    torch.cuda.synchronize()
    time.sleep(0.05)
    peak_total = max(peak_total, get_gpu_total_mem_mb(DEVICE))

    # -------------------------
    # Aggregate
    # -------------------------
    def _mean_std(arr):
        arr = np.array(arr, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    io_avg, io_std = _mean_std(t_io)
    pre_avg, pre_std = _mean_std(t_pre)
    inf_avg, inf_std = _mean_std(t_inf)
    post_avg, post_std = _mean_std(t_post)
    tot_avg, tot_std = _mean_std(t_total)

    # Torch allocator peak (참고용)
    vram_torch_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # -------------------------
    # VRAM deltas
    # -------------------------
    delta_from_baseline0 = peak_total - baseline0
    delta_after_predictor = peak_total - baseline1

    return {
        # latency split
        "io_avg_ms": io_avg, "io_std_ms": io_std,
        "pre_avg_ms": pre_avg, "pre_std_ms": pre_std,
        "inf_avg_ms": inf_avg, "inf_std_ms": inf_std,
        "post_avg_ms": post_avg, "post_std_ms": post_std,
        "total_avg_ms": tot_avg, "total_std_ms": tot_std,

        # VRAM (total device used, baseline split)
        "vram_baseline0_mb": baseline0,
        "vram_baseline1_mb": baseline1,
        "vram_total_peak_mb": peak_total,
        "vram_delta_from_baseline0_mb": delta_from_baseline0,
        "vram_delta_after_predictor_mb": delta_after_predictor,

        # reference
        "vram_torch_peak_mb": vram_torch_peak,
    }


def append_csv(row):
    file_exists = os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    init_nvml()

    model_files = get_model_paths()
    images = load_test_images()

    print(f"Found {len(model_files)} models. Testing sizes: {IMG_SIZES}")

    for model_file in model_files:
        model_format = model_file.suffix.replace(".", "")
        model_name = model_file.stem
        # 학습 imgsz 정보 추출
        try:
            model_train_imgsz = int(model_name.split("_")[-1])  # 예: yolo11l_p2_cessna_640 → 640
        except ValueError:
            raise ValueError(f"Cannot extract imgsz from model name: {model_name}")
        # PT만 IMG_SIZES 전체 평가, 나머지는 학습 imgsz 1개만 평가
        if model_format == "pt":
            sizes_to_test = IMG_SIZES
        else:
            sizes_to_test = [model_train_imgsz]

        print(f"\n--- Evaluating {model_name} ({model_format}) ---")
        print(f"Sizes to test: {sizes_to_test}")

        model = YOLO(str(model_file))

        torch.cuda.synchronize()
        time.sleep(0.1)
        model_load_baseline = get_gpu_total_mem_mb(DEVICE)
        print(f"[VRAM] After model load: {model_load_baseline:.1f} MB")

        if model_format == "pt":  # PT일 경우만 직접 .model 접근
            model.model.to("cuda")
            model.model.float()
            model.model.eval()

        for current_imgsz in sizes_to_test:
            print(f" >> Evaluating at Size: {current_imgsz}")

            # 1️⃣ Accuracy
            print("- Evaluate Accuracy...")
            before_val = get_gpu_total_mem_mb(DEVICE)
            acc = evaluate_accuracy(model, model_name, model_format, current_imgsz)
            torch.cuda.synchronize()
            time.sleep(0.1)
            after_val = get_gpu_total_mem_mb(DEVICE)
            print(f"[VRAM] Before val: {before_val:.1f} MB")
            print(f"[VRAM] After val: {after_val:.1f} MB")
            torch.cuda.empty_cache()

            # 2️⃣ Core latency  (pt만 가능)
            print("- Measure Core latency...")
            if model_format == "pt":
                core_avg, core_std = measure_core_latency(model, current_imgsz)
            else:
                core_avg, core_std = np.nan, np.nan

            # 3️⃣ Pipeline latency (split)
            print("- Measure Pipeline latency (split: io/pre/inf/post)...")
            pipe = measure_pipeline_latency_split(model, images, current_imgsz)

            fps_avg = 1000.0 / pipe["total_avg_ms"]
            fps_std = (pipe["total_std_ms"] / pipe["total_avg_ms"]) * fps_avg

            row = {
                "model_name": model_name,
                "format": model_format,
                "img_size": current_imgsz,
                "mAP50-95": round(acc["mAP50-95"], 4),
                "AP_small": round(acc["AP_small"], 4),
                "recall": round(acc["recall"], 4),
                "precision": round(acc["precision"], 4),
                "core_latency_avg_ms": None if np.isnan(core_avg) else round(core_avg, 3),
                "core_latency_std_ms": None if np.isnan(core_std) else round(core_std, 3),
                "pipeline_latency_avg_ms": round(pipe["total_avg_ms"], 3),
                "pipeline_latency_std_ms": round(pipe["total_std_ms"], 3),
                "io_latency_avg_ms": round(pipe["io_avg_ms"], 3),
                "io_latency_std_ms": round(pipe["io_std_ms"], 3),
                "pre_latency_avg_ms": round(pipe["pre_avg_ms"], 3),
                "pre_latency_std_ms": round(pipe["pre_std_ms"], 3),
                "infer_latency_avg_ms": round(pipe["inf_avg_ms"], 3),
                "infer_latency_std_ms": round(pipe["inf_std_ms"], 3),
                "post_latency_avg_ms": round(pipe["post_avg_ms"], 3),
                "post_latency_std_ms": round(pipe["post_std_ms"], 3),
                "fps_avg": round(fps_avg, 2),
                "fps_std": round(fps_std, 2),
                # VRAM (total device used - baseline split)
                "vram_baseline0_mb": round(pipe["vram_baseline0_mb"], 1),
                "vram_baseline1_mb": round(pipe["vram_baseline1_mb"], 1),
                "vram_total_peak_mb": round(pipe["vram_total_peak_mb"], 1),
                "vram_delta_from_baseline0_mb": round(pipe["vram_delta_from_baseline0_mb"], 1),
                "vram_delta_after_predictor_mb": round(pipe["vram_delta_after_predictor_mb"], 1),
                # Torch allocator reference
                "vram_torch_peak_mb": round(pipe["vram_torch_peak_mb"], 1),
                "gpu": torch.cuda.get_device_name(0),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds")
            }

            append_csv(row)
            print(f"    - Size {current_imgsz} Done.")

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()