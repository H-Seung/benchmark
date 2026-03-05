"""
Model Evaluation Script

Changes:
🔵 Accuracy moved to last
🔵 Core latency now universal (all formats)
🔵 VRAM simplified to:
    - system_vram_before_mb
    - service_vram_mb
🔴 Removed baseline0/1/peak/delta
🔴 Removed std for io/pre/infer/post
(🔵 = 수정, 🟢 = 추가, 🔴 = 제거된 개념 정리)
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
# NVML (total device memory, WDDM friendly)
# =============================

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


def get_gpu_total_mem_mb(device_index: int = 0):
    if not _NVML_AVAILABLE:
        return float("nan")
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem.used / (1024 ** 2)


# =============================
# Model Path
# =============================

def get_model_paths():
    model_files = []
    for folder_name in TARGET_MODEL_FOLDERS:
        folder = MODELS_ROOT / folder_name
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


# =============================
# Accuracy
# =============================

def yolo_predict_to_coco(gt_json, pred_json, out_json):

    with open(gt_json, "r") as f:
        gt = json.load(f)

    gt_cat_id = gt["categories"][0]["id"]

    filename_to_id = {
        Path(img["file_name"]).name: img["id"]
        for img in gt["images"]
    }

    with open(pred_json, "r") as f:
        yolo_preds = json.load(f)

    coco_dets = []
    for p in yolo_preds:
        file_name = Path(p["file_name"]).name
        if file_name not in filename_to_id:
            continue

        coco_dets.append({
            "image_id": int(filename_to_id[file_name]),
            "category_id": int(gt_cat_id),
            "bbox": [float(b) for b in p["bbox"]],
            "score": float(p["score"]),
        })

    with open(out_json, "w") as f:
        json.dump(coco_dets, f)


def calculate_ap_small(pred_coco_path):

    coco_gt = COCO(GT_JSON)
    coco_dt = coco_gt.loadRes(pred_coco_path)

    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.evaluate()
    e.accumulate()
    e.summarize()

    return float(e.stats[3])


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
        half=False,
        project=save_dir.parent,
        name=save_dir.name,
        verbose=False
    )

    results = metrics.results_dict

    pred_json = save_dir / "predictions.json"
    pred_coco_json = save_dir / "predictions_coco.json"

    yolo_predict_to_coco(GT_JSON, pred_json, pred_coco_json)
    ap_small = calculate_ap_small(str(pred_coco_json))

    return {
        "mAP50-95": float(results["metrics/mAP50-95(B)"]),
        "recall": float(results["metrics/recall(B)"]),
        "precision": float(results["metrics/precision(B)"]),
        "AP_small": ap_small
    }


# =============================
# 🔵 Core Latency (Universal 적용)
# =============================

def measure_core_latency(model, imgsz):

    # 🔵 predictor 강제 생성 (모든 형식 대응)
    model.predict(
        source=np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
        imgsz=imgsz,
        device=DEVICE,
        verbose=False
    )

    predictor = model.predictor
    im = torch.randn(1, 3, imgsz, imgsz, device="cuda").float()

    for _ in range(CORE_WARMUP):
        predictor.inference(im)
    torch.cuda.synchronize()

    times = []
    for _ in range(CORE_ITERS):
        t0 = time.perf_counter()
        predictor.inference(im)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    return times.mean()*1000, times.std()*1000


# =============================
# 🔵 Pipeline Split (std 제거)
# =============================

def measure_pipeline_latency_split(model, images, imgsz):

    model.predict(source=images[0], imgsz=imgsz, device=DEVICE, verbose=False)
    predictor = model.predictor

    for img in images[:PIPE_WARMUP]:
        model(img, imgsz=imgsz, device=DEVICE, verbose=False)
    torch.cuda.synchronize()

    t_io, t_pre, t_inf, t_post, t_total = [], [], [], [], []

    for img_path in images[PIPE_WARMUP:]:

        t0 = time.perf_counter()
        im0 = cv2.imread(img_path)
        t1 = time.perf_counter()

        im0s = [im0]

        tp0 = time.perf_counter()
        im = predictor.preprocess(im0s)
        tp1 = time.perf_counter()

        ti0 = time.perf_counter()
        preds = predictor.inference(im)
        torch.cuda.synchronize()
        ti1 = time.perf_counter()

        tpo0 = time.perf_counter()
        predictor.postprocess(preds, im, im0s)
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

    total_avg = np.mean(t_total)
    total_std = np.std(t_total)

    return {
        "io_avg_ms": np.mean(t_io),
        "pre_avg_ms": np.mean(t_pre),
        "inf_avg_ms": np.mean(t_inf),
        "post_avg_ms": np.mean(t_post),
        "total_avg_ms": total_avg,
        "total_std_ms": total_std,
    }


# =============================
# MAIN
# =============================

def append_csv(row):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():

    model_files = get_model_paths()
    images = load_test_images()

    for model_file in model_files:

        model_format = model_file.suffix.replace(".", "")
        model_name = model_file.stem

        try:
            model_train_imgsz = int(model_name.split("_")[-1])
        except:
            raise ValueError(f"Cannot extract imgsz from {model_name}")

        if model_format == "pt":
            sizes_to_test = IMG_SIZES
        else:
            sizes_to_test = [model_train_imgsz]

        model = YOLO(str(model_file))

        for current_imgsz in sizes_to_test:

            print(f"\n=== {model_name} | {current_imgsz} ===")

            # 🟢 system VRAM before
            system_vram_before = get_gpu_total_mem_mb(DEVICE)
            print(f"[VRAM] Before service: {system_vram_before:.1f} MB")

            # 🟢 first inference for engine init
            model(np.zeros((current_imgsz, current_imgsz, 3), dtype=np.uint8),
                  imgsz=current_imgsz,
                  device=DEVICE,
                  verbose=False)
            torch.cuda.synchronize()
            time.sleep(0.1)

            # 🟢 service steady-state VRAM
            service_vram = get_gpu_total_mem_mb(DEVICE)
            print(f"[VRAM] After service: {system_vram_before:.1f} MB")

            # 🔵 Core latency
            core_avg, core_std = measure_core_latency(model, current_imgsz)

            # 🔵 Pipeline latency
            pipe = measure_pipeline_latency_split(model, images, current_imgsz)

            fps_avg = 1000.0 / pipe["total_avg_ms"]
            fps_std = (pipe["total_std_ms"] / pipe["total_avg_ms"]) * fps_avg

            # 🔵 Accuracy moved to last
            acc = evaluate_accuracy(model, model_name, model_format, current_imgsz)

            row = {
                "model_name": model_name,
                "format": model_format,
                "img_size": current_imgsz,

                "mAP50-95": round(acc["mAP50-95"], 4),
                "AP_small": round(acc["AP_small"], 4),
                "recall": round(acc["recall"], 4),
                "precision": round(acc["precision"], 4),

                "core_latency_avg_ms": round(core_avg, 3),
                "core_latency_std_ms": round(core_std, 3),

                "io_latency_ms": round(pipe["io_avg_ms"], 3),
                "pre_latency_ms": round(pipe["pre_avg_ms"], 3),
                "infer_latency_ms": round(pipe["inf_avg_ms"], 3),
                "post_latency_ms": round(pipe["post_avg_ms"], 3),

                "pipeline_latency_ms": round(pipe["total_avg_ms"], 3),
                "pipeline_latency_std_ms": round(pipe["total_std_ms"], 3),

                "fps_avg": round(fps_avg, 2),
                "fps_std": round(fps_std, 2),

                "system_vram_before_mb": round(system_vram_before, 1),
                "service_vram_mb": round(service_vram, 1),
                "vram_delta_mb": round(service_vram - system_vram_before, 1),

                "gpu": torch.cuda.get_device_name(0),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds")
            }

            append_csv(row)

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()