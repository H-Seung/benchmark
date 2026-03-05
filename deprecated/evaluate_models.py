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


def measure_pipeline_latency(model, images, imgsz):
    # Warmup
    for img in images[:PIPE_WARMUP]:
        model(img, imgsz=imgsz, device=DEVICE, verbose=False)

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    times = []

    with torch.no_grad():
        for img in images[PIPE_WARMUP:]:
            start = time.time()
            model(img, imgsz=imgsz, device=DEVICE, verbose=False)
            torch.cuda.synchronize()
            times.append(time.time() - start)

    times = np.array(times)

    vram_max = torch.cuda.max_memory_allocated() / 1024**2

    return times.mean()*1000, times.std()*1000, vram_max


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
        if model_format == "pt":  # PT일 경우만 직접 .model 접근
            model.model.to("cuda")
            model.model.float()
            model.model.eval()

        for current_imgsz in sizes_to_test:
            print(f" >> Evaluating at Size: {current_imgsz}")

            # 1️⃣ Accuracy
            print("- Evaluate Accuracy...")
            acc = evaluate_accuracy(model, model_name, model_format, current_imgsz)
            torch.cuda.empty_cache()

            # 2️⃣ Core latency  (pt만 가능)
            print("- Measure Core latency...")
            if model_format == "pt":
                core_avg, core_std = measure_core_latency(model, current_imgsz)
            else:
                core_avg, core_std = np.nan, np.nan

            # 3️⃣ Pipeline latency
            print("- Measure Pipeline latency...")
            pipe_avg, pipe_std, vram_max = measure_pipeline_latency(model, images, current_imgsz)

            fps_avg = 1000.0 / pipe_avg
            fps_std = (pipe_std / pipe_avg) * fps_avg

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
                "pipeline_latency_avg_ms": round(pipe_avg, 3),
                "pipeline_latency_std_ms": round(pipe_std, 3),
                "fps_avg": round(fps_avg, 2),
                "fps_std": round(fps_std, 2),
                "vram_max_mb": round(vram_max, 1),
                "gpu": torch.cuda.get_device_name(0),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds")
            }

            append_csv(row)
            print(f"    - Size {current_imgsz} Done.")

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
