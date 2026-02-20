"""
YOLO prediction 결과를 COCO format으로 변환하고
AP_small / AP_medium / AP_large / mAP을 계산하여 JSON + CSV로 저장
출력 : yolo_predictions_coco.json, eval_metrics.json, (누적 저장) eval_metrics_predict.csv
"""

import json
import csv
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ================== 사용자 설정 ==================

MODEL = "yolo11m"   # 다음 실험: yolov8l, yolo11x, yolo11l
DATASET = f"raw_fhd"  # "tiles_2x4" or "raw"
TRAIN_IMGSZ = 640
EPOCHS = 300
TASK = "predict"  # "train" or "predict"

# ================================================

if TASK == "train":
    GT_JSON = f"../data/coco/{DATASET}/annotations/instances_val.json"
if TASK == "predict":
    GT_JSON = f"../data/coco/{DATASET}/annotations/instances_test.json"
run_name = f"{MODEL}_{DATASET}_{TRAIN_IMGSZ}_e{EPOCHS}"  # yolov8x_raw_640_e300
RUN_DIR = f"../runs/yolo/{run_name}"

# -------------------------------------------------
# YOLO prediction → COCO detection 변환
# -------------------------------------------------
def yolo_predict_to_coco(gt_json, pred_json, out_json):
    with open(gt_json, "r") as f:
        gt = json.load(f)

    assert len(gt["categories"]) == 1, "Single-class GT만 지원"
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
    print(f"[INFO] Converted: {len(coco_dets)}, Skipped: {skipped}")


# -------------------------------------------------
# COCO metric 계산
# -------------------------------------------------
def calculate_coco_metrics(gt_json, pred_coco_json):
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(pred_coco_json))

    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.evaluate()
    e.accumulate()
    e.summarize()

    stats = e.stats.tolist()
    return {
        "mAP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
    }


# -------------------------------------------------
# CSV 누적 저장 (비교 실험용)
# -------------------------------------------------
def append_to_csv(csv_path, row_dict):
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

# -------------------------------------------------
# main
# -------------------------------------------------
if __name__ == "__main__":
    run_dir = Path(RUN_DIR)
    run_name = run_dir.name

    if TASK == "train":
        pred_json = run_dir / "predictions.json"
        pred_coco_json = run_dir / "yolo_predictions_coco.json"
        eval_json = run_dir / "eval_metrics.json"
        eval_csv = run_dir.parent.parent / "eval_metrics_train.csv"  # runs/

    if TASK == "predict":
        pred_json = run_dir / "predict/predictions.json"
        pred_coco_json = run_dir / "predict/yolo_predictions_coco.json"
        eval_json = run_dir / "predict/eval_metrics.json"
        eval_csv = run_dir.parent.parent / "eval_metrics_predict.csv"  # runs/

    if not pred_json.exists():
        raise FileNotFoundError(f"predictions.json not found: {pred_json}")

    # 1. YOLO → COCO
    yolo_predict_to_coco(GT_JSON, pred_json, pred_coco_json)


    # 2. COCO metric 계산
    metrics = calculate_coco_metrics(GT_JSON, pred_coco_json)

    metrics_out = {
        "run_name": run_name,
        "model_family": "yolo",
        "model": MODEL,
        "dataset": DATASET,
        **metrics,
    }

    # 3. JSON 저장 (각 run 디렉토리 내부 : runs/yolo/{run_name}/)
    with open(eval_json, "w") as f:
        json.dump(metrics_out, f, indent=2)

    # 4. CSV 누적 저장 (runs/)
    append_to_csv(eval_csv, metrics_out)

    print("\n[DONE] Evaluation completed")
    print(f" - JSON: {eval_json}")
    print(f" - CSV : {eval_csv}")
    print(f" - mAP : {metrics['mAP']:.4f}")
