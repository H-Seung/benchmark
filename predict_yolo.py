"""
터미널 `python predict_yolo.py` 실행
지정된 모델과 데이터셋으로 테스트셋에 대한 예측을 수행하고 JSON 파일로 저장
출력 : predictions.json, eval_metrics_yolo.json
"""

from ultralytics import YOLO
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent  # model-poc-cessna/
RUN_DIR = BASE_DIR / "runs/yolo"
MODEL = "yolo11m"
# DATASET = "tiles_2x4"
DATASET = f"raw_fhd"  # "tiles_2x4" or "raw"
TRAIN_IMGSZ = 640
EPOCHS = 300
RUN_FOLDER = f"{MODEL}_{DATASET}_{TRAIN_IMGSZ}_e{EPOCHS}"

WEIGHT = RUN_DIR / RUN_FOLDER / "weights/best.pt"

PROJECT = f"runs/yolo/{RUN_FOLDER}"
out_path = Path(f"{PROJECT}/predict/eval_metrics_yolo.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

def main():
    model = YOLO(str(WEIGHT))

    metrics = model.val(
        data=f"data/yolo/{DATASET}.yaml",
        split="test",
        save_json=True,
        imgsz=640,
        conf=0.001,
        iou=0.65,
        batch=16,
        device=0,
        half=False,  # FP32
        project=PROJECT,
        name="predict",
        exist_ok=True
    )

    # COCO metric 추출 (YOLO 내부 구조)
    s = metrics.box.mean_results()
    # 순서: [mAP, AP50, AP75, AP_small, AP_medium, AP_large]
    print("s:",s)

    metrics_out = {
        "model": MODEL,
        "dataset": f"{DATASET}",
        "mAP": float(metrics.box.map),       # mAP@[.5:.95]
        "AP50": float(metrics.box.map50),
        "AP75": float(metrics.box.map75),
    }

    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2)


if __name__ == '__main__':
    main()