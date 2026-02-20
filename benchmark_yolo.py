"""
YOLO 모델의 latency, VRAM 사용량, 안정성 평가를 수행
출력 : (누적 저장) latency_vram_predict.csv
"""

import time
import torch
from ultralytics import YOLO
from pathlib import Path
import csv
import datetime
import numpy as np


# =========================
# 설정
# =========================
PRECISIONS = ["fp32", "fp16"]
MODELS = ["yolo11m",
          "yolo11l"
          ]
DATASETS = ["raw_fhd",
            # "tiles_2x4"
            ]

STABILITY_ITERS = 30 # 안정성 평가 반복 횟수

BASE_DIR = Path(__file__).resolve().parent.parent  # model-poc-cessna/
RUN_DIR = BASE_DIR / "runs/yolo"
TEST_LIST_MAP = {
    "tiles_2x4": BASE_DIR / "data/splits/tiles_test.txt",
    "tiles_4x2": BASE_DIR / "data/splits/tiles_4x2_test.txt",
    "raw_fhd": BASE_DIR / "data/splits/raw_fhd_test.txt",
}

CSV_PATH = BASE_DIR / "runs/latency_vram_predict.csv"

DEVICE = 1
IMG_SIZE = 640
NUM_IMAGES = 500        # 벤치마크에 사용할 이미지 수
WARMUP = 100
GPU_INFO = "RTX_4070"

# =========================

# CSV append 함수
def append_benchmark_csv(csv_path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


for MODEL in MODELS:
    for DATASET in DATASETS:
        for PRECISION in PRECISIONS:
            # =========================
            # RUN 정보
            # =========================
            RUN_FOLDER = f"{MODEL}_{DATASET}_{IMG_SIZE}_e300"
            WEIGHT = RUN_DIR / RUN_FOLDER / "weights/best.pt"
            TEST_LIST = TEST_LIST_MAP[DATASET]

            # =========================
            # 이미지 리스트 로드
            # =========================
            with open(TEST_LIST) as f:
                images = [str(BASE_DIR / line.strip()) for line in f if line.strip()]

            images = images[:NUM_IMAGES]
            print(f"\n[YOLO] MODEL={MODEL}, DATASET={DATASET}, PRECISION={PRECISION}")
            print(f"[YOLO] Images loaded: {len(images)}")

            # =========================
            # 모델 로드
            # =========================
            model = YOLO(str(WEIGHT))
            model.fuse()  # fuse를 먼저 수행 (FP32 상태에서)
            model.model.to("cuda")

            if PRECISION == "fp16":
                model.model.half()
            else:
                model.model.float()

            model.model.eval()

            # =========================
            # Warmup
            # =========================
            print("[YOLO] Warmup...")
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to("cuda")
            if PRECISION == "fp16":
                dummy = dummy.half()

            for _ in range(WARMUP):
                model.model(dummy)

            torch.cuda.synchronize()

            # =========================
            # 시간 Benchmark
            # =========================
            torch.cuda.reset_peak_memory_stats()
            latencies = []

            print("[YOLO] Benchmark start")

            with torch.no_grad():
                for img in images:
                    start = time.time()
                    model(img, imgsz=IMG_SIZE, device=DEVICE)
                    torch.cuda.synchronize()
                    latencies.append(time.time() - start)

            lat = np.array(latencies)
            avg_latency_ms = lat.mean() * 1000
            min_latency_ms = lat.min() * 1000
            max_latency_ms = lat.max() * 1000
            std_latency_ms = lat.std() * 1000
            avg_fps = 1.0 / lat.mean()

            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

            # =========================
            # 안정성 평가 (출력)
            # =========================
            xs = []
            detected = 0
            if DATASET=="tiles_2x4":
                STABILITY_IMAGE = BASE_DIR / "data/tiles_2x4/images/20211113134901_20211113141045_1.mp4_20220307_114757.664_r0c3.png" # 안정성 평가 전용 이미지
            elif DATASET=="tiles_4x2":
                STABILITY_IMAGE = BASE_DIR / "data/tiles_4x2/images/20211113134901_20211113141045_1.mp4_20220307_114757.664_r1c1.png" # 안정성 평가 전용 이미지
            elif DATASET=="raw_fhd":
                STABILITY_IMAGE = BASE_DIR / "Z:/home/rs02/NAS/Dataset/Cessna_220429_fhd/images/20211113134901_20211113141045_1.mp4_20220307_123355.406.png" # 안정성 평가 전용 이미지
            test_img = str(STABILITY_IMAGE)

            print("\n[YOLO] 안정성 평가 start")
            with torch.no_grad():
                for _ in range(STABILITY_ITERS):
                    result = model(test_img, imgsz=IMG_SIZE, device=DEVICE)[0]
                    if len(result.boxes) > 0:
                        detected += 1   # detect 개수 count (detection stability)
                        box = result.boxes.xyxy[0]
                        x_center = (box[0] + box[2]) / 2  # bbox x 중심 흔들림 (bbox jitter)
                        xs.append(x_center.item())

            bbox_jitter_px = float(np.std(xs)) if len(xs) > 1 else 0.0
            detection_stability = detected / STABILITY_ITERS * 100

            # =========================
            # CSV 저장
            # =========================
            print("\n========== YOLO Benchmark ==========")
            print(f"Images        : {len(images)}")
            print(f"Avg Latency   : {avg_latency_ms:.2f} ms")
            print(f"Min Latency   : {min_latency_ms:.2f} ms")
            print(f"Max Latency   : {max_latency_ms:.2f} ms")
            print(f"Std Dev       : {std_latency_ms:.2f} ms")
            print(f"Avg FPS       : {avg_fps:.2f}")
            print(f"detected_cnt  : {detected}")
            print(f"Bbox jitter   : {bbox_jitter_px:.2f} px")
            print(f"Detection stability: {detection_stability:.1f} %")
            print(f"Peak VRAM     : {peak_mem_mb:.1f} MB")
            print("===================================")


            row = {
                "model_family": "yolo",
                "model_name": MODEL,
                "dataset": DATASET,
                "precision": PRECISION,
                "batch_size": 1,
                "img_size": IMG_SIZE,
                "num_images": len(images),
                "avg_latency_ms": round(avg_latency_ms, 4),
                "min_latency_ms": round(min_latency_ms, 4),
                "max_latency_ms": round(max_latency_ms, 4),
                "std_latency_ms": round(std_latency_ms, 4),
                "avg_fps": round(avg_fps, 2),
                "detected_cnt": detected,
                "bbox_jitter_px": round(bbox_jitter_px, 2),
                "detection_stability_pct": round(detection_stability, 1),
                "peak_vram_mb": round(peak_mem_mb, 1),
                "gpu": GPU_INFO,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            }

            append_benchmark_csv(CSV_PATH, row)

            print(f"[YOLO] Benchmark saved to {CSV_PATH}")