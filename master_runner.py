"""
master_runner.py
- One-run automation:
  - discovers model files under MODELS_ROOT/TARGET_MODEL_FOLDERS
  - runs single_eval.py in a NEW process per (model, imgsz)
  - results are appended to the same CSV

Usage:
  python master_runner.py
"""

import subprocess
from pathlib import Path
import sys

# =============================
# CONFIG (copied from your script)
# =============================
IMG_SIZES = [640, 960, 1088]
MODELS_ROOT = Path("models")
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
VALID_SUFFIXES = [".pt", ".onnx", ".engine", ".torchscript"]
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
    return sorted(model_files)


def extract_train_imgsz(model_stem: str) -> int:
    # 예: yolo11l_p2_cessna_640 -> 640
    return int(model_stem.split("_")[-1])


def main():
    model_files = get_model_paths()
    print(f"Found {len(model_files)} model files.")

    for model_path in model_files:
        model_format = model_path.suffix.replace(".", "")
        model_name = model_path.stem

        try:
            train_imgsz = extract_train_imgsz(model_name)
        except Exception:
            print(f"[SKIP] cannot extract train imgsz from: {model_name}")
            continue

        # PT는 여러 imgsz 평가, 나머지는 학습 imgsz만
        if model_format == "pt":
            sizes_to_test = IMG_SIZES
        else:
            sizes_to_test = [train_imgsz]

        for imgsz in sizes_to_test:
            print(f"\n[RUN] {model_path} @ imgsz={imgsz}")

            # new process per case
            cmd = [
                sys.executable,  # 현재 실행 중인 python 경로 사용
                "single_eval.py",
                "--model", str(model_path),
                "--imgsz", str(imgsz),
                "--train_imgsz", str(train_imgsz),
            ]
            ret = subprocess.run(cmd)

            if ret.returncode != 0:
                print(f"[ERROR] failed: {model_path} @ {imgsz} (code={ret.returncode})")
                # continue to next one

    print("\nAll done.")


if __name__ == "__main__":
    main()