# ultralytics 공식 벤치마크(커스텀) 실행 / 모델 export 및 성능 경향성 확인
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install ultralytics

from ultralytics.utils.benchmarks_custom import benchmark
from ultralytics import YOLO

# # Load the YOLO26 model
# model = YOLO("yolo26n.pt")

# # Export the model to TensorRT format
# model.export(format="engine")  # creates 'yolo26n.engine'

if __name__ == "__main__":
    # 테스트할 모델 리스트 정의
    model_names = [
        # "yolo11l_p2_cessna_640",
        # "yolo11l_p2_cessna_960",
        # "yolo11l_p2_cessna_1088",
        # "yolo11l_cessna_640",
        # "yolo11l_cessna_960",
        # "yolo11l_cessna_1088",
        "yolo26l_cessna_640",
        "yolo26l_cessna_960",
        "yolo26l_cessna_1088",
    ]

    for name in model_names:
        print(f"\n{'=' * 30}")
        print(f"Starting Benchmark: {name}")
        print(f"{'=' * 30}")

        # 모델 이름에서 imgsz 추출 (문자열 마지막의 숫자 부분)
        target_imgsz = int(name.split('_')[-1])
        model_path = f"models/{name}/{name}.pt"

        # Benchmark 실행
        benchmark(
            model=model_path,
            data="cfg/cessna_fhd.yaml",
            imgsz=target_imgsz,
            half=False,
            device=0,
            verbose=True
        )
        print(f"✅ Finished: {name}\n")
