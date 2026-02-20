# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install ultralytics

from ultralytics.utils.benchmarks_custom import benchmark
from ultralytics import YOLO

# # Load the YOLO26 model
# model = YOLO("yolo26n.pt")

# # Export the model to TensorRT format
# model.export(format="engine")  # creates 'yolo26n.engine'

if __name__ == "__main__":
    # Benchmark on GPU
    benchmark(
        model="models/yolo11l_p2_cessna_640/yolo11l_p2_cessna_640.pt", data="cfg/cessna_fhd.yaml", imgsz=640, half=False, device=0, verbose=True
    )