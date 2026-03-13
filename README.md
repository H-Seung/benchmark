# YOLO Model Performance Benchmark
이 프로젝트는 다양한 포맷(`.pt`, `.onnx`, `.engine`, `.torchscript`)의 YOLO 모델을 대상으로 **정확도(Accuracy)**, **지연 시간(Latency)**, **자원 사용량(VRAM)**을 객관적으로 측정하고 분석합니다.

## 가상환경 세팅

- python 3.11
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics
pip install netron  # 모델 시각화 툴
```
TensorRT:
기존에 C++ TensorRT SDK 를 설치했던 상태이므로 python과 연동하는 작업만 진행

[NVIDIA 공식문서](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#method-5-zip-file-installation-windows) 참고

**Step 5 (Optional): Install Python wheels**

```
 python.exe -m pip install tensorrt-10.3.0-cp311-none-win_amd64.whl
```
**python 만 사용한다면 애초에 SDK를 설치할 필요는 없고, pip 방식으로 설치하면 된다.**

## 스크립트 실행순서
1. `yolo_txt_to_coco.py` : YOLO 라벨(txt) → COCO GT(json)  (ap_small 계산을 위해 yolo도 변환 필요)
2. `ultralytics_bench(export).py` : 모델을 PyTorch, ONNX, TensorRT, TorchScript 형식으로 변환하여 저장 및 벤치마크(ultralytics 메서드 활용)
2. `master_runner.py` : single_eval.py를 지정한 모델, img_size, 모델확장자 별로 실행하여 평가 결과를 csv로 누적 저장


> **Note**: 정확한 벤치마크를 위해 측정 중에는 다른 프로세스(ex. 인터넷 브라우저)를 종료하는 것이 권장됩니다.</br>
> 또한, 안정적인 성능을 원한다면 GPU 클럭을 고정할 수 있습니다. (ex. `nvidia-smi -lgc 2500,2500` - RTX 4070 Ti 기준) ([NVIDIA Docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html?utm_source=chatgpt.com#gpu-clock-locking-and-floating-clock))


## 경로 설정
- `master_runner.py` : 테스트 img_size, 평가할 모델 폴더, 모델 확장자 설정
- `single_eval.py` : data yaml, test dataset, coco 라벨 데이터(GT(json)) 경로 설정 필수

## 지원 모델 확장자
- `.pt` : PyTorch 모델
- `.onnx` : ONNX 모델
- `.engine` : TensorRT 엔진 모델 (ONNX → TensorRT 변환 필요)
- `.torchscript` : TorchScript 모델
- 평가하기 전 `ultralytics_bench(export).py`로 모델 변환 필요

---
## 📊 성능 지표 정의 (Metrics Definition)

### 1. Accuracy (정확도)

COCO Evaluation 표준을 따르며, 테스트 데이터셋을 통해 모델의 검출 능력을 평가합니다.

* **mAP50-95**: IoU 임계값 0.5~0.95 구간의 평균 정밀도 (종합 성능).
* **AP_small**: $32 \times 32$ 픽셀 미만의 소형 객체에 대한 검출 정확도.
* **Precision / Recall**: 모델의 정밀도 및 재현율.

### 2. Latency (지연 시간)

추론 속도는 신뢰성을 위해 **Core(순수 연산)** 와 **Pipeline(전 과정)** 으로 나누어 측정합니다.</br>
실시간 추론 상황을 모방하기 위해 loop문으로 이미지를 1장씩 처리하는 방식으로 진행합니다.

#### **A. Core Latency (Inference Only)**

외부 I/O 및 전처리를 배제한 **GPU 가속기 상의 순수 모델 연산 속도**입니다.

* **Method**: 메모리에 상주된 Dummy Tensor(Input)를 활용한 반복 추론.
* **Warmup**: GPU 초기 가동 및 L2 캐싱을 위해 **100회** 실행.
* **Iterations**: **300회** 반복 실행 후 평균 및 표준편차 계산.
* **Synchronization**: `torch.cuda.synchronize()`를 호출하여 비동기 연산이 완전히 완료된 시점을 기록.
* **Constraint**: 측정 중 GPU 클럭 변동을 방지하기 위해 일정한 부하를 유지.

#### **B. Pipeline Latency Split**

실제 서비스 환경에서의 전체 처리 과정을 단계별로 측정합니다.
* **Method**: preprocess 에서 넘어오는 실제 test data 이미지를 활용한 추론.
* **Warmup**: **20회** 실행.
* **Iterations**: **200회** 반복 실행 후 평균 및 표준편차 계산.
* 나머지 동일 

stages:
* **IO**: 스토리지에서 이미지를 읽어오는 시간 (`cv2.imread`).
* **Pre-process**: 이미지 리사이징, 정규화, FP16/32 변환 및 H2D 전송.
* **Infer**: 실제 하드웨어 가속기를 통한 추론 연산.
* **Post-process**: NMS(Non-Maximum Suppression) 및 좌표 복원 처리.
* -> **FPS (Avg)**: 전체 파이프라인 합산 시간 기반의 초당 프레임 수.

### 3. Resource (자원 사용량)

NVIDIA Management Library(NVML)를 통해 하드웨어 점유율을 실시간 모니터링합니다.

* **VRAM Delta**: 모델 로드 전/후의 점유 차이를 통해 **모델이 사용하는 순수 GPU 메모리**를 산출
* **GPU Util (%)**: 추론 중 GPU 연산 코어의 평균 활용률.
* **CPU Usage (%)**: 이 프로세스가 사용한 CPU time을 전체 CPU capacity로 나눈 값 (측정 변동 최소화를 위해 `Single-thread` 모드로 고정 실행)

---

## 🛠 실행 방법 (Usage)

### 주요 인자(Arguments)

| Parameter | Type | Description                                           |
| --- | --- |-------------------------------------------------------|
| `--model` | `str` | 모델 파일 경로 (`.pt`, `torchscript`, `.onnx`, `.engine` 등) |
| `--imgsz` | `int` | 테스트를 수행할 이미지 해상도                                      |
| `--train_imgsz` | `int` | (선택) 학습 시 해상도 기록용                                     |

### 단일 모델 평가
특정 모델 파일과 이미지 사이즈를 지정하여 벤치마크를 수행하고 결과를 CSV에 누적합니다.
```bash
python single_eval.py --model "models/yolo11l_cessna_640.engine" --imgsz 640 --train_imgsz 640
```
### 다중 모델 평가
`master_runner.py`를 활용하여 지정된 폴더 내의 모든 모델 파일과 이미지 사이즈 조합에 대해 자동으로 벤치마크를 수행합니다.
```bash
master_runner.py 스크립트에서 모델 폴더, 이미지 사이즈 리스트, 모델 확장자 설정 후 실행
```
---

## 📝 출력 결과 (CSV Output)

실행 결과는 `evaluation_results.csv` 파일에 아래 항목으로 자동 기록됩니다.

* 모델 정보 (Name, Format, Train Imgsz, Test Imgsz)
* 정확도 지표 (mAP, AP_small, Precision, Recall)
* 지연 시간 상세 (Core, IO, Pre, Infer, Post, FPS)
* 자원 사용량 (VRAM Delta, GPU/CPU Util)
* 테스트 환경 (GPU Model, Timestamp)

---
