# 모델 벤치마크 프로젝트
용도
> - yolo26, yolo11 학습 및 비교
> - yolo11l+p2 (외부 프로젝트) 벤치마크


### 가상환경 세팅

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

### 스크립트 실행순서
1. `yolo_txt_to_coco.py` : YOLO 라벨(txt) → COCO GT(json)  (ap_small 계산을 위해 yolo도 변환 필요)
2. `ultralytics_bench(export).py` : 모델을 PyTorch, ONNX, TensorRT, TorchScript 형식으로 변환하여 저장 및 벤치마크(ultralytics 메서드 활용)
2. `master_runner.py` : single_eval.py를 지정한 모델, img_size, 모델확장자 별로 실행하여 평가 결과를 csv로 누적 저장

### 경로 설정
- `master_runner.py` : 테스트 img_size, 평가할 모델 폴더, 모델 확장자 설정
- `single_eval.py` : data yaml, test dataset, coco 라벨 데이터(GT(json)) 경로 설정 필수

### 모델 확장자
- `.pt` : PyTorch 모델
- `.onnx` : ONNX 모델
- `.engine` : TensorRT 엔진 모델 (ONNX → TensorRT 변환 필요)
- `.torchscript` : TorchScript 모델
- 평가하기 전 `ultralytics_bench(export).py`로 모델 변환 필요