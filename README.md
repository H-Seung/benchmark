> - yolo26, yolo11 학습 및 비교
> - yolo11l+p2 (외부 프로젝트) 벤치마크


### 가상환경 세팅

- python 3.11
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics
pip install netron  # 모델 시각화 툴
```

기존에 C++ TensorRT SDK 를 설치했던 상태이므로 python과 연동하는 작업만 진행

[NVIDIA 공식문서](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#method-5-zip-file-installation-windows) 참고


**Step 5 (Optional): Install Python wheels**

```
 python.exe -m pip install tensorrt-10.3.0-cp311-none-win_amd64.whl
```
python 만 사용한다면 애초에 SDK를 설치할 필요는 없고, pip 방식으로 설치 진행하면 된다.

### 스크립트 실행순서
1. (옵션)`yolo_txt_to_coco.py` : YOLO 라벨(txt) → COCO GT(json)
2. `predict_yolo.py` : test셋에 validation 수행 및 json 결과 저장
3. `calcul_yolo_metric.py` : YOLO json 결과를 coco 포맷으로 변환 & AP_small 등 계산
4. `benchmark_yolo.py` : 시간/vram 측정 코드