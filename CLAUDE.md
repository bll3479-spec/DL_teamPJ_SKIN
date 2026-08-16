# 개발 환경 노트

## 하드웨어 (이 노트북)
- GPU: Intel(R) Iris(R) Xe Graphics (내장그래픽) — **NVIDIA GPU 없음**
- Python: 3.11.0 (`py -0p`로 확인, `C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe`)
- 주의: PATH의 `python` 명령은 Windows Store 스텁을 가리킴. 실제 인터프리터는 `py` 런처로 확인할 것.

## requirements.txt 이슈 (다른 환경에서 받아온 것)
- `torch==2.13.0+cu130`, `torchvision==0.28.0+cu130`처럼 버전 뒤에 `+cuXXX`가 붙은 건 **CUDA 빌드 로컬 버전 태그**(PEP 440 local version identifier)로, PyPI가 아니라 PyTorch 전용 인덱스(`download.pytorch.org/whl/cuXXX`)에서만 배포됨.
- 기본 `pip install -r requirements.txt`는 PyPI만 보기 때문에 `+cu130` 태그를 찾지 못해 실패 (`numpy` 버전 무시 경고는 이 실패의 백트래킹 과정에서 나온 부수적 로그였음).
- **이 노트북은 NVIDIA GPU가 없으므로 CUDA 빌드 자체가 애초에 무의미함.** `+cuXXX` 태그 없는 CPU 전용 버전(`torch==2.13.0`, `torchvision==0.28.0`)을 PyPI에서 설치해야 함.

## GTX 1650 vs Intel Iris Xe (딥러닝 관점)
- GTX 1650은 저가형이지만 NVIDIA GPU라서 CUDA/cuDNN 지원 → PyTorch GPU 가속 가능.
- Iris Xe는 NVIDIA가 아니므로 드라이버를 아무리 최신으로 깔아도 CUDA가 존재하지 않음 (`torch.cuda.is_available()`이 항상 `False`).
- 대안으로 `torch-directml`(DirectML 기반)이 있으나 성숙도·속도 모두 CUDA에 크게 못 미침.
- CNN 이미지 학습 기준으로 GTX 1650조차 이 노트북(CPU/Iris Xe)보다 체감상 수 배~수십 배 빠름.

## 프로젝트 현황
- `requirements.txt`는 이미 CPU 전용 버전(`torch==2.3.1`, `torchvision==0.18.1` 등, `+cuXXX` 태그 없음)으로 정리되어 있음. 단, 이 노트북 Python 3.11 환경엔 아직 미설치 상태 (`pip list` 확인 결과 pip/setuptools 외 없음) — 작업 시작 전 `py -m pip install -r requirements.txt` 필요.
- 코드가 추가되면 `.cuda()`, `device="cuda"`, `torch.cuda.amp`, `DataParallel` 등 CUDA 하드코딩 여부를 먼저 확인해야 함 — 분기 없이 고정돼 있으면 이 노트북에서 즉시 에러(`RuntimeError: No CUDA GPUs are available`) 발생.

### Data 폴더 구조
- `Data/Training/`, `Data/Validation/` 각각 `01_Source_Data`(원본 이미지 `.png`, 512x512)와 `02_Labeling_Data`(JSON 라벨) 하위 폴더 보유.
- 접두사 규칙: `TS_`(Training Source), `TL_`(Training Labeling), `VS_`(Validation Source), `VL_`(Validation Labeling).
- 클래스 15종, 클래스별 폴더 구조 (`TS_actinic_keratosis` 등) — Training 기준 클래스당 800장.
- **폴더명 한글→영문 변환 완료**: 원래 `TL_광선각화증`처럼 한글로 되어 있던 TL/VS/VL 폴더명을 TS와 동일한 영문 병명으로 전부 통일함 (`Utils/change_folder_name.py`로 처리). 변환 스크립트는 `os.listdir` + 한글→영문 매핑 딕셔너리 + `os.rename` 방식. 참고: `화농_육아종`은 언더스코어(공백 아님)로 통일.
- `01_Source_Data`는 `torchvision.datasets.ImageFolder`로 커스텀 Dataset 없이 바로 로드 가능 (하위 폴더명이 곧 클래스명).
- `02_Labeling_Data`의 JSON은 `diagnosis_name`(한글 병명), `bbox`(병변 위치/크기 좌표) 등을 포함. 병변 크롭 학습 시 이 bbox 활용 예정.

### 다음 목표: ResNet18 기반 피부질환 분류 모델
- **진행 방식**: 사용자가 직접 코드를 작성하고 Claude는 단계별로 코칭/리뷰하는 방식 (Claude가 직접 코드를 짜지 않음).
- **학습 실행 위치**: 이 노트북에서는 파이프라인 개발 및 소규모 데이터로 동작 테스트만 진행. 전체 데이터(Training 12,000장) 학습은 GPU 있는 환경(팀원 PC/Colab/학교 서버)에서 실행 예정.
- **이미지 사용 방식**: 두 가지 버전을 별도 `.py` 파일로 구현 예정.
  - `train_original.py`: 원본 이미지 그대로 사용 (`ImageFolder` 활용, 구현 단순, 1차 baseline)
  - `train_bbox.py`: JSON의 bbox로 병변 부분만 크롭해서 사용 (커스텀 Dataset 클래스 필요)
- 합의된 로드맵: 패키지 설치 → Dataset/ImageFolder → Transform → DataLoader → ResNet18(사전학습 가중치, 마지막 레이어 15클래스로 교체) → Loss/Optimizer → 학습 루프(소규모 테스트) → 검증 루프 → bbox 버전 구현.

## 권장 워크플로우
- 모델 학습(특히 대량 이미지 데이터셋 학습)은 GPU가 있는 환경(팀원 PC, 학교 서버, Colab 등)에서 수행.
- 이 노트북은 코드 작성, 데이터 전처리, 소규모 디버깅/추론 테스트 용도로 사용.
- requirements.txt는 이 노트북용으로 CPU 전용 torch/torchvision 버전으로 별도 관리 (GPU 환경의 requirements.txt와 분리 권장).
