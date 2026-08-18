# 개발 환경 노트

## 하드웨어 (이 노트북)
- GPU: Intel(R) Iris(R) Xe Graphics (내장그래픽) — **NVIDIA GPU 없음**
- Python: 3.11.0 (`py -0p`로 확인, `C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe`) — 이건 순정 시스템 파이썬, 패키지 미설치 상태.
- 주의: PATH의 `python` 명령은 Windows Store 스텁을 가리킴. 실제 인터프리터는 `py` 런처로 확인할 것.

## 딥러닝 작업용 Python 환경 (conda `CV`)
- `torch`/`torchvision` 등 `requirements.txt` 패키지는 **conda 가상환경 `CV`**에 설치되어 있음 (`C:\Users\user\anaconda3\envs\CV\python.exe`, Python 3.11.15).
- 확인됨: `torch==2.3.1+cpu`, `torch.cuda.is_available() == False` (예상대로, GPU 없으므로 정상).
- 딥러닝 코드 실행 시 반드시 이 환경 사용: 터미널에서 `conda activate CV` 후 실행, 또는 VS Code에서 인터프리터를 `anaconda3\envs\CV\python.exe`로 선택.
- 시스템 파이썬(`py`)과 conda `CV` 환경은 별개이므로 혼동 주의 — 패키지 설치/확인은 항상 CV 환경 기준으로 할 것.

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

### ResNet18 기반 피부질환 분류 모델
- **진행 방식**: 사용자가 직접 코드를 작성하고 Claude는 단계별로 코칭/리뷰하는 방식 (Claude가 직접 코드를 짜지 않음).
- **학습 실행 위치**: 이 노트북에서는 파이프라인 개발 및 소규모 데이터로 동작 테스트만 진행. 전체 데이터(Training 12,000장) 학습은 GPU 있는 환경(팀원 PC/Colab/학교 서버)에서 실행 예정.
- **이미지 사용 방식**: 두 가지 버전을 별도 `.py` 파일로 구현.
  - `train_original.py` — **완료 및 검증됨**. `ImageFolder`로 원본 이미지 그대로 사용, 공통 모델 정의는 `Models/model.py`(`build_model(num_classes)` 함수, ResNet18 + ImageNet 사전학습 가중치 + fc레이어 15클래스로 교체)로 분리. 전체 Training 데이터(12,000장) 1 epoch, CPU로 약 12분 소요, **Validation Accuracy 96%**로 파이프라인 정상 동작 확인 완료. `wandb`(`wandb 0.28.2`, CV 환경에 설치됨) 연동 완료 — `wandb.init()` + 학습loss/검증accuracy `wandb.log()` 기록 중.
  - `train_bbox.py` — **구현 진행 중**. JSON의 bbox로 병변 부분만 크롭해서 사용. `ImageFolder`를 못 쓰므로 `torch.utils.data.Dataset` 상속받은 커스텀 Dataset 클래스 필요. 라벨(JSON)과 이미지 파일은 **파일명(예: `Z4_24006_P0_L0`)으로 매칭**하고 상위 폴더 접두사만 `TL_`↔`TS_`(`VL_`↔`VS_`) 치환해서 이미지 경로 유추 — JSON 안의 `photograph.file_path`(예: `"사마귀/..."`)는 원 제공처 기준 경로라 우리 폴더 구조와 안 맞으므로 사용하지 않음. JSON의 `bbox` 필드(`xpos`, `ypos`, `width`, `height`)로 `PIL Image.crop()` 수행 예정. `model.py`, 학습/검증 루프, wandb 로깅은 `train_original.py`와 최대한 재사용할 예정.
- 로드맵(완료: 패키지 설치 → ImageFolder → Transform → DataLoader → ResNet18 모델 → Loss/Optimizer → 학습 루프 → 검증 루프 → wandb 연동) → **다음: bbox용 커스텀 Dataset 구현**.

## GPU 환경 세션 기록 (2026-08-18)

### 새 컴퓨터 (GTX 1650, CUDA 가능) 발견
- 이 CLAUDE.md 상단에 기록된 노트북(Intel Iris Xe, conda `CV` 환경)과는 **다른 컴퓨터**에서 작업 진행함. `nvidia-smi`로 **NVIDIA GeForce GTX 1650** 확인됨 (CUDA 사용 가능).
- 이 컴퓨터엔 conda 자체가 없음 (`conda` 명령 없음). 시스템 Python 3.12(`C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe`, PATH의 `python`이 이걸 가리킴)에 pip으로 직접 패키지 설치.
- `requirements.txt`는 CPU 전용 태그 없는 버전(`torch==2.3.1` 등)이지만, GPU를 실제로 쓰려면 PyTorch 공식 인덱스에서 CUDA 빌드를 따로 설치해야 함:
  ```
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
  → 설치 결과: `torch==2.5.1+cu121`, `torch.cuda.is_available() == True`, `torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1650"`.
- `wandb`는 `requirements.txt`에 없지만 `train_original.py`가 사용하므로 별도 설치 필요 (`pip install wandb`). 이 컴퓨터에서 `wandb login <API key>`로 로그인 완료, 프로젝트: https://wandb.ai/bll3479-/DL_temaPJ_SKIN (계정 `bll3479`).

### GPU 학습 시 코드에 device 처리 필요
- `train_original.py` 원본에는 `.to(device)`가 전혀 없어서, GPU 환경이어도 기본 상태로는 그냥 CPU로 돌아감.
- GPU를 실제로 쓰려면 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`를 만들고 `model`과 각 배치의 `images`/`labels`를 `.to(device)`로 옮기는 코드가 필요함 (현재 파일엔 반영 안 돼 있음 — 다시 GPU로 돌릴 때 매번 추가해야 함).
- 검증(validation)도 원본 구조는 전체 epoch 학습이 끝난 뒤 딱 한 번만 실행됨 → epoch별로 val accuracy를 비교하려면 검증 루프를 epoch for문 안으로 옮기고 `wandb.log`도 `{"epoch", "train_loss", "val_accuracy"}`로 합쳐서 한 번에 기록해야 함 (몇 epoch이 최적인지 판단하려면 필수).

### Epoch 튜닝 실험 결과 (전체 Training 12,000장 / 전체 Validation, GTX 1650, ResNet18 전이학습, lr=0.001, 5 epoch)
- 실행 1회차 (wandb 미연결, 콘솔 로그만):

  | Epoch | Train Loss | Val Accuracy |
  |---|---|---|
  | 1 | 0.2231 | 97.33% |
  | 2 | 0.1061 | 97.00% |
  | 3 | 0.0694 | 98.13% (peak) |
  | 4 | 0.0649 | 97.87% |
  | 5 | 0.0599 | 96.80% |

- 실행 2회차 (wandb 연결, run: `ResNet18-original` / `4aar6mm4`):

  | Epoch | Train Loss | Val Accuracy |
  |---|---|---|
  | 1 | 0.2422 | 80.00% |
  | 2 | 0.0924 | 95.53% |
  | 3 | 0.0829 | 97.53% |
  | 4 | 0.0585 | 97.73% |
  | 5 | 0.0572 | 99.47% (peak) |

- **결론**: 합성데이터 특성상(클래스 내 변동성 낮음) 3~5 epoch 안에 val accuracy가 97~99%대로 수렴함. 실행마다(랜덤 초기화/셔플) 어느 epoch이 peak인지는 달라지지만(3 epoch vs 5 epoch), 5 epoch이면 baseline 성능은 충분히 확보됨. 추가 개선이 필요하면 epoch 수보다 augmentation/backbone 비교에 투자하는 게 나음.
- 참고: `num_epochs`는 실험 후 `5`로 남아있고, device 처리(`.to(device)`) 및 epoch별 검증 구조는 실험 이후 원본 구조(학습 전 이 CLAUDE.md 파일 상단에 설명된, 검증이 루프 밖에서 한 번만 도는 구조)로 되돌아간 상태 — 다음에 GPU로 epoch별 비교 실험을 다시 하려면 위 "GPU 학습 시 코드에 device 처리 필요" 내용을 다시 적용해야 함.

## 딥러닝 코드 실행 관련 팁
- 스크립트를 `Utils/`나 다른 하위 폴더로 옮겨서 실행할 경우, 코드 안 상대경로(`r'./Data/...'`)는 **스크립트 실행 시점의 현재 작업 디렉토리** 기준이라는 점 주의 (스크립트 파일 위치 기준이 아님). 프로젝트 루트에서 실행해야 정상 작동.
- `.py` 파일을 터미널에서 실행할 때는 Jupyter/REPL과 달리 `print()` 없이 값만 써두면 화면에 아무것도 안 뜸.

## 권장 워크플로우
- 모델 학습(특히 대량 이미지 데이터셋 학습)은 GPU가 있는 환경(팀원 PC, 학교 서버, Colab 등)에서 수행.
- 이 노트북은 코드 작성, 데이터 전처리, 소규모 디버깅/추론 테스트 용도로 사용.
- requirements.txt는 이 노트북용으로 CPU 전용 torch/torchvision 버전으로 별도 관리 (GPU 환경의 requirements.txt와 분리 권장).
