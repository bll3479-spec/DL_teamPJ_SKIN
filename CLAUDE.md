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
- **이미지 사용 방식**: 원본 이미지 그대로 사용하는 방식으로 확정 (bbox 크롭 버전은 검토했으나 불필요 판단, 폐기). 파일명도 원래 `train_original.py`였으나 bbox 버전(`train_bbox.py`)이 없어지면서 `train.py`로 변경함.
  - `train.py` — 파이프라인 자체는 **완료 및 검증됨**. `ImageFolder`로 원본 이미지 그대로 사용, 공통 모델 정의는 `Models/model.py`(`build_model(num_classes)` 함수, ResNet18 + ImageNet 사전학습 가중치 + fc레이어 15클래스로 교체)로 분리. 전체 Training 데이터(12,000장) 1 epoch, CPU로 약 12분 소요, **Validation Accuracy 96%**로 파이프라인 정상 동작 확인. `wandb`(`wandb 0.28.2`, CV 환경에 설치됨) 연동 완료 — `wandb.init()` + 학습loss/검증accuracy `wandb.log()` 기록.
  - **가중치 저장 코드 추가함**: `os.makedirs('checkpoints', exist_ok=True)` + `torch.save(model.state_dict(), './checkpoints/resnet18_original.pth')` (처음엔 `'chechpoints'` 오타 및 저장 경로 누락 버그 있었으나 수정 완료). **단, 이 노트북에서 실행해서 파일 생성까지 확인하지는 않은 상태** — 사용자가 다른(GPU) 환경에서 실행할 예정이라 이 노트북에는 `checkpoints/` 폴더가 아직 없음.
- 로드맵(완료: 패키지 설치 → ImageFolder → Transform → DataLoader → ResNet18 모델 → Loss/Optimizer → 학습 루프 → 검증 루프 → wandb 연동 → 가중치 저장 코드 작성) → **다음: 다른(GPU) 환경에서 실제 실행 — 여러 epoch 본 학습 + 가중치 저장 확인**.

## GPU 환경 세션 기록 (2026-08-18)

### 새 컴퓨터 (GTX 1650, CUDA 가능) 발견
- 이 CLAUDE.md 상단에 기록된 노트북(Intel Iris Xe, conda `CV` 환경)과는 **다른 컴퓨터**에서 작업 진행함. `nvidia-smi`로 **NVIDIA GeForce GTX 1650** 확인됨 (CUDA 사용 가능).
- (2026-08-18 기준) 이 컴퓨터엔 conda 자체가 없었음 (`conda` 명령 없음). 시스템 Python 3.12(`C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe`, PATH의 `python`이 이걸 가리킴)에 pip으로 직접 패키지 설치.
- **(2026-08-19 업데이트, 정정)** 이후 이 컴퓨터에도 **conda `CV` 환경이 생성됨** (`C:\ProgramData\anaconda3\envs\CV\python.exe`). `requirement_me.txt`(`torch==2.3.1`, `+cu` 태그 없음)를 이 환경에 `pip install -r`로 설치했더니 **CPU 전용 빌드**가 깔려서 `torch.cuda.is_available()==False`가 됨 — 학습 스크립트를 이 환경으로 실행하면 GPU가 있어도 CPU로 돎 (실제로 한 번 이 상태로 15 epoch 학습이 CPU로 돌고 있었던 걸 발견해서 중간에 종료함). CUDA 빌드로 재설치:
  ```
  "C:\ProgramData\anaconda3\envs\CV\python.exe" -m pip uninstall torch torchvision -y
  "C:\ProgramData\anaconda3\envs\CV\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
  재설치 후 `torch==2.5.1+cu121`, `cuda_available=True` 확인됨. **이 노트북 시스템 Python(3.12, PATH 기준)과 conda `CV` 환경은 별개의 site-packages를 쓰므로, GPU 학습 시 반드시 어떤 인터프리터로 실행 중인지(`python ./train.py`를 어느 python.exe가 실행하는지) 확인 필요** — 헷갈리기 매우 쉬움 (겉보기엔 같은 `python` 명령이라 구분이 안 됨, `tasklist`/`wmic process ... get CommandLine,ExecutablePath`로 실제 실행 파일 경로를 확인해야 구분 가능).
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

## 세션 기록 (2026-08-19) — 파이프라인 모듈화, 버그 수정, 평가지표 확장

### 작업 방식 관련 (중요)
- 이 세션 초반에 Claude가 코드를 직접 작성/수정한 적이 있었는데, 사용자가 즉시 강하게 지적함 ("왜 너가 코드 짜 코드는 내가 입력할거야 다시 원상복구해") → 바로 원복함. **이 프로젝트의 "코드는 사용자가 직접 작성, Claude는 코칭/리뷰만" 원칙(이 문서 상단 ResNet18 섹션에 이미 명시)이 재확인됨.** 이후 세션 전체에서 Claude는 코드를 직접 고치지 않고, 어느 줄에 뭘 어떻게 고칠지 구체적으로 짚어주는 방식으로만 진행함 (CLAUDE.md 갱신처럼 사용자가 명시적으로 요청한 문서 작업은 예외).

### GPU 환경에서 CPU로 학습되고 있던 문제 발견 및 해결
- `train.py`를 GPU 머신에서 실행했는데 `nvidia-smi` 확인 결과 GPU 사용률이 계속 4~5%, 프로세스 목록에도 안 잡힘 → 실제로는 conda `CV` 환경(`C:\ProgramData\anaconda3\envs\CV\python.exe`)의 CPU 전용 torch로 돌고 있었음. 위 "새 컴퓨터" 섹션에 정정 기록. CUDA 빌드 재설치 후 `nvidia-smi dmon`으로 클럭이 idle(300MHz)에서 boost(1800MHz대)로 올라가는 것까지 확인해서 GPU 사용 확정.
- 학습 속도 추정: wandb `output.log`에서 tqdm 진행률(예: 202/375 batches, 2.11 it/s) 읽어서 전체 소요시간 역산하는 방법 사용 — batch 수 × epoch 수 ÷ it/s로 계산.

### 코드 구조 리팩토링 (진행자: 사용자, Claude는 리뷰만)
기존에 `train.py` 하나에 다 있던 코드를 아래처럼 모듈화함:
- `Utils/dataloader.py` — `get_dataloaders(train_dir, val_dir, batch_size, num_workers=4)`: transform 정의 + `ImageFolder` + `DataLoader` 생성, `(train_loader, val_loader)` 반환
- `evaluate.py` — `evaluate(model, val_loader, device, epoch, top_checkpoints, checkpoint_dir)`: 검증 루프 + accuracy/f1 계산 + top-3 체크포인트 저장/삭제 관리, `(accuracy, f1)` 반환
- `train.py` — `train_one_epoch(model, train_loader, criterion, optimizer, device)`(1 epoch 학습, avg_loss 반환) + `fit(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, checkpoint_dir)`(epoch 루프 오케스트레이션, `train_one_epoch`+`evaluate` 호출, wandb 로깅). **함수 정의만 있고 최상위 실행 코드는 없음** (import해도 부수효과 없음)
- `Models/model.py` — 기존 `bulid_model(num_classes)` 그대로
- `main.py` — 실제 진입점. `if __name__ == '__main__':` 블록 안에서 `wandb.init()` → `get_dataloaders()` → `bulid_model()`/`device`/`criterion`/`optimizer` 생성 → `fit()` 호출까지 조립

리팩토링 중 반복적으로 나왔던 버그 패턴(다음에 비슷한 리팩토링할 때 참고):
- **함수가 파라미터를 받아놓고 본문에서 하드코딩된 값으로 덮어쓰거나 무시하는 패턴**이 여러 번 나옴 (`get_dataloaders`의 `train_dir`/`val_dir`/`batch_size`, `evaluate`의 `checkpoint_dir`, `Utils/dataloader.py`의 `num_workers`가 각각 한 번씩 이 패턴으로 버그였음) — 파라미터 추가할 때 본문에서 실제로 그 이름을 쓰는지 항상 재확인 필요
- **`os.makedirs`가 실제 저장 시점보다 늦게(또는 루프 밖) 호출**되어 `FileNotFoundError` 나는 패턴이 2번 발생 — 폴더 생성은 첫 저장 이전, 루프 진입 전에 위치해야 함
- **함수 리팩토링 후 반환값 형태가 바뀌었는데 호출부가 그대로인 경우**(`get_dataloaders`에 `return` 누락, `evaluate`가 나중에 `accuracy` 하나에서 `(accuracy, f1)` 튜플로 바뀌었는데 `train.py`가 처음엔 언패킹 안 함) — 함수 시그니처/반환값 바꿀 때마다 모든 호출부를 같이 확인해야 함
- **모듈을 함수 전용으로 만들다가 최상위 실행 코드(`wandb.init`, 모델 생성, 함수 호출)를 못 옮기고 남겨둬서, 다른 파일에서 `import`할 때 그 코드가 부수효과로 실행돼버리는 문제**가 있었음 — `train.py`를 최종적으로 "함수 정의만 있는 모듈"로 정리하고 나서야 해결
- `Windows`에서 `torch.device` 오탈자, `f'./checkpoints/resent18...'`처럼 `resnet18`→`resent18` 오타, `val_dir`을 실수로 `Training` 경로로 복붙, `avg_loss`를 `train_one_epoch` 반환값(이미 평균) 위에서 `fit()`이 또 `len(train_loader)`로 나눠서 이중 나눗셈되는 버그, `if __name__ == ' __main__':`처럼 문자열에 공백이 섞여 조건이 항상 False가 되는 오타 등도 발견해서 수정함

### 평가지표: accuracy → accuracy + F1(macro) 추가
- `evaluate.py`에 `sklearn.metrics`의 `accuracy_score`, `f1_score(average='macro')` 추가. 클래스가 완전 균등(Training 800장/클래스, Validation 100장/클래스)이라 accuracy와 macro F1이 실제로는 비슷하게 움직이지만, 클래스 불균형에 더 강건한 지표라 **top-3 체크포인트 랭킹 기준을 f1으로 통일**하기로 결정함 (accuracy도 wandb 로그·파일명엔 같이 남김).
- 두 지표를 동시에 랭킹 기준으로 쓰면 "어느 게 더 나은 epoch인지" 판단이 모호해지므로, 랭킹 기준은 항상 하나로 고정하는 게 낫다는 결론 (참고용 지표는 여러 개 로깅해도 무방).

### AI Hub 원본 데이터셋(전체 15,000장, 8:1:1 분할) baseline과 비교
- 출처: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71864 ("피부종양 이미지 합성 데이터", 15종, 클래스당 1,000장)
- AI Hub 제시 baseline: ResNet101(15종 분류, image_size=256, epoch=10, dropout=0.5), EfficientNet-B3(양성/악성 이진분류, image_size=256, epoch=16, dropout=0.3) — 둘 다 batch_size=32, lr=0.001, Adam, 8:1:1 분할
- 우리 프로젝트는 ResNet18(더 경량), image_size=224, dropout 없음, train/val 2분할만(test set 없음) 사용 — 저사양 환경 제약 때문으로 추정. batch_size/lr/optimizer는 baseline과 동일.

### 현재 프로젝트의 Data 폴더 상세 구조 (재확인)
```
Data/Training/01_Source_Data/TS_<class>/   각 800장 (15클래스 × 800 = 12,000장)
Data/Training/02_Labeling_Data/TL_<class>/ (이미지와 1:1 대응 JSON)
Data/Validation/01_Source_Data/VS_<class>/ 각 100장 (15클래스 × 100 = 1,500장)
Data/Validation/02_Labeling_Data/VL_<class>/
```
- Training:Validation 클래스당 비율 8:1로 균등, `ImageFolder` 구조에 그대로 부합

### predict.py는 아직 미구현 (선택사항으로 결론)
- 현재 `evaluate()`가 학습 루프 안에서 이미 예측(argmax)과 top-3 가중치 저장을 수행하므로, "학습 + 좋은 가중치 확보"라는 목적 자체는 이미 충족됨.
- 별도 `predict.py`는 ① 저장된 `.pth`를 실제로 다시 로드해도 잘 동작하는지 검증(왕복 테스트, 아직 한 번도 안 해봄), ② confusion matrix 등 accuracy/f1 숫자 이상의 세부 진단 정보 확보 용도로 필요할 때 추가하기로 함. 급하지 않음.
- Validation 셋을 학습 중 top-3 체크포인트 선택에도 이미 쓰고 있어서, 같은 셋으로 predict까지 하면 완전히 독립적인 최종 평가는 아님(선택 편향 가능성) — 별도 test 셋은 없는 상태(AI Hub 원본은 8:1:1이었지만 이 프로젝트는 2분할로 확정한 것, 위 참고).

## 딥러닝 코드 실행 관련 팁
- 스크립트를 `Utils/`나 다른 하위 폴더로 옮겨서 실행할 경우, 코드 안 상대경로(`r'./Data/...'`)는 **스크립트 실행 시점의 현재 작업 디렉토리** 기준이라는 점 주의 (스크립트 파일 위치 기준이 아님). 프로젝트 루트에서 실행해야 정상 작동.
- `.py` 파일을 터미널에서 실행할 때는 Jupyter/REPL과 달리 `print()` 없이 값만 써두면 화면에 아무것도 안 뜸.

## 권장 워크플로우
- 모델 학습(특히 대량 이미지 데이터셋 학습)은 GPU가 있는 환경(팀원 PC, 학교 서버, Colab 등)에서 수행.
- 이 노트북은 코드 작성, 데이터 전처리, 소규모 디버깅/추론 테스트 용도로 사용.
- requirements.txt는 이 노트북용으로 CPU 전용 torch/torchvision 버전으로 별도 관리 (GPU 환경의 requirements.txt와 분리 권장).
