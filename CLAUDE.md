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
- 리포지토리에는 아직 실제 학습/추론 코드(`.py`, `.ipynb`)가 없음. `Data/` 폴더(피부질환 이미지 데이터셋, 예: 광선각화증 등)와 `requirements.txt`만 존재.
- 코드가 추가되면 `.cuda()`, `device="cuda"`, `torch.cuda.amp`, `DataParallel` 등 CUDA 하드코딩 여부를 먼저 확인해야 함 — 분기 없이 고정돼 있으면 이 노트북에서 즉시 에러(`RuntimeError: No CUDA GPUs are available`) 발생.

## 권장 워크플로우
- 모델 학습(특히 대량 이미지 데이터셋 학습)은 GPU가 있는 환경(팀원 PC, 학교 서버, Colab 등)에서 수행.
- 이 노트북은 코드 작성, 데이터 전처리, 소규모 디버깅/추론 테스트 용도로 사용.
- requirements.txt는 이 노트북용으로 CPU 전용 torch/torchvision 버전으로 별도 관리 (GPU 환경의 requirements.txt와 분리 권장).
