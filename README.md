# ETA 기반 의사결정을 통한 다중 GPU 지식 증류 파이프라인에서의 장애 대응 기법
### 이화여자대학교 컴퓨터공학과 졸업프로젝트 23팀 문원정, 최지희, 김현영

Synapse는 TSPipe 기반의 지식 증류(Knowledge Distillation, KD) 파이프라인 학습 코드를 확장하여, GPU 성능 저하 및 장애 상황에서의 failover, 재시작, 재분할 정책을 실험하기 위한 연구용 프로젝트입니다.

기본 TSPipe 구조 위에 다음 기능을 추가해 실험합니다.

- profiling 기반 모델 파티셔닝 및 stage time 예측
- GPU health monitoring 및 slowdown detection
- 장애 발생 시 checkpoint 저장, K-1 GPU 재시작, partition replan/degrade 정책
- 실험 로그, 성능 지표, 그래프 생성

## 프로젝트 설명

이 프로젝트는 원본 TSPipe의 pipeline parallel training 구조를 기반으로 합니다. TSPipe는 teacher network와 student network의 실행을 파이프라인으로 스케줄링하여 KD 학습 처리량을 높이는 프레임워크입니다.

현재 저장소에서는 여기에 GPU 장애 대응 실험을 위한 기능을 추가했습니다.

- `KEEP`: 약한 성능 저하에서는 현재 파티션 유지
- `REPLAN`: 지속적인 성능 저하에서는 GPU별 stage time을 고려해 파티션 재계산
- `DEGRADE`: GPU 장애 또는 심각한 성능 저하에서는 해당 GPU를 제외하고 K-1개 GPU로 재시작

주요 실험 대상은 ImageNet100 기반 KD 학습이며, teacher/student 모델 예시는 `vit_large`, `resnet152`, `resnet50`, `vit_base` 등을 포함합니다. Failover 실험은 정상 TSPipe 실행과 slowdown/failure injection이 있는 실행을 비교하여, 재분할 및 재시작 정책이 전체 ETA와 처리량에 미치는 영향을 관찰합니다.

## 구현 구성 블록

<table>
  <tr>
    <td align="center" width="25%">
      <b>Runtime</b><br/>
      <img src="https://img.shields.io/badge/Python%203.9-3776AB?style=plastic&logo=python&logoColor=white"/><br/>
      <img src="https://img.shields.io/badge/CUDA-Multi%20GPU-76B900?style=plastic&logo=nvidia&logoColor=white"/>
    </td>
    <td align="center" width="25%">
      <b>KD Training</b><br/>
      <img src="https://img.shields.io/badge/PyTorch-Teacher--Student-EE4C2C?style=plastic&logo=pytorch&logoColor=white"/><br/>
      <img src="https://img.shields.io/badge/timm%20%7C%20Transformers-Model%20Zoo-111827?style=plastic"/>
    </td>
    <td align="center" width="25%">
      <b>Pipeline Runtime</b><br/>
      <img src="https://img.shields.io/badge/TSPipe-Extended%20Runtime-2D3748?style=plastic"/><br/>
      <img src="https://img.shields.io/badge/Pipeline%20Parallelism-KD%20Scheduling-4B5563?style=plastic"/>
    </td>
    <td align="center" width="25%">
      <b>Failover Policy</b><br/>
      <img src="https://img.shields.io/badge/ETA%20Planner-Policy%20Selector-2563EB?style=plastic"/><br/>
      <img src="https://img.shields.io/badge/KEEP%20%7C%20REPLAN%20%7C%20DEGRADE-Failover-DC2626?style=plastic"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Monitoring</b><br/>
      <img src="https://img.shields.io/badge/PyNVML-GPU%20Health-76B900?style=plastic&logo=nvidia&logoColor=white"/><br/>
      <img src="https://img.shields.io/badge/nvidia--smi-GPU%20Status-111827?style=plastic"/>
    </td>
    <td align="center">
      <b>Config & Logs</b><br/>
      <img src="https://img.shields.io/badge/PyYAML-Config-4B5563?style=plastic"/><br/>
      <img src="https://img.shields.io/badge/JSON%20%7C%20JSONL-Experiment%20Logs-6B7280?style=plastic"/>
    </td>
    <td align="center">
      <b>Analysis</b><br/>
      <img src="https://img.shields.io/badge/NumPy-Numerical%20Analysis-013243?style=plastic&logo=numpy&logoColor=white"/><br/>
      <img src="https://img.shields.io/badge/pandas-Result%20Summary-150458?style=plastic&logo=pandas&logoColor=white"/>
    </td>
    <td align="center">
      <b>Visualization</b><br/>
      <img src="https://img.shields.io/badge/matplotlib-Graphs-11557C?style=plastic"/><br/>
      <img src="https://img.shields.io/badge/TensorBoard-Training%20Logs-FF6F00?style=plastic"/>
    </td>
  </tr>
</table>

## Source Code 설명

### 핵심 디렉터리

```text
.
├── tspipe/                         # TSPipe 런타임 및 failover 확장 구현
├── benchmarks/soft_target/         # KD 학습 스크립트, 모델, loss, planner
├── dataset/                        # ImageNet100 등 데이터셋 로더 및 class 목록
├── scripts/                        # 실험 결과 분석 및 plotting 유틸리티
├── analysis/                       # 실험 그래프 및 분석 산출물
├── results/                        # E2E 실험 결과 예시 및 로그
├── failover_logs/                  # failover 실험 로그
├── failover_results*/              # failover 실험 결과 및 시각화 파일
└── test_*.py, test_*.sh             # failover/planner 회귀 테스트
```

### TSPipe runtime 및 failover 확장

- `tspipe/tspipe.py`: TSPipe 메인 클래스입니다. YAML partition 설정을 읽고, worker 초기화, checkpoint 저장, failover restart config 처리, GPU/partition mismatch 보정, failover logger 연동을 수행합니다.
- `tspipe/gpu_worker.py`: GPU worker 실행 및 task 처리 로직입니다. compute/communication task, profiling hook, slowdown injection 경로가 포함됩니다.
- `tspipe/gpu_task.py`: GPU task type, forward/backward/communication task 실행 단위를 정의합니다.
- `tspipe/scheduler.py`: 파이프라인 스케줄링 및 동적 재분할 관련 로직을 담당합니다.
- `tspipe/communicator.py`: TSPipe 내부 worker 간 queue/channel 통신을 담당합니다.
- `tspipe/profiler.py`, `tspipe/profiler_utils.py`: GPU task timing, NVTX/profiling log, partition별 task summary 생성을 담당합니다.
- `tspipe/gpu_health_monitor.py`: CUDA context test, process 상태 확인, forced failure injection 등을 통해 GPU 장애 이벤트를 생성합니다.
- `tspipe/slowdown_detector.py`: 최근 stage time window와 baseline을 비교하여 slowdown ratio를 계산하고 sustained slowdown 여부를 판단합니다.
- `tspipe/failover_logger.py`: failover event, performance metric, experiment summary를 JSON/JSONL로 기록합니다.

### KD benchmark 코드

- `benchmarks/soft_target/train_base.py`: teacher 또는 baseline 모델을 학습/초기화합니다. 내부에서 PyTorch DDP를 초기화하므로 `torchrun`으로 실행해야 합니다.
- `benchmarks/soft_target/train_kd.py`: KD 학습 및 TSPipe/failover 실험의 주 실행 파일입니다. `--tspipe-enable`, slowdown injection, failover bootstrap, restart resume, healthy checkpoint 저장 등을 처리합니다.
- `benchmarks/soft_target/train_kd_profiling.py`: profiling 중심 KD 실행 파일입니다. `--prepare-planner`로 planner alpha/beta metadata 생성을 수행할 수 있습니다.
- `benchmarks/soft_target/tspipe.yaml`: 기본 4 GPU partition 설정입니다.
- `benchmarks/soft_target/tspipe_restart_kminus1.yaml`: GPU 감소 후 restart partition 예시입니다.
- `benchmarks/soft_target/models/`: ViT, ResNet, EfficientNet, DeiT wrapper 및 model factory입니다.
- `benchmarks/soft_target/kd_losses/`: logits KD와 soft target KD loss 구현입니다.
- `benchmarks/soft_target/utils.py`: metric, checkpoint 저장, pretrained model loading, parameter count helper입니다.

### Planner 세부 모듈

- `benchmarks/soft_target/planner/stage_time_predictor.py`: `snet.csv`, `tnet.csv`를 읽어 각 GPU stage의 bottleneck time을 예측합니다.
- `benchmarks/soft_target/planner/mathematical_optimizer.py`: ETA 기반 failover decision의 중심 모듈입니다. 현재 partition, runtime alpha/beta, checkpoint/restart 상태를 관리합니다.
- `benchmarks/soft_target/planner/dynamic_policy_selector.py`: KEEP/REPLAN/DEGRADE 중 어떤 정책이 남은 학습 시간 관점에서 유리한지 선택합니다.
- `benchmarks/soft_target/planner/eta_calculator.py`: restart cost와 remaining step 수를 이용해 정책별 ETA를 계산합니다.
- `benchmarks/soft_target/planner/dynamic_alpha_beta_estimator.py`: runtime timing batch에서 GPU별 compute/communication slowdown ratio를 갱신합니다.
- `benchmarks/soft_target/planner/restart_cost_benchmark.py`: checkpoint save/load 및 restart cost를 측정합니다.
- `benchmarks/soft_target/planner/profile/snet.csv`: student network layer-wise profile입니다.
- `benchmarks/soft_target/planner/profile/tnet.csv`: teacher network layer-wise profile입니다.

### 데이터 로더 및 모델 코드

- `dataset/datasets.py`: `ImageNet100` class가 `dataset/imagenet100.txt`를 읽어 ImageNet 폴더에서 100개 class만 필터링합니다.
- `dataset/loader.py`: two-crop transform, blur, solarization 등 augmentation helper를 포함합니다.
- `benchmarks/soft_target/models/factory.py`: `vit`, `resnet`, `efficientnet`, `deit` 이름에 따라 모델을 생성합니다.

### 실행/분석 스크립트

- `run_e2e_failover.sh`: failover 발생 시 exit code `42`를 감지하여 재시작 루프를 관리합니다. restart config가 있으면 `CUDA_VISIBLE_DEVICES`와 partition 수를 갱신합니다.
- `e2e_failover_batch*.sh`: batch size별 failover E2E 실험 스크립트입니다.
- `e2e_tspipe_batch*.sh`: batch size별 기본 TSPipe 비교 실험 스크립트입니다.
- `run_failover_experiment.py`: `basic`, `advanced`, `profiling_overhead` failover 실험을 관리합니다.
- `run_experiment0_checkpoint.py`: checkpoint interval benchmark입니다.
- `run_experiment0b_failover.py`: failure injection benchmark입니다.
- `scripts/*.py`: plotting, summary, background load, figure generation helper입니다.

## How to Build

이 프로젝트는 별도의 컴파일 또는 빌드 과정이 필요한 패키지가 아니라 Python 기반 연구 코드입니다. 따라서 일반적인 build 단계는 환경 구성과 의존성 설치로 대체됩니다.

권장 환경은 다음과 같습니다.

- Python 3.9 계열
- CUDA 사용 가능 GPU 환경
- PyTorch + torchvision
- NVIDIA GPU 상태 확인을 위한 `nvidia-smi`
- 단일 노드 다중 GPU 실험 기준 4개 이상의 GPU 권장

설치 후 build 검증은 다음처럼 수행할 수 있습니다.

```bash
conda activate tspipe

# 기존 __pycache__ 권한 문제를 피하려면 임시 pycache 경로 사용
PYTHONPYCACHEPREFIX=/tmp/synapse_pycache python -m compileall -q tspipe dataset benchmarks/soft_target

# planner 필수 입력 파일 확인
test -f benchmarks/soft_target/planner/profile/snet.csv
test -f benchmarks/soft_target/planner/profile/tnet.csv
test -f dataset/imagenet100.txt
```

`compileall`은 Python syntax 수준의 정적 확인입니다. 실제 GPU training 가능 여부는 `How to Test`의 E2E smoke test로 확인합니다.

## How to Install

### 1. Conda 환경 생성

```bash
conda create -n tspipe python=3.9 -y
conda activate tspipe
```

### 2. 주요 Python 패키지 설치

현재 저장소에는 실행 당시 환경을 기록한 `benchmarks/soft_target/requirements_snapshot.txt`가 포함되어 있습니다. 이 파일은 conda build 경로가 포함된 스냅샷이므로, 새 환경에서는 아래처럼 핵심 패키지를 직접 설치하는 방식을 권장합니다.

```bash
pip install torch torchvision torchaudio
pip install numpy pandas pyyaml tqdm tensorboard tensorboardX psutil GitPython
pip install matplotlib pillow timm==0.4.9 transformers==4.33.3 nvidia-ml-py
```

CUDA 버전에 맞는 PyTorch 설치 명령은 실행 환경에 맞게 조정해야 합니다. 예를 들어 CUDA 12.4 wheel이 필요한 환경에서는 PyTorch 공식 index URL을 사용해 설치합니다.

### 3. PYTHONPATH 설정

저장소 루트에서 실행할 경우 대부분의 스크립트는 상대 import로 동작합니다. 필요하면 다음처럼 루트를 `PYTHONPATH`에 추가합니다.

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### 4. 시스템 설정

멀티 GPU/멀티 프로세스 실험에서는 open file limit을 충분히 크게 설정하는 것이 좋습니다.

```bash
ulimit -n 409600
```

단일 노드에서 NCCL 통신 문제를 줄이기 위해 실험 스크립트들은 다음 환경 변수를 사용합니다.

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
```

## How to Run

### 1. 데이터 준비

ImageNet100 실험은 원본 ImageNet 이미지가 저장소에 포함되어 있지 않으므로, 아래 링크에서 데이터셋을 다운받은 후 로컬 ImageNet 경로를 준비해야 합니다.

다운 링크 : https://www.image-net.org/download.php

예상 구조:

```text
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

실행 시:

```bash
export IMG_ROOT=/path/to/imagenet
```

`dataset/imagenet100.txt`에 포함된 synset만 사용됩니다.

### 2. Teacher/Student checkpoint 준비

KD 실험에는 teacher checkpoint와 student 초기 checkpoint가 필요합니다.

Teacher 예시:

```bash
cd benchmarks/soft_target
torchrun --nproc_per_node=4 train_base.py \
  --img_root "$IMG_ROOT" \
  --save_root ./results/base/ \
  --epochs 20 \
  --data_name imagenet100 \
  --net_name vit_large \
  --num_class 100 \
  --batch_size 128 \
  --note base-i100-vit-large
```

Student 초기 checkpoint 예시:

```bash
cd benchmarks/soft_target
torchrun --nproc_per_node=4 train_base.py \
  --img_root "$IMG_ROOT" \
  --save_root ./results/base/ \
  --epochs 0 \
  --data_name imagenet100 \
  --net_name resnet152 \
  --num_class 100 \
  --batch_size 128 \
  --note base-i100-resnet152
```

주의: `train_base.py`는 `torch.distributed.init_process_group()`을 호출하므로 일반 `python train_base.py`가 아니라 `torchrun`으로 실행하는 것이 맞습니다.

### 3. TSPipe KD 실행

`run_e2e_failover.sh`는 내부적으로 `benchmarks/soft_target/train_kd.py`를 실행합니다. `train_kd.py` 자체가 TSPipe worker/NCCL/RPC를 관리하므로 E2E launcher는 `torchrun`이 아니라 단일 `python` 프로세스를 사용합니다.

```bash
cd /acpl-ssd10/Synapse-0325
export CUDA_VISIBLE_DEVICES=0,1,2,3
export BASE_SAVE_ROOT=./results
export RUN_NOTE=kd_tspipe_demo

bash ./run_e2e_failover.sh \
  --img_root "$IMG_ROOT" \
  --data_name imagenet100 \
  --t_name vit_large \
  --s_name resnet152 \
  --kd_mode st \
  --lambda_kd 0.1 \
  --T 4.0 \
  --t_model /path/to/teacher/model_best.pth.tar \
  --s_init /path/to/student/initial_r152.pth.tar \
  --batch_size 64 \
  --num_class 100 \
  --epochs 1 \
  --max-steps-per-epoch 50 \
  --tspipe-enable \
  --tspipe-config benchmarks/soft_target/tspipe.yaml
```

### 4. Failover-aware E2E 실행

아래 예시는 특정 GPU에 compute slowdown을 주입하고, policy가 REPLAN/DEGRADE를 선택하면 exit code `42`를 통해 launcher가 재시작하는 흐름을 확인합니다.

```bash
export DEFAULT_VISIBLE_GPUS=0,1,2,3
export BASE_SAVE_ROOT=./results
export RUN_NOTE=e2e_failover_demo
export MAX_RESTARTS=2
export FAILOVER_SLOWDOWN_THRESHOLD_SEC=10.0

bash ./run_e2e_failover.sh \
  --img_root "$IMG_ROOT" \
  --data_name imagenet100 \
  --t_name vit_large \
  --s_name resnet152 \
  --kd_mode st \
  --lambda_kd 0.1 \
  --T 4.0 \
  --t_model /path/to/teacher/model_best.pth.tar \
  --s_init /path/to/student/initial_r152.pth.tar \
  --batch_size 64 \
  --num_class 100 \
  --epochs 1 \
  --max-steps-per-epoch 100 \
  --tspipe-enable \
  --tspipe-config benchmarks/soft_target/tspipe.yaml \
  --inject-slowdown-gpu 0 \
  --slowdown-task-scope compute \
  --slowdown-mode fixed \
  --slowdown-fixed-ms 800 \
  --slowdown-start 20 \
  --slowdown-end 100
```

Batch size별 wrapper도 포함되어 있습니다.

```bash
bash e2e_failover_batch64.sh
bash e2e_tspipe_batch64.sh
```

단, 이 스크립트들에는 특정 서버 기준의 데이터/checkpoint 경로가 포함되어 있으므로 실행 환경에 맞게 경로를 수정해야 합니다.

## How to Test

### 1. Planner 회귀 테스트

GPU 없이도 failover policy의 핵심 흐름을 확인할 수 있는 테스트입니다.

```bash
python test_minimal_failover.py
```

이 테스트는 `KEEP -> REPLAN -> DEGRADE` 정책 전환과 degraded partition에서 장애 GPU가 제외되는지 확인합니다.

### 2. Restart monitoring 회귀 테스트

재시작 후 optimizer baseline과 slowdown detector baseline이 올바르게 복원되는지 확인합니다.

```bash
python test_failover_restart_monitoring.py
```

### 3. Failover 로직 테스트

Mock TSPipe 객체를 사용해 GPU failure event, emergency checkpoint 저장, restart config 생성을 검증합니다.

```bash
python test_failover_logic.py --target-gpu 4
```

이 테스트는 실제 학습 전체를 돌리지는 않지만, `tspipe.gpu_health_monitor`와 `tspipe.failover_logger`를 사용합니다. CUDA/GPU 환경에 따라 결과가 달라질 수 있습니다.

### 4. E2E failover smoke test

실제 KD 학습과 failover restart loop를 함께 실행하려면 teacher/student checkpoint와 ImageNet 경로가 필요합니다. 빠른 smoke test는 `--max-steps-per-epoch`를 작게 지정합니다.

```bash
export DEFAULT_VISIBLE_GPUS=0,1,2,3
export RUN_NOTE=e2e_failover_smoke

bash ./run_e2e_failover.sh \
  --img_root "$IMG_ROOT" \
  --data_name imagenet100 \
  --t_name vit_large \
  --s_name resnet152 \
  --kd_mode st \
  --lambda_kd 0.1 \
  --t_model /path/to/teacher/model_best.pth.tar \
  --s_init /path/to/student/initial.pth.tar \
  --batch_size 8 \
  --num_class 100 \
  --epochs 1 \
  --max-steps-per-epoch 5 \
  --tspipe-enable \
  --tspipe-config benchmarks/soft_target/tspipe.yaml
```

실행 후 확인할 파일:

```bash
cat results/<RUN_NOTE>/e2e_summary.log
ls results/<RUN_NOTE>/
find failover_logs -name experiment_summary.json | tail
```

## Sample Data 설명

저장소에는 실험 재현과 구조 확인을 위한 샘플/결과 데이터가 포함되어 있습니다.

- `dataset/imagenet100.txt`: ImageNet100 class 목록
- `benchmarks/soft_target/planner/profile/snet.csv`: student network layer-wise profile
- `benchmarks/soft_target/planner/profile/tnet.csv`: teacher network layer-wise profile
- `results/demo/`, `results/demo_batch64/`: demo 실행 결과와 profiling log
- `failover_logs/`, `failover_results*/`: failover 실험 로그, event jsonl, performance jsonl, 요약 json
- `exp0_final/`, `exp0b_results/`, `exp0b_results_v2/`: checkpoint/failure injection 실험 요약
- `analysis/`: 실험 결과를 바탕으로 생성된 그래프 이미지

전체 ImageNet 데이터셋은 저장소에 포함되어 있지 않으며, 실제 학습 실행 시 `--img_root`로 외부 ImageNet 경로를 지정해야 합니다.

## Database or Data Used

별도의 관계형 데이터베이스나 서버형 DB는 사용하지 않습니다. 실험 데이터와 결과는 파일 시스템에 저장됩니다.

사용되는 데이터/파일 형식은 다음과 같습니다.

- 이미지 데이터셋: ImageNet/ImageNet100 디렉터리 구조
- ImageNet100 class allow-list: `dataset/imagenet100.txt`
- planner profile: `benchmarks/soft_target/planner/profile/*.csv`
- 설정 파일: YAML (`tspipe.yaml`, `tspipe_restart_kminus1.yaml` 등)
- 실험 로그: `.log`, `.txt`
- 구조화 결과: `.json`, `.jsonl`
- 체크포인트: PyTorch `.pth`, `.pth.tar`, `.pt`
- 분석 결과: `.png`, `.svg`

실제 학습에 필요한 원본 ImageNet 데이터와 대형 teacher/student checkpoint는 저장소에 포함하고 있지 않습니다. 대신 README의 실행 예시처럼 `--img_root`, `--t_model`, `--s_init` 인자로 외부 경로를 지정해야 합니다. 

## Used Open Source

이 프로젝트는 다음 오픈소스 프로젝트와 라이브러리를 기반으로 합니다.

- TSPipe: 원본 pipeline parallel KD 프레임워크
- torchgpipe: TSPipe 계열 pipeline/microbatching 구현의 기반 아이디어
- Knowledge-Distillation-Zoo: soft target KD benchmark 구조 및 loss 구성
- PyTorch, torchvision, torchaudio
- timm
- transformers
- NumPy, pandas
- PyYAML
- tqdm
- TensorBoard, tensorboardX
- psutil
- GitPython
- nvidia-ml-py 또는 pynvml
- matplotlib, Pillow

원본 TSPipe의 `LICENSE`와 `CITATION.cff`는 저장소 루트에 유지되어 있습니다. TSPipe를 사용하거나 확장한 연구 결과를 공개할 경우 원 저작자와 라이선스를 함께 확인해야 합니다.

## 참고 문서

- `README_FAILOVER.md`: GPU failover 실험 시스템 설명
- `TSPipe_Failover_Implementation_Guide.md`: failover 구현 가이드
- `FAILOVER_ARCHITECTURE_ANALYSIS.md`: failover architecture 분석
- `FLOWCHART_VALIDATION_REPORT.md`: flowchart 검증 보고서
- `benchmarks/soft_target/planner/README_COMPLETE.md`: planner 관련 상세 설명
