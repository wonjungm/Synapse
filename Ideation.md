# Project Synapse: Ideation (v1.0)

## 프로젝트 아이디어 1: 다중 GPU 환경에서 효율적인 지식 증류 파이프라인 수행을 위한 모델 파티셔닝 및 스케줄링 기법 연구

### 1. 핵심 키워드 (Keywords)
- Knowledge Distillation, Model Partitioning, Multi-GPU Scheduling, Distributed Training Efficiency, Pipeline Parallelism

### 2. 누구를 위해 (For Whom)
- 초거대 모델을 다루는 대학·산업 연구자
- 고성능 멀티 GPU 서버(A100, RTX 6000 Ada 등)를 운영하는 AI 스타트업/랩 연구팀
- 대형 Teacher → Student 구조에서 병목을 겪는 시스템/모델 엔지니어

### 3. 누구의 어떤 문제 해결 위해 (Problem to Solve)
- 초거대 모델 기반 지식 증류(teacher → student) 파이프라인은 막대한 계산량과 메모리 사용량 때문에 다중 GPU 환경에서도 비효율적일 수 있음. (단일 모델 학습 대비 1.5–2배 이상의 연산/메모리 비용이 발생)
- GPU 간 부하 불균형(load imbalance)과 통신 오버헤드가 발생해 학습 속도가 느려지고 리소스 활용 효율이 저하되는 문제가 있음.
- 파티셔닝이 비효율적일 경우 stage-level pipeline stall 증가하고 결과적으로 학습 속도 저하, GPU 활용률 감소, 자원 낭비 발생.
- 따라서, 다중 GPU 자원에서 지식 증류 과정을 최적화하여 모델 훈련 속도와 자원 활용도를 극대화하는 방법이 필요함.

### 4. 어떤 기술을 사용해서 (With What Technology)
- 모델 파티셔닝 기법: Teacher forward 전용 프로파일링, Student forward/backward 비용 분리 계산, activation/gradient 크기를 기반으로 통신 시간 추정
- 스케줄링 최적화: Teacher forward를 pipeline bubble에 삽입하는 비대칭 스케줄링(TS-Pipe 기반)을 통하 GPU 간 idle time 최소화.
- 분산 학습 프레임워크: DeepSpeed, Megatron-LM, PyTorch FSDP(Fully Sharded Data Parallel) 등 활용.
- 필요 시 mixed precision training(FP16, BF16) 및 offloading 기법을 결합하여 메모리 효율 극대화.

### 5. 무얼 만들려고 하는가 (What to Build)
- **결과물:**
  - Teacher–Student 구조에 특화된 자동화 모델 파티셔닝 알고리즘(Planner)
  - KD 파이프라인 지연(latency)·GPU 활용률·통신 오버헤드를 모두 고려한 StageTime cost model
  - 다양한 GPU 환경(예: 4×A100, 8×RTX 6000)에서의 벤치마크 및 재현 가능한 코드 베이스
- **목표:**
  - 기존 naive 분산 학습 대비 학습 시간 단축, GPU 활용률 증가, 통신 오버헤드 감소를 정량적으로 측정 및 발표
  - 지식 증류 과정을 효율화하여, 연구자들이 더 빠르고 경제적으로 student 모델을 개발할 수 있도록 지원
해 거대 LLM과 유사한 수준의 성능을 보이면서도, 모델 크기는 1/10 이하, 추론 비용은 수십 분의 1로 절감된 '가성비' 모델을 제작하고 그 효용성을 입증함.
