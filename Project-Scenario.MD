| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | 다중 GPU에서 효율적인 지식 증류 파이프라인 수행을 위한 모델 파티셔닝 및 스케쥴링 기법 연구 |
| 프로젝트 키워드 | Knowledge Distillation, Model Partitioning, Scheduling, Multi-GPU Training, Distributed Optimization |
| 트랙 | 연구 |
| 프로젝트 멤버 | 김현영, 문원정, 최지희 |
| 팀지도 교수 | 심재형 |
| 무엇을 만들고자 하는가 | Teacher–Student 구조 기반의 지식 증류 과정을 다중 GPU 환경에서 수행할 수 있도록 모델을 파티셔닝하고, GPU 스케줄링을 최적화한 파이프라인 |
| 고객 | GPU 자원 최적화가 중요한 클라우드/AI 인프라 기업 |
| Pain Point | - **기업 AI 엔지니어**: 수십억 파라미터 규모의 LLM을 경량화해 서비스에 적용하려 하지만, 교사·학생 모델을 동시에 학습시키는 과정에서 GPU 메모리가 빠르게 소진된다. 이로 인해 학습 속도가 늦어지고, 모델 배포 일정이 지연된다.<br><br>- **클라우드 인프라 담당자**: 연구팀들이 대규모 증류 실험을 반복하면서 GPU 자원 사용이 폭증하지만, 분산 처리 효율이 떨어진다. 결국 운영 비용이 불필요하게 커지고, 클러스터 전체 자원 활용률이 낮아진다. |
| 사용할 소프트웨어 패키지의 명칭과 핵심기능/용도 | PyTorch: 기본 딥러닝 프레임워크. 모델 정의, 학습 루프 구현, GPU 연산 지원<br>DeepSpeed: 대규모 모델 학습 최적화. 파이프라인 병렬화, ZeRO 메모리 최적화, 분산 학습 지원 |
| 사용할 소프트웨어 패키지의 명칭과 URL | PyTorch: https://pytorch.org/<br>DeepSpeed: https://www.deepspeed.ai/ |
| 팀그라운드룰 | https://github.com/wonjungm/Synapse/blob/a5cbcfbee38afb6d1e9854926609495f76ede18b/GroundRule.md |
| 최종수정일 | 2025.09.14 |
