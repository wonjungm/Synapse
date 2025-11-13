# Project Synapse (v1.0)

## 팀 (Team)
- **팀명 (Team Name):** Synapse (시냅스)
- **팀원 (Team Members):** 최지희, 문원정, 김현영
- **지도교수 (Advisor):** 심재형 교수님

## 프로젝트 개요 (Project Overview)
본 프로젝트는 대규모 언어 모델(LLM)의 지식 증류(Knowledge Distillation) 과정에서 발생하는 막대한 연산량과 GPU 자원 활용의 비효율 문제를 해결하는 것을 목표로 합니다. 이를 위해 교사–학생 모델 간 구조적 비대칭성과 성능이 서로 다른 이기종 GPU 환경을 모두 고려하여, 연산 부하를 균형 있게 분산하고 통신 오버헤드를 최소화하는 파이프라인 병렬화 전략을 개발합니다. 각 레이어의 연산 시간, 메모리 사용량, 통신량을 정밀하게 프로파일링하여 Cost Function을 수치화하고, 이를 기반으로 동적 계획법(DP)을 적용한 최적의 모델 파티셔닝과 스케줄링을 구현합니다. 

Teacher 모델은 ViT-Large, Student 모델은 ResNet-152로 설정하여 구조적 차이를 반영하고, DeepSpeed와 PyTorch 기반의 분산 학습 환경에서 다양한 실험을 수행하여 제안 기법의 효율성을 검증할 예정입니다. 본 연구는 GPU 자원의 활용도를 높이고 학습 속도를 개선하며, 실질적인 에너지 효율과 비용 절감 효과까지 달성하는 것을 궁극적인 목표로 합니다.


## 주요 키워드 (Keywords)
`LLM`, `Knowledge Distillation`, `Model Partitioning`, `Multi-GPU Training`, `Scheduling`, `Distributed Optimization`

## 저장소 구조 (Repository Structure)
- `README.md`: 프로젝트의 개요와 목표를 설명합니다.
- `IDEATION.md`: 프로젝트 주제 선정을 위한 아이디어 브레인스토밍 내용을 담습니다.
- `GROUNDRULES.md`: 원활한 협업을 위한 팀 내부 규칙을 정의합니다.

## 버전 정보 (Version)
- 본 파일은 프로젝트의 최종 계획을 담은 Version 1.0 문서입니다.
- This document is the Version 1.0 release, containing the finalized project plan.
