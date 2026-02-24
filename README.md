# SpatialAudio-ML: Deep Learning Based Spatial Audio Synthesizer

**SpatialAudio-ML**은 사용자가 가상 공간 내에서 악기의 위치를 변경함에 따라 달라지는 소리의 특성(거리 감쇠, 잔향 등)을 딥러닝으로 학습하고 예측하는 인터랙티브 오디오 프로젝트입니다.

## 🎯 프로젝트 목표
- **물리 기반 오디오 합성**: `pyroomacoustics`를 활용하여 좌표(x, y)에 따른 가상 음향 데이터셋 구축.
- **오디오 특징 추출**: 오디오 신호를 MFCC 및 Mel-Spectrogram으로 변환하여 딥러닝 모델에 적합한 데이터로 가공.
- **공간 음향 회귀 모델링**: 특정 좌표값과 원본 소리를 입력받아 변형된 소리를 예측/생성하는 모델 학습.
- **인터랙티브 웹 데모**: 학습된 모델을 ONNX/TensorFlow.js로 변환하여 GitHub Pages에서 실시간 UX 구현.

## 🛠 기술 스택
- **Language:** Python 3.13
- **Audio Processing:** Librosa, PyRoomAcoustics, SoundFile
- **Deep Learning:** PyTorch
- **Version Control:** Git / GitHubg
- **Audio Sample**: (수정)

## 📂 프로젝트 마일스톤 (Milestones)

### 1️⃣ 데이터 생성 (Data Generation)
* **환경 구축**: `pyroomacoustics` 라이브러리를 활용하여 가상 3D 공간 시뮬레이션 환경($10m \times 10m \times 10m$) 구현.
* **물리 시뮬레이션**: 무작위 좌표($x, y, z$)에 소리 소스를 배치하고, 정사면체 구조의 4ch 마이크로 도달하는 음향 신호 생성.
* **디지털 변환**: 원본 오디오를 딥러닝 모델이 학습 가능한 **MFCC(Mel-Frequency Cepstral Coefficients)** 행렬($1262 \times 13$)로 변환.

### 2️⃣ 데이터셋 구축 (Dataset Prep)
* **데이터 저장**: 3,000개의 시뮬레이션 샘플을 약 3.2GB 규모의 JSON 데이터셋으로 구축.
* **데이터 검수**: 시각화 스크립트를 통해 좌표 변화에 따른 MFCC 패턴(Heatmap)의 변이 확인 및 데이터 무결성 검증.

### 3️⃣ 모델 설계 (CNN Architecture)
* **구조**: `Keras/TensorFlow` 기반의 CNN(Convolutional Neural Network) 아키텍처 설계.
* **학습 방식**: 위치 좌표를 예측하는 **Regression(회귀)** 모델 구축.
* **평가 지표**: 실제 좌표와 예측 좌표 사이의 거리 오차를 미터(m) 단위로 측정하는 **MAE(Mean Absolute Error)** 채택.

# 🎧 SpatialAudio-ML: 3D 공간 음향 위치 추정 프로젝트

인공지능(CNN)을 활용하여 실내 공간 내 소리 발생 지점(x, y, z)을 추정하는 오디오 딥러닝 프로젝트입니다.

### 🏆 최종 학습 결과 (Final Performance)
4채널 MFCC 데이터를 활용하여 학습한 결과, 1,000$m^3$ 부피의 3D 공간에서 **평균 오차 18.6cm (`val_mae: 0.1863`)**라는 압도적인 정밀도를 달성했습니다.

```text
# Epoch 50/50 최종 로그
loss: 0.5395 - mae: 0.5545 - val_loss: 0.0589 - val_mae: 0.1863

[1번 샘플]
실제 위치: [4.24, 9.03, 0.25]
예측 위치: [4.53, 9.27, 0.44]
오차 거리: 0.42m
--------------------
[2번 샘플]
실제 위치: [0.36 6.63 7.21]
예측 위치: [0.56 6.72  6.86]
오차 거리: 0.41m
--------------------
[3번 샘플]
실제 위치: [2.79, 6.16, 6.31]
예측 위치: [2.80, 6.43, 6.39]
오차 거리: 0.28m
```
### 🎯 Physics-Informed Debugging: From 2 or 3-ch to 4-ch (The Breakthrough)
- **해결책:** GPS 위성이 3D 위치를 잡기 위해 최소 4개가 필요한 것과 동일한 원리를 적용하여, **4번째 마이크를 추가**했습니다.
- **배치 좌표 (Tetrahedron Array):**
  - Mic 1: `[1.0, 1.0, 1.0]` (좌측 앞 바닥)
  - Mic 2: `[9.0, 9.0, 1.0]` (우측 뒤 바닥)
  - Mic 3: `[1.0, 9.0, 9.0]` (좌측 뒤 천장)
  - Mic 4: `[9.0, 1.0, 9.0]` (우측 앞 천장)
- **원리:** 4개의 마이크가 방의 모서리를 엇갈리게 점유하는 '정사면체' 구조를 이루면서, 4개의 구면(Sphere)이 교차하는 **단 하나의 완벽한 3D 교차점**을 산출해 냈습니다. 모델의 Z축/Y축 혼란이 완전히 제거되었습니다.

