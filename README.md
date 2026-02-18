# SpatialAudio-ML: Deep Learning Based Spatial Audio Synthesizer

**SpatialAudio-ML**은 사용자가 가상 공간 내에서 악기의 위치를 변경함에 따라 달라지는 소리의 특성(거리 감쇠, 잔향 등)을 딥러닝으로 학습하고 예측하는 인터랙티브 오디오 프로젝트입니다.

## 🎯 프로젝트 목표
- **물리 기반 오디오 합성**: `pyroomacoustics`를 활용하여 좌표(x, y)에 따른 가상 음향 데이터셋 구축.
- **오디오 특징 추출**: 오디오 신호를 MFCC 및 Mel-Spectrogram으로 변환하여 딥러닝 모델에 적합한 데이터로 가공.
- **공간 음향 회귀 모델링**: 특정 좌표값과 원본 소리를 입력받아 변형된 소리를 예측/생성하는 모델 학습.
- **인터랙티브 웹 데모**: 학습된 모델을 ONNX/TensorFlow.js로 변환하여 GitHub Pages에서 실시간 UX 구현.

## 🛠 기술 스택
- **Language:** Python 3.14+
- **Audio Processing:** Librosa, PyRoomAcoustics, SoundFile
- **Deep Learning:** PyTorch
- **Environment:** Anaconda / Jupyter Notebook
- **Version Control:** Git / GitHubg