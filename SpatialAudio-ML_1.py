import os
import librosa
import numpy as np
import json
import pyroomacoustics as pra

# 설정값
DATASET_PATH = "original_audio" # 원본 소리가 있는 곳
JSON_PATH = "data/spatial_data.json"
SAMPLE_RATE = 22050
NUM_SAMPLES_PER_COORD = 5000 # 생성할 가상 데이터 수

def save_spatial_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512):
    # 데이터를 담을 딕셔너리
    data = {
        "coords": [],
        "mfcc": []
    }

    # 1. 원본 소리 로드
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("에러: original_audio 폴더에 샘플 파일이 없습니다!")
        return
    # (일단 폴더의 첫 번째 파일을 사용한다고 가정)
    file_name = os.listdir(dataset_path)[0]
    file_path = os.path.join(dataset_path, file_name)
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # 2. 방의 가로, 세로, 높이 설정 (단위: 미터)
    # [가로(x), 세로(y), 높이(z)]
    room_dim = [10.0, 10.0, 10.0] # 10m x 10m x 3m 방
    mic_positions = np.array([
    # 마이크 위치 (x, y, z)

    #여기서 오류가 나는 이유는 마이크 3과 마이크 4의 z축 정보가 모두 1.0으로 동일하기 때문입니다. 
    #이렇게 되면 모델이 z축 방향의 위치를 구분할 수 없게 됩니다. 
    # 따라서 마이크 3과 마이크 4의 z축 정보를 다르게 설정하여 모델이 
    # z축 방향의 위치도 학습할 수 있도록 해야 합니다.
    #이상적인 마이크 배치는 정사면체 형태로, 각 마이크가 x, y, z축에서 고유한 위치를 가지도록 하는 것입니다.
    #구체적인 좌표는 방의 크기와 원하는 분포에 따라 다를 수 있지만, 예시로는 다음과 같이 설정할 수 있습니다:
    [1.0, 1.0, 1.0],  # 마이크 1 (기준점)
    [9.0, 9.0, 1.0],  # 마이크 2 (x축과 y축에서 멀리 떨어진 위치)
    [1.0, 9.0, 9.0],  # 마이크 3 (높이 8m로 설정하여 z축 정보도 포함)
    [9.0, 1.0, 9.0]  # 마이크 4 (높이 2m로 설정하여 z축 정보도 포함)
    ]).T # shape: (3, 3) -> (3, 3)
    # 3. 방 생성
    # absorption: 벽의 흡음률 (0에 가까울수록 반사가 심해 울리고, 1에 가까울수록 조용함)
    # fs: 샘플링 레이트 (우리의 오디오 설정과 맞춰야 함)
    room = pra.ShoeBox(room_dim, fs=22050, absorption=0.1, max_order=15)

    # 2. 루프를 돌며 가상 좌표 생성 및 시뮬레이션
    for i in range(NUM_SAMPLES_PER_COORD):
        # 무작위 소스 좌표 생성
        x, y, z = np.random.uniform(0.0, 9.8, 3)
        source_pos = [x, y, z] 

        # --- [시뮬레이션 핵심 파트] ---
        # 1) 방 생성 및 설정
        room = pra.ShoeBox(room_dim, fs=SAMPLE_RATE, absorption=0.2, max_order=15)
        room.add_source(source_pos, signal=signal)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs=SAMPLE_RATE))

        # 2) 시뮬레이션 실행 (Ray Tracing)
        room.simulate()

        # 3) 마이크에 수신된 신호 추출
        # room.mic_array.signals[0]는 시뮬레이션된 오디오 파형입니다.
        simulated_signal = room.mic_array.signals[0]
        # 4) MFCC 추출
        # signals[0]은 왼쪽 마이크, signals[1]은 오른쪽 마이크
        mfcc_1 = librosa.feature.mfcc(y=room.mic_array.signals[0], sr=SAMPLE_RATE, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_2 = librosa.feature.mfcc(y=room.mic_array.signals[1], sr=SAMPLE_RATE, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_3 = librosa.feature.mfcc(y=room.mic_array.signals[2], sr=SAMPLE_RATE, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_4 = librosa.feature.mfcc(y=room.mic_array.signals[3], sr=SAMPLE_RATE, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        # MFCC는 (특징 수, 시간 프레임 수) 형태로 나옵니다. 우리는 (시간 프레임 수, 특징 수, 채널 수) 형태로 모델에 넣어야 하므로 차원 재배치와 채널 결합이 필요합니다.
        mfcc_stereo = np.stack([mfcc_1.T, mfcc_2.T, mfcc_3.T, mfcc_4.T], axis=-1) # (1262, 13, 4) 형태로 저장
        # ----------------------------

        # 데이터 저장
        data["coords"].append([x, y, z]) 
        data["mfcc"].append(mfcc_stereo.tolist())

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{NUM_SAMPLES_PER_COORD} 완료...")

    # 3. JSON 저장
    # 폴더가 없으면 생성
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    
    print(f"모든 데이터가 {json_path}에 저장되었습니다!")

    # 실행
if __name__ == "__main__":
    save_spatial_mfcc(DATASET_PATH, JSON_PATH)