import os
import librosa
import numpy as np
import json
import pyroomacoustics as pra

# 설정값
DATASET_PATH = "original_audio" # 원본 소리가 있는 곳
JSON_PATH = "data/spatial_data.json"
SAMPLE_RATE = 22050
NUM_SAMPLES_PER_COORD = 1000 # 생성할 가상 데이터 수

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
    room_dim = [10.0, 10.0, 3.0] # 10m x 10m x 3m 방
    mic_positions = np.array([
    [1.9, 2.0, 1.5], # 왼쪽 귀
    [2.1, 2.0, 1.5]  # 오른쪽 귀
    ]).T # shape: (3, 2)
    # 3. 방 생성
    # absorption: 벽의 흡음률 (0에 가까울수록 반사가 심해 울리고, 1에 가까울수록 조용함)
    # fs: 샘플링 레이트 (우리의 오디오 설정과 맞춰야 함)
    room = pra.ShoeBox(room_dim, fs=22050, absorption=0.1, max_order=15)

    # 2. 루프를 돌며 가상 좌표 생성 및 시뮬레이션
    for i in range(NUM_SAMPLES_PER_COORD):
        # 무작위 소스 좌표 생성
        x, y = np.random.uniform(0.0, 9.8, 2)
        source_pos = [x, y, 1.5] # 높이는 마이크와 동일하게 1.5m로 고정

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
        mfcc_l = librosa.feature.mfcc(y=room.mic_array.signals[0], sr=SAMPLE_RATE, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_r = librosa.feature.mfcc(y=room.mic_array.signals[1], sr=SAMPLE_RATE, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # 두 채널을 겹쳐서 저장 (마치 컬러 이미지의 R, G 채널처럼)
        mfcc_stereo = np.stack([mfcc_l.T, mfcc_r.T], axis=-1) # (1262, 13, 2)
        # ----------------------------

        # 데이터 저장
        data["coords"].append([x, y, 1.5]) # z는 고정된 1.5m
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