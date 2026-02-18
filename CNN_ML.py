import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

# 1. 데이터 로딩 (JSON -> Numpy)
JSON_PATH = "data/spatial_data.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

# 정확한 규격 설정
fixed_length = 1262
num_features = 13
num_channels = 2

mfcc_list = []
valid_indices = []

print("데이터 검사 중...")

for i, m in enumerate(data["mfcc"]):
    try:
        # 1. 일단 넘파이 배열로 변환 시도
        m_np = np.array(m)
        
        # 2. 차원이 3차원이 아니면 이 데이터는 문제가 있는 것이므로 버림
        if m_np.ndim != 3:
            continue
            
        # 3. (13, 1262, 2)인 경우 (1262, 13, 2)로 변경
        if m_np.shape[0] == num_features:
            m_np = m_np.transpose(1, 0, 2)
            
        # 4. 규격에 맞는 빈 그릇 생성
        temp = np.zeros((fixed_length, num_features, num_channels))
        
        # 5. 가능한 범위만큼만 복사 (자르거나 0으로 채우기)
        h = min(m_np.shape[0], fixed_length)
        w = min(m_np.shape[1], num_features)
        c = min(m_np.shape[2], num_channels)
        
        temp[:h, :w, :c] = m_np[:h, :w, :c]
        
        mfcc_list.append(temp)
        valid_indices.append(i)
        
    except:
        # 에러 나는 데이터는 그냥 무시하고 다음으로!
        continue

# 최종 결합
if len(mfcc_list) > 0:
    X = np.stack(mfcc_list, axis=0)
    X = np.squeeze(X) #불필요한 차원 제거
    y = np.array(data["coords"])[valid_indices, :2]
    print(f"✅ 성공: {len(X)}개의 데이터를 확보했습니다!")
    print(f"최종 shape: X={X.shape}, y={y.shape}")
else:
    print("❌ 실패: 모든 데이터의 규격이 맞지 않습니다. JSON 파일을 점검해야 합니다.")



# 학습 데이터와 테스트 데이터 분리 (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(f"데이터 준비 완료! 학습용: {len(X_train)}, 테스트용: {len(X_test)}")

# # 2. 커스텀 정확도 함수 정의 (예: 오차가 0.1m(10cm) 이내면 정답으로 간주)
# def spatial_accuracy(y_true, y_pred):
#     # 유클리드 거리 계산 (3차원 거리)
#     distance = K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))
#     # 거리가 0.1 미만인 것들의 평균 (즉, 비율)
#     return K.mean(K.cast(K.less(distance, 0.1), "float32"))


lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001
)
# 3. CNN 모델 설계 (Regressor)
def build_spatial_model(input_shape):
    model = models.Sequential([
        # 특징 추출 파트
        # Input shape에는 '샘플 수(None)'를 제외한 나머지 차원만 씁니다.
        layers.Input(shape=(1262, 13, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # 의사결정 파트 (Regression)
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), # 과적합 방지
        layers.Dense(2)      # 출력: x, y, z 좌표  
    ])
    
    # 회귀 문제이므로 Optimizer는 Adam, Loss는 MSE(평균제곱오차)를 씁니다.
    # compile 위에 추가
    model.compile(optimizer='adam', loss='mse', metrics=['mae'], run_eagerly=False)
    return model

# 3. 모델 생성 및 요약 출력
input_shape = (1262, 13, 1, 2) # (시간, 특징수, 채널)
model = build_spatial_model(input_shape)
model.summary()

# 4. 학습 시작
print("\n학습을 시작합니다...")

# y_train_scaled = y_train / 4.8
# y_test_scaled = y_test / 4.8

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), lr_schedule]
)
# 5. 학습 결과 시각화
# 손실 그래프
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('MAE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.tight_layout()
plt.show()

# 테스트 데이터 3개만 샘플링
test_samples = X_test[:3]
true_coords = y_test[:3]
pred_coords = model.predict(test_samples)

for i in range(3):
    print(f"[{i+1}번 샘플]")
    print(f"실제 위치: {true_coords[i]}")
    print(f"예측 위치: {pred_coords[i]}")
    print(f"오차 거리: {np.linalg.norm(true_coords[i] - pred_coords[i]):.2f}m")
    print("-" * 20)

# 6. 모델 저장
model.save("models/spatial_audio_model.h5")
print("\n✅ 모델 학습 완료 및 저장 성공! (spatial_audio_model.h5)")