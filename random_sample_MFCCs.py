import json
import numpy as np
import matplotlib.pyplot as plt
import random

# 1. 데이터 로드
with open("data/spatial_data.json", "r") as f:
    data = json.load(f)

# 2. 첫 번째 샘플 꺼내기
random_idx = random.randint(0, len(data["coords"]) - 1)
coords = data["coords"][random_idx]
mfcc = np.array(data["mfcc"][random_idx])  # 리스트를 numpy 배열로 변환

# 3. 샘플 추출
coords = data["coords"][random_idx]
mfcc = np.array(data["mfcc"][random_idx])

print("-" * 30)
print(f"선택된 샘플 인덱스: {random_idx}")
print(f"해당 샘플의 좌표 (x, y): {coords}")
print(f"MFCC 데이터 크기: {mfcc.shape}") # (시간, 특징수)
print("-" * 30)

# 4. 시각화 (Transposed MFCC)
plt.figure(figsize=(10, 4))
# 학습 모델에 들어가는 형태에 맞춰 특징(y축)과 시간(x축)으로 시각화
plt.imshow(mfcc.T, aspect='auto', origin='lower', cmap='viridis')
plt.title(f"Random Sample MFCC - Position: {coords}")
plt.xlabel("Time Frames")
plt.ylabel("MFCC Coefficients")
plt.colorbar(label="Amplitude")
plt.tight_layout()

# 5. 결과 보기
plt.show()