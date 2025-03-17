#!/usr/bin/env python3
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.preprocessing import StandardScaler
import joblib

# ========= CMAQNet 모델 (공유 라이브러리) =========
class CMAQNet(object):
    # 공유 라이브러리 로드
    model = ctypes.CDLL('/home/user/workdir/CMAQ_Emulator/GreenEco/libo3_model.so')
    # 함수 프로토타입: void call(float *c_inputs, float *c_outputs, int batch_size)
    model.call.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # 컨트롤 입력 (flattened)
        ctypes.POINTER(ctypes.c_float),  # 출력 (flattened)
        ctypes.c_int                     # 배치 사이즈
    ]
    model.call.restype = None

    @staticmethod
    def predict(ctrl_x, batch_size: int = 1):
        num_samples = ctrl_x.shape[0]
        ctrl_x = ctrl_x.astype(np.float32)
        num_batches = num_samples // batch_size
        ctrl_batches = np.array_split(ctrl_x, num_batches)
        batched_outputs = np.zeros((num_batches, batch_size, 82, 67, 1), dtype=np.float32)
        pbar = tqdm.tqdm(total=num_batches, desc='Predicting')
        for i in range(num_batches):
            CMAQNet.model.call(
                ctrl_batches[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batched_outputs[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size
            )
            pbar.update(1)
        pbar.close()
        return batched_outputs.reshape((num_samples, 82, 67, 1))

# ========= 예측 및 시각화 =========
total_size = 16
batch_size = 4
# 랜덤 데이터 생성: control matrix, shape (total_size, 17*2)
inputs = (np.random.rand(total_size, 17*2) + 0.5).astype(np.float32)
outputs = CMAQNet.predict(inputs, batch_size)

# 스케일러 로드 후 inverse transform 적용
scaler = joblib.load('/home/user/workdir/CMAQ_Emulator/GreenEco/scale_parameter.pkl')
n_samples = outputs.shape[0]  # 테스트 샘플 수
prediction = scaler.inverse_transform(outputs.reshape(n_samples, -1))
outputs = prediction.reshape(outputs.shape)

# 16개 샘플을 4x4 그리드로 구성하여 시각화 후 파일로 저장
plt.figure(figsize=(12, 12))
for i in range(total_size):
    plt.subplot(4, 4, i + 1)
    plt.imshow(outputs[i, :, :, 0][::-1], cmap='jet')
    plt.title(f'Sample {i+1}')
    plt.axis('off')
    plt.colorbar()
plt.tight_layout()
# 결과 이미지를 출력
# plt.show()

# 결과 이미지를 파일로 저장
plt.savefig('/home/user/workdir/CMAQ_Emulator/GreenEco/result/result.jpg')

plt.close()
