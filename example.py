import ctypes
import numpy as np
import matplotlib.pyplot as plt
import tqdm

class CMAQNet(object):
    # ver1 모델용 공유 라이브러리 로드 (입력: (119,), 출력: (82, 67, 1))
    model = ctypes.CDLL('/home/user/workdir/CMAQ_Emulator/build/libforward.so')
    # 함수 프로토타입: void call(float* input, float* output, int batch_size)
    model.call.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # 입력 배열 (flatten된 데이터)
        ctypes.POINTER(ctypes.c_float),  # 출력 배열 (flatten된 데이터)
        ctypes.c_int                     # 배치 사이즈
    ]
    model.call.restype = None

    @staticmethod
    def predict(x, batch_size: int = 1):
        # x: control matrix, shape: (num_samples, 119)
        num_samples = x.shape[0]
        x = x.astype(np.float32)  # ensure float32
        # 배치 단위로 분할 (각 배치: (batch_size, 119))
        batched_x = np.array_split(x, num_samples // batch_size)
        # 모델 출력: shape: (batch_size, 82, 67, 1)
        batched_outputs = np.zeros((len(batched_x), batch_size, 82, 67, 1), dtype=np.float32)
        pbar = tqdm.tqdm(total=len(batched_x), desc='Predicting')
        for i in range(len(batched_x)):
            CMAQNet.model.call(
                batched_x[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batched_outputs[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size
            )
            pbar.update(1)
        pbar.close()
        return batched_outputs.reshape((num_samples, 82, 67, 1))

# 테스트: 총 16개 샘플, 한 번에 모두 예측
total_size = 16
batch_size = total_size

# 랜덤 데이터 생성: control matrix, shape (total_size, 119)
inputs = (np.random.rand(total_size, 119) + 0.5).astype(np.float32)
outputs = CMAQNet.predict(inputs, batch_size)

print("출력 shape:", outputs.shape)

# 16개 샘플을 4x4 그리드로 시각화
plt.figure(figsize=(12, 12))
for i in range(total_size):
    plt.subplot(4, 4, i + 1)
    plt.imshow(outputs[i, :, :, 0][::-1], cmap='jet')
    plt.title(f'Sample {i+1}')
    plt.axis('off')
    plt.colorbar()
plt.tight_layout()
plt.show()
