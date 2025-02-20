import ctypes
import numpy as np
import matplotlib.pyplot as plt
import tqdm

class CMAQNet(object):
    model = ctypes.CDLL('build/libforward.so')
    model.call.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input array
        ctypes.POINTER(ctypes.c_float),  # output array
        ctypes.c_int                     # batch size
    ]
    model.call.restype = None

    @staticmethod
    def predict(x, batch_size: int = 1):
        x_num = x.shape[0]
        x = x.astype(np.float32)  # ensure input is float32
        batched_x = np.array_split(x, x_num // batch_size)
        # 출력 배열: (배치 묶음 수, batch_size, 82, 67, 1)
        batched_outputs = np.zeros((len(batched_x), batch_size, 82, 67, 1), dtype=np.float32)
        pbar = tqdm.tqdm(total=len(batched_x), desc='Predicting')
        for i in range(len(batched_x)):
            CMAQNet.model.call(
                batched_x[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batched_outputs[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size)
            pbar.update(1)
        return batched_outputs.reshape((x_num, 82, 67, 1))

# total_size와 batch_size는 예시로 16개의 샘플, 배치 크기 1로 설정
total_size = 16
batch_size = 1

# C 코드가 기대하는 입력 shape: (batch_size, 17, 7)
# 여기서는 total_size == batch_size * num_batches 이므로 (16, 17, 7)
inputs = (np.random.rand(total_size, 17, 7) + 0.5).astype(np.float32)
print("Input shape:", inputs.shape)

outputs = CMAQNet.predict(inputs, batch_size)

# 출력 shape: (total_size, 82, 67, 1)
# 16개의 샘플을 4x4 그리드로 시각화합니다.
for i in range(total_size):
    plt.subplot(4, 4, i + 1)
    plt.imshow(outputs[i, :, :, 0][::-1], cmap='jet')
    plt.colorbar()
plt.tight_layout()
plt.show()
