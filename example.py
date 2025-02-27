import ctypes
import numpy as np
import matplotlib.pyplot as plt
import tqdm

class CMAQNet(object):
    # 공유 라이브러리 로드 (C forward.c의 새 call() 함수 사용)
    model = ctypes.CDLL('/home/user/workdir/CMAQ_Emulator/build/libforward.so')
    # 새 함수 프로토타입: void call(float* c_ctrl_inputs, float* c_meteo_inputs, float* c_outputs, int batch_size)
    model.call.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # 컨트롤 입력 (flattened)
        ctypes.POINTER(ctypes.c_float),  # 메테오 입력 (flattened)
        ctypes.POINTER(ctypes.c_float),  # 출력 (flattened)
        ctypes.c_int                     # 배치 사이즈
    ]
    model.call.restype = None

    @staticmethod
    def predict(ctrl_x, meteo_x, batch_size: int = 1):
        num_samples = ctrl_x.shape[0]
        ctrl_x = ctrl_x.astype(np.float32)
        meteo_x = meteo_x.astype(np.float32)
        num_batches = num_samples // batch_size

        ctrl_batches = np.array_split(ctrl_x, num_batches)
        meteo_batches = np.array_split(meteo_x, num_batches)
        batched_outputs = np.zeros((num_batches, batch_size, 82, 67, 1), dtype=np.float32)
        pbar = tqdm.tqdm(total=num_batches, desc='Predicting')
        for i in range(num_batches):
            CMAQNet.model.call(
                ctrl_batches[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                meteo_batches[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batched_outputs[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size
            )
            pbar.update(1)
        pbar.close()
        return batched_outputs.reshape((num_samples, 82, 67, 1))

total_size = 16
batch_size = 4
# 컨트롤 입력: (total_size, 119)
ctrl_inputs = (np.random.rand(total_size, 119) + 0.5).astype(np.float32)
# 메테오 입력: (total_size, 82, 67, 7)
meteo_inputs = np.random.uniform(low=0.003, high=0.010, size=(total_size, 82, 67, 7)).astype(np.float32)

print("Control input shape:", ctrl_inputs.shape)
print("Meteo input shape:", meteo_inputs.shape)

outputs = CMAQNet.predict(ctrl_inputs, meteo_inputs, batch_size)
print("Output shape:", outputs.shape)


# 16개 샘플을 4x4 그리드로 시각화
plt.figure(figsize=(12, 12))
for i in range(total_size):
    plt.subplot(4, 4, i + 1)
    plt.imshow(outputs[i, :, :, 0][::-1], cmap='jet')
    plt.title(f'Sample {i+1}')
    plt.axis('off')
    plt.colorbar()
plt.tight_layout()
plt.savefig('result_1.png')