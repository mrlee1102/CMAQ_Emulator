import ctypes
import numpy as np
import matplotlib.pyplot as plt
import tqdm

class CMAQNet(object):
    model = ctypes.CDLL('/home/user/workdir/model_compile/build/libforward.so')
    model.call.argtypes = [
        ctypes.POINTER(ctypes.c_float), # input array
        ctypes.POINTER(ctypes.c_float), # output array
        ctypes.c_int # batch size
    ]
    model.call.restype = None

    @staticmethod
    def predict(x, batch_size:int=1):
        x_num = x.shape[0]
        x = x.astype(np.float32) # make sure the input is float32
        batched_x = np.array_split(x, x_num // batch_size)
        batched_outputs = np.zeros((len(batched_x), batch_size, 82, 67, 1), dtype=np.float32)
        pbar = tqdm.tqdm(total=len(batched_x), desc='Predicting')
        for i in range(len(batched_x)):
            print("before model.call")
            CMAQNet.model.call(
                batched_x[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batched_outputs[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size)
            pbar.update(1)
        return batched_outputs.reshape((x_num, 82, 67, 10))


total_size = 1
batch_size = 1

# control matrix, batch x region x sector
inputs = (np.random.rand(total_size, 119, 7, 17) + 0.5).astype(np.float32)
print(inputs.shape)
outputs = CMAQNet.predict(inputs, batch_size)

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(outputs[i, :, :, 0][::-1], cmap='jet')
    plt.colorbar()
plt.show()
