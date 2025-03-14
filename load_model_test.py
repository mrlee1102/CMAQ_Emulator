import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 모델 로드 (이미 저장된 모델 경로 사용)
# conc_yearly_unet_model = tf.keras.models.load_model('/mnt/dsk1/yhlee/workdir/cmaqnet/models/cond_unet_o3_all')
conc_yearly_unet_model = tf.keras.models.load_model('/home/user/workdir/CMAQ_Emulator/main/src/model/o3_prediction/final_model_main_v2')
# conc_yearly_unet_model = tf.keras.models.load_model('/mnt/dsk1/yhlee/workdir/cmaqnet/models/model_deploy/conc_pm25')
conc_yearly_unet_model.summary()
# 총 샘플 개수 및 배치 사이즈 설정
total_size = 16
batch_size = total_size  # 한 번에 모든 샘플 예측

# ctrl_input = np.random.random((total_size, 119)).astype(np.float32)
# met_input = np.random.random((total_size, 82, 67, 7)).astype(np.float32)
# print("Control input shape:", ctrl_input.shape)
# print("Meteorological input shape:", met_input.shape)
# pred = conc_yearly_unet_model.predict([ctrl_input, met_input])

# inputs = (np.random.rand(total_size, 119) + 0.5).astype(np.float32)
# outputs = conc_yearly_unet_model.predict(inputs, batch_size)

# 예측 결과 시각화: 출력 shape은 (total_size, 82, 67, 1)
# plt.figure(figsize=(12, 12))
# for i in range(total_size):
#     plt.subplot(4, 4, i + 1)
#     # 이미지 세로 방향을 뒤집어서 시각화
#     plt.imshow(outputs[i, :, :, 0][::-1], cmap='jet')
#     plt.title(f'Sample {i+1}')
#     plt.axis('off')
#     plt.colorbar()
# plt.tight_layout()
# plt.show()
