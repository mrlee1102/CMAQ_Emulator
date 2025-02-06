import os

import numpy as np
import pandas as pd
import netCDF4 as nc
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.model.cmaqnet_cond_unet import build_model

# define constants
'''
model_path:str = ...
epochs:int = ...
batch_size:int = ...
test_split:float = ...
random_seed:int = ...
'''
model_path = '/home/user/workdir/main/src/model/small_1/final_model'  # 모델 저장 경로
epochs = 100  # 훈련 반복 횟수
batch_size = 32  # 배치 크기
test_split = 0.2  # 테스트 데이터 비율 (20%)
random_seed = 42  # 랜덤 시드

# build model
model = build_model(
    ctrl_dim=17*5, # 17 regions * 5 precursor activities
    cond_dim=[1, 1], # timestep / boundary activity
    emb_dims=[128, 128],
    emb_type=['time', 'normal'],
    out_channel=1,
    hidden_size=[128, 96],
    hidden_depth=4,
    in_filters=20,
    kernel_size=3,
    activation='silu',
    dropout=0.0,
    use_abs=True)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.MeanSquaredError(),)
model.summary()

# define callback
def scheduler(epoch, lr):
    if epoch < 500: return 1e-3
    else: return 1e-4
callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

callback_ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath='/home/user/workdir/main/src/model/small_1/final_model-{epoch:02d}-{val_loss:.2f}',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    save_freq='epoch')

# load data
emis_ctrl_2019_05 = pd.read_csv(
    '/home/user/workdir/main/resources/ctrl/precursor_control_2019.csv', index_col=0)
emis_ctrl_2019_05['Timestep'] = 0.0
emis_ctrl_2019_05['Boundary'] = 0.5
# emis_ctrl_2019_02 = emis_ctrl_2019_05.copy()
# emis_ctrl_2019_02['Boundary'] = 1.0
emis_ctrl_2019_10 = emis_ctrl_2019_05.copy()
emis_ctrl_2019_10['Boundary'] = 1.0
ctrl_data = pd.concat([
    # emis_ctrl_2019_02,
    emis_ctrl_2019_05,
    emis_ctrl_2019_10
], axis=0)
ctrl_data = ctrl_data.reset_index(drop=True).values
emis_data, time_data, boundary_data = ctrl_data[:, :85], ctrl_data[:, 85], ctrl_data[:, 86]

base_path_2019 = '/home/user/workdir/main/datasets/concentration/2019'

conc_path = []
for i in range(1, 120): 
    conc_path.append(os.path.join(base_path_2019, '0.50', f'ACONC.{i}'))
for i in range(1, 120): 
    conc_path.append(os.path.join(base_path_2019, '1.00', f'ACONC.{i}'))
# -------------------------
conc_data = []
for path in conc_path:
    with nc.Dataset(path) as f:
        conc_data.append(f.variables['PM2_5'][:].data.squeeze())
conc_data = np.array(conc_data).reshape(len(conc_path), 82, 67, 1)

X_emis_train, X_emis_test, X_time_train, X_time_test, X_boundary_train, X_boundary_test, y_train, y_test = train_test_split(emis_data, time_data, boundary_data, conc_data, test_size=test_split, random_state=random_seed)

# train model
history = model.fit(
    x=[X_emis_train, X_time_train, X_boundary_train],
    y=y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=[[X_emis_test, X_time_test, X_boundary_test], y_test],
    callbacks=[callback_lr, callback_ckpt])
model.save(model_path)
