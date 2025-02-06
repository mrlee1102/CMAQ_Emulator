import os

import numpy as np
import pandas as pd
import netCDF4 as nc
from sklearn.model_selection import train_test_split

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.utils import Sequence
    
from src.model.cmaqnet_cond_unet import build_model

# define constants
model_path = '/home/user/workdir/main/src/model/large_1/final_model'  # 모델 저장 경로
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
    filepath='/home/user/workdir/main/src/model/large_1/final_model-{epoch:02d}-{val_loss:.2f}',
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
X_train, X_test, y_train_path, y_test_path = train_test_split(
    ctrl_data, conc_path, test_size=test_split, random_state=random_seed)

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, shuffle:bool=True, random_seed:int=42):
        self.shuffle = shuffle
        self.random_state = np.random.RandomState(random_seed)
        self.ctrl_data = x_set
        self.conc_path = y_set
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.ctrl_data)
    
    def __getitem__(self, index):
        i = self.indexes[index]
        emis_data, time_data, boundary_data = \
            self.ctrl_data[i, :85], self.ctrl_data[i, 85], self.ctrl_data[i, 86]
        path = self.conc_path[i]
        with nc.Dataset(path) as f:
            conc_data = f.variables['PM2_5'][:].data.squeeze().reshape(82, 67, 1)
        return [emis_data, time_data, boundary_data], conc_data
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ctrl_data))
        if self.shuffle:
            self.random_state.shuffle(self.indexes)

train_generator = DataGenerator(X_train, y_train_path)
test_generator = DataGenerator(X_test, y_test_path)

# train model
history = model.fit(
    train_generator,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=test_generator,
    callbacks=[callback_lr, callback_ckpt])
model.save(model_path)
