# Import
import os
import sys
sys.path.append('/home/user/workdir/CMAQ_Emulator/main')

import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset

from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import r2_score

import geopandas as gpd
from shapely.geometry import Point
import matplotlib as mpl
import matplotlib.pyplot as plt

dataset_2013 = pd.read_csv('/home/user/workdir/CMAQ_Emulator/main/resources/ctrl/dataset_for_o3_interaction_v1.csv')

ctrl_data = pd.concat([dataset_2013], axis=0)
ctrl_data = ctrl_data.reset_index(drop=True).values
emis_data = ctrl_data[:, :17*9]
label_path_2013 = '/home/user/workdir/CMAQ_Emulator/main/datasets/concentration/2013'
label_path = []
for i in range(1, 120): 
    label_path.append(os.path.join(label_path_2013, '1.00', f'ACONC.{i}'))
label_data = []
for path in label_path:
    with nc.Dataset(path) as f:
        label_data.append(f.variables['O3'][:].data.squeeze())  # ncf 파일 내 목적변수를 지정 
label_data = np.array(label_data).reshape(len(label_data), 82, 67, 1)

from src.model.cmaqnet_unet import build_model

model_path = '/home/user/workdir/CMAQ_Emulator/main/src/model/o3_prediction/v6/final_model'  # 모델 저장 경로

epochs = 1000  # 훈련 반복 횟수
batch_size = 32  # 배치 크기
test_split = 0.4  # 테스트 데이터 비율 (50%)
random_seed = 42  # 랜덤 시드

X_emis_train, X_emis_test, y_train, y_test = train_test_split(emis_data, label_data, test_size=test_split, random_state=random_seed)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs")
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model(
        ctrl_dim=17*9,
        out_channel=1,
        hidden_size=[256, 128],
        hidden_depth=4,
        in_filters=20,
        kernel_size=3,
        activation='silu',
        dropout=0.0,
        use_abs=True
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.MeanAbsoluteError(),
    )
    
# define callback
def scheduler(epoch, lr):
    # if epoch <= 1000: return 2.5e-3
    if epoch <= 1000: return 1e-3
    else: return 1e-4
callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

callback_ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath='/home/user/workdir/CMAQ_Emulator/main/src/model/o3_prediction/v6/final_model-{epoch:02d}-{val_loss:.4f}',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    save_freq='epoch')

history = model.fit(
    x=[X_emis_train],   # 입력 데이터 (지역 별 배출량, 경계 조건 값)
    y=y_train,                                          # 입력 데이터의 Label 값 (netCDF에서 PM2.5 값)
    epochs=epochs,
    batch_size=batch_size,
    validation_data=[[X_emis_test], y_test], # test 데이터
    callbacks=[callback_lr]) #, callback_ckpt])
model.save(model_path)

def plot_loss(history):
    epochs = range(1, len(history.history['loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # 1. 전체 에포크에 대한 학습 및 검증 손실
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history.history['loss'], label='Training Loss', color='blue')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. 마지막 10 에포크의 손실 (세부 분석)
    plt.subplot(2, 2, 2)
    if len(epochs) >= 10:
        last_epochs = epochs[-10:]
        plt.plot(last_epochs, history.history['loss'][-10:], label='Training Loss', color='blue')
        plt.plot(last_epochs, history.history['val_loss'][-10:], label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss (Last 10 Epochs)')
        plt.legend()
        plt.grid(alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Not enough epochs for zoomed plot", ha='center')
    
    # 3. 학습 손실과 검증 손실의 차이
    plt.subplot(2, 2, 3)
    loss_diff = np.array(history.history['val_loss']) - np.array(history.history['loss'])
    plt.plot(epochs, loss_diff, label='Val Loss - Train Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Difference')
    plt.title('Difference between Validation and Training Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 4. 이동 평균 (window=10)을 통한 평활화된 손실 추세
    plt.subplot(2, 2, 4)
    window = 10
    if len(epochs) >= window:
        train_ma = [np.mean(history.history['loss'][max(0, i-window):i]) for i in range(1, len(history.history['loss'])+1)]
        val_ma = [np.mean(history.history['val_loss'][max(0, i-window):i]) for i in range(1, len(history.history['val_loss'])+1)]
        plt.plot(epochs, train_ma, label='Training Loss MA', color='blue', linestyle='--')
        plt.plot(epochs, val_ma, label='Validation Loss MA', color='orange', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (Moving Average)')
        plt.title('Moving Average of Loss (window = 10)')
        plt.legend()
        plt.grid(alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Not enough epochs for moving average plot", ha='center')
    
    plt.tight_layout()
    plt.show()
''' 16min '''

plot_loss(history)