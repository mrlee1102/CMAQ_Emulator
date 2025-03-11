import os
import sys
import glob

CMAQNET_DIR = '/mnt/dsk1/yhlee/workdir/cmaqnet/'
if CMAQNET_DIR not in sys.path:
    sys.path.append(CMAQNET_DIR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["SM_FRAMEWORK"] = "tf.keras"

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import netCDF4 as nc
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import Sequence
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from sklearn.model_selection import train_test_split
from cmaqnet.alloc import allocation, const_allocation
from cmaqnet.model import get_unet_model, load_model
SEASON = ['January', 'April', 'July', 'October']
CTRL_KEY_LIST = ['ALL_POW','ALL_IND','ALL_MOB','ALL_RES','NH3_AGR','ALL_SLV','ALL_OTH']
EMIS_KEY = ['SO2', 'NH3', 'VOCs', 'CO', 'PM2_5', 'NOx']
CONC_KEY = ['PM2.5_SO4', 'PM2.5_NH4', 'PM2.5_NO3', 'PM2.5_Total']#, 'O3']
REGION_CODE = {
    'A': 'Seoul City', 'B': 'Incheon City', 'C': 'Busan City', 'D': 'Daegu City',
    'E': 'Gwangju City', 'F': 'Gyeonggi-do', 'G': 'Gangwon-do', 'H': 'Chungbuk-do',
    'I': 'Chungnam-do', 'J': 'Gyeongbuk-do', 'K': 'Gyeongnam-do', 'L': 'Jeonbuk-do',
    'M': 'Jeonnam-do', 'N': 'Jeju-do', 'O': 'Daejeon City', 'P': 'Ulsan City', 'Q': 'Sejong City'
}

alloc_df = pd.DataFrame(allocation, columns=['Region_Name', 'Region_Code', 'Row', 'Column', 'Ratio'])
const_alloc_df = pd.DataFrame(const_allocation, columns=['Region_Name', 'Region_Code', 'Row', 'Column', 'Ratio'])

TARGET=os.environ['TARGET']
EPOCHS=int(os.environ['EPOCHS'])
BATCH_SIZE=108

ctrl = pd.read_csv('/mnt/dsk1/yhlee/workdir/cmaqnet/datasets/control_matrix.csv', index_col=0)
conc_path = [sorted(glob.glob(f'/dataset/npy/conc/hourly_new_split/RSM_{i}/*.npy')) for i in range(1, 120)]

train_time = [31*24, 37*24, 38*24, 38*24]
inter_time = [52*24, 54*24, 54*24, 61*24]
train_i_idx = [
    0,
    train_time[0] + inter_time[0],
    train_time[0] + inter_time[0] + train_time[1] + inter_time[1],
    train_time[0] + inter_time[0] + train_time[1] + inter_time[1] + train_time[2] + inter_time[2]
]
date = [
    np.array(list(range(train_i_idx[0], train_i_idx[0]+train_time[0])), dtype=np.float32).reshape(-1, 1),
    np.array(list(range(train_i_idx[1], train_i_idx[1]+train_time[1])), dtype=np.float32).reshape(-1, 1),
    np.array(list(range(train_i_idx[2], train_i_idx[2]+train_time[2])), dtype=np.float32).reshape(-1, 1),
    np.array(list(range(train_i_idx[3], train_i_idx[3]+train_time[3])), dtype=np.float32).reshape(-1, 1),
]
date = np.concatenate(date, axis=0).squeeze()

weights = {'pm25':1, 'o3':1000}

class DataGenerator(Sequence):
    def __init__(self, batch_size=32, shuffle=True):
        self.X_time = [
            np.array(list(range(train_i_idx[0], train_i_idx[0]+train_time[0])), dtype=np.float32).reshape(-1, 1),
            np.array(list(range(train_i_idx[1], train_i_idx[1]+train_time[1])), dtype=np.float32).reshape(-1, 1),
            np.array(list(range(train_i_idx[2], train_i_idx[2]+train_time[2])), dtype=np.float32).reshape(-1, 1),
            np.array(list(range(train_i_idx[3], train_i_idx[3]+train_time[3])), dtype=np.float32).reshape(-1, 1),
        ]
        self.X_time = np.concatenate(self.X_time, axis=0).squeeze().astype(np.float32)
        self.X_met = np.load('/mnt/dsk1/yhlee/workdir/cmaqnet/datasets/interpolated_meteo_v2.npy').astype(np.float32)
        self.y = np.load('/mnt/dsk1/yhlee/workdir/cmaqnet/datasets/2013_measured_masked_o3.npy').astype(np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X_time) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_ctrl = np.ones((self.batch_size, 119), dtype=np.float32)
        X_time = self.X_time[indices]
        X_met = self.X_met[X_time.astype(int)]
        y_conc = self.y[X_time.astype(int), :, :, :1] * weights[TARGET]
        y_conc = np.log(y_conc + 1)
        y_mask = self.y[X_time.astype(int), :, :, 1:]
        return [X_ctrl, X_met, y_conc, y_mask]
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.X_time))
        if self.shuffle:
            np.random.shuffle(self.indices)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    conc_yearly_unet_model = tf.keras.models.load_model(f'/mnt/dsk1/yhlee/workdir/cmaqnet/models/cond_unet_o3_all_attn')
    criterion = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    conc_yearly_unet_model.summary()
    def compute_loss(y_true, y_pred, model_losses):
        per_loss = criterion(y_true, y_pred)
        loss = tf.nn.compute_average_loss(per_loss, global_batch_size=BATCH_SIZE)
        if model_losses:
            loss += tf.nn.scale_regularization_loss(tf.add_n(conc_yearly_unet_model.losses))
        return loss

    def train_step(inputs):
        X_ctrl, X_met, y_conc, y_mask = inputs
        with tf.GradientTape() as tape:
            y_pred = conc_yearly_unet_model([X_ctrl, X_met], training=True)
            y_pred = y_pred * y_mask
            loss = compute_loss(y_conc, y_pred, conc_yearly_unet_model.losses)
        grads = tape.gradient(loss, conc_yearly_unet_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, conc_yearly_unet_model.trainable_variables))
        return loss

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    train_generator = DataGenerator(batch_size=BATCH_SIZE, shuffle=True)
    
    best_model = None
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        total_loss = 0
        num_steps = 0
        for step, datasets in enumerate(train_generator):
            loss = distributed_train_step(datasets)
            total_loss += loss
            num_steps += 1
            print(f'Epoch {epoch+1}/{EPOCHS} - Step {step+1}/{len(train_generator)} - Loss: {loss.numpy():.4f}', end='\r')
        total_loss /= num_steps
        if total_loss < best_loss:
            best_model = conc_yearly_unet_model
            best_loss = total_loss
        print(f'Epoch {epoch+1}/{EPOCHS} - Step {step+1}/{len(train_generator)} - Loss: {total_loss:.4f}')
    best_model.save(f'/home/user/workdir/CMAQ_Emulator/o3_model/cond_unet_{TARGET}_all')
