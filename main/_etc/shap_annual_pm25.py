import os
import gc
import glob
os.chdir('/workdir')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import netCDF4 as nc
import numpy as np
import pandas as pd
import joblib

import shap

from scipy.stats import qmc
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from sklearn.model_selection import train_test_split

emis_ctrl_2013_10 = pd.read_csv('/workdir/datasets/emission_control_2013.csv', index_col=None)
emis_ctrl_2019_10 = pd.read_csv('/workdir/datasets/emission_control_2019.csv', index_col=0)
emis_ctrl_2019_10.columns = emis_ctrl_2013_10.columns
emis_ratio = pd.read_csv('/workdir/datasets/emission_ratio_2013_2019.csv', index_col=0)
emis_ratio = emis_ratio.reindex(columns=['NOx', 'SOx', 'VOC', 'NH3', 'PM2.5'])
emis_ratio = emis_ratio.values.flatten().reshape(1, -1)
emis_ctrl_2019_10 /= emis_ratio

emis_ctrl_2013_10['Boundary'] = 1.0
emis_ctrl_2019_10['Boundary'] = 1.0
emis_ctrl_2019_05 = emis_ctrl_2019_10.copy()
emis_ctrl_2019_05['Boundary'] = 0.5

ctrl_data = pd.concat([emis_ctrl_2019_05, emis_ctrl_2019_10], axis=0)
ctrl_data = ctrl_data.reset_index(drop=True)

emis_data, bnd_data = ctrl_data.iloc[:, :-1].values, ctrl_data.iloc[:, -1].values

base_path_2013 = '/workdir/datasets/concentration/2013'
base_path_2019 = '/workdir/datasets/concentration/2019'

conc_path = []
for i in range(1, 120):
    conc_path.append(os.path.join(base_path_2019, '0.50', f'ACONC.{i}'))
for i in range(1, 120):
    conc_path.append(os.path.join(base_path_2019, '1.00', f'ACONC.{i}'))

conc_data = []
for path in conc_path:
    with nc.Dataset(path) as f:
        conc_data.append(f.variables['PM2_5'][:].data.squeeze())
conc_data = np.array(conc_data).reshape(len(conc_path), 82, 67, 1)


X_train, _, bnd_train, _, _, _ = train_test_split(
    emis_data, bnd_data, conc_data, test_size=0.3, random_state=42)

shap_samples = qmc.LatinHypercube(d=86, seed=42).random(n=1000)
shap_samples = qmc.scale(
    shap_samples,
    [0.5]*85 + [0.5],
    [1.5]*85 + [1.0])

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.models.load_model(f'/workdir/models_new/cond_unet_pm25_yearly')
    def pred(inputs):
        emis_inputs, bnd_inputs = inputs[:, :-1], inputs[:, -1]
        pred = model.predict([emis_inputs, bnd_inputs], verbose=0)
        tf.keras.backend.clear_session()
        gc.collect()
        return pred.reshape(-1, 82*67)

    exp_input = np.concatenate([X_train, bnd_train.reshape(-1, 1)], axis=1)
    explainer = shap.PermutationExplainer(pred, exp_input)
    explanation = explainer(shap_samples)
    joblib.dump(explanation, f'/workdir/experiments/explanation_yearly_pm25_2019.shap',)