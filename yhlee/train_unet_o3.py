import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/mnt/dsk1/yhlee/workdir/cmaqnet'))))

from datetime import datetime, timedelta
import glob
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["SM_FRAMEWORK"] = "tf.keras"

warnings.filterwarnings('ignore', category=UserWarning)

import netCDF4 as nc
import tensorflow as tf
# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import tensorflow as tf
from sklearn.model_selection import train_test_split

def ioa_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)
def nmae_score(y_true, y_pred):
    return np.mean(np.abs(y_true + 1e-7 - y_pred) / np.abs(y_true + 1e-7))

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
REGION_GROUP = {
    'A': 'MR', 'B': 'MR', 'F': 'MR', # Metropolitan Region
    'H': 'CR', 'I': 'CR', 'O': 'CR', 'Q': 'CR', # Chungcheong Region
    'C': 'YR', 'D': 'YR', 'J': 'YR', 'K': 'YR', 'P': 'YR', # Yeongnam Region
    'E': 'JR', 'L': 'JR', 'M': 'JR', # Jeolla Region
    'G': 'GR', # Gangwon Region
    'N': 'JJR', # Jeju Region
}

FEATURE_LBL = []
for name in REGION_CODE.values():
    for key in CTRL_KEY_LIST:
        FEATURE_LBL.append(f"{name} {key}")

def read_cdf_map_data(
        data_path:str,
        target:str,
        prefix:str,
        keys:list[str]) -> np.ndarray:
    paths = [os.path.join(f'{data_path}/{target}/', f'{prefix}.{i+1}') for i in range(119)]
    datasets = [[nc.Dataset(path, 'r')[key][0, 0].data.tolist() for key in keys] for path in paths]
    return np.transpose(datasets, (0, 2, 3, 1))

def read_ctrl_matrix(data_path:str) -> np.ndarray:
    path = f'{data_path}/control_matrix.csv'
    return pd.read_csv(path, index_col=0).values

def get_ctprvn_map() -> gpd.GeoDataFrame:
    path = '/mnt/dsk1/yhlee/workdir/cmaqnet/datasets/geoinfo/ctp_rvn.shp'
    ctprvn = gpd.GeoDataFrame.from_file(path, encoding='cp949')
    ctprvn.crs = 'EPSG:5179'
    return ctprvn

def get_base_raster(ctprvn:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    proj = '+proj=lcc +lat_1=30 +lat_2=60 +lon_1=126 +lat_0=38 +lon_0=126 +ellps=GRS80 +units=m'
    points = [Point(i, j)
                for i in range(-180000, -180000 + 9000 * 67, 9000)
                for j in range(-585000, -585000 + 9000 * 82, 9000)]
    grid_data = gpd.GeoDataFrame(points, geometry='geometry', columns=['geometry'])
    grid_data.crs = ctprvn.to_crs(proj).crs
    grid_data.loc[:,'x_m'] = grid_data.geometry.x
    grid_data.loc[:,'y_m'] = grid_data.geometry.y
    grid_data.loc[:,'value'] = 0
    grid_data.loc[:,'index'] = grid_data.index
    return grid_data

def get_region_pixel_indices() -> list:
    ctprvn = get_ctprvn_map()
    grid_data = get_base_raster(ctprvn)

    cities = {
        0: '강원도', 1: '경기도', 2: '경상남도', 3: '경상북도',
        4: '광주광역시', 5: '대구광역시', 6: '대전광역시', 7: '부산광역시',
        8: '서울특별시', 9: '세종특별자치시', 10: '울산광역시', 11: '인천광역시',
        12: '전라남도', 13: '전라북도', 14: '제주특별자치도', 15: '충청남도',
        16: '충청북도'
    }

    gdf_joined_loc = ['CTPRVN_CD', 'CTP_ENG_NM', 'CTP_KOR_NM', 'index_right0']
    gdf_joined = gpd.sjoin(ctprvn, grid_data.to_crs(5179), predicate='contains')

    indices = gpd.GeoDataFrame(pd.merge(
        left=grid_data, right=gdf_joined.loc[:,gdf_joined_loc], 
        how='left', left_on='index', right_on='index_right0'
    ), geometry='geometry').dropna()
    pixel_indices = \
        [[(idx%82, idx//82) for idx in indices.loc[indices.CTP_KOR_NM==cities[region]].index.tolist()]
         for region, _ in cities.items()]
    return pixel_indices

atob = {
    0: 'G', 1: 'F', 2: 'K', 3: 'J', 4: 'E', 5: 'D',
    6: 'O', 7: 'C', 8: 'A', 9: 'Q', 10: 'P', 11: 'B',
    12: 'M', 13: 'L', 14: 'N', 15: 'I', 16: 'H'
}

ctprvn = get_ctprvn_map()
proj = '+proj=lcc +lat_1=30 +lat_2=60 +lon_1=126 +lat_0=38 +lon_0=126 +ellps=GRS80 +units=m'
ctprvn_proj = ctprvn.to_crs(proj)

grid_alloc = pd.read_csv('/mnt/dsk1/yhlee/workdir/cmaqnet/datasets/grid_allocation.csv')
grid_alloc = grid_alloc.sort_values(by=['Row', 'Column', 'Ratio'], ascending=[True, True, False])
grid_alloc = grid_alloc.drop_duplicates(subset=['Row', 'Column'], keep='first').reset_index(drop=True)

pixel_indices = get_region_pixel_indices()
total_index = []
for idx, grids in enumerate(pixel_indices):
    for grid in grids:
        total_index.append([
            grid[1], grid[0], 100.0, atob[idx], REGION_CODE[atob[idx]]
        ])
    # total_index += idx
# total_index = [list(index) for index in total_index]
total_index = pd.DataFrame(total_index, columns=grid_alloc.columns)
grid_alloc = pd.concat([
    grid_alloc.drop(columns=['Ratio', 'Region_Name']),
    total_index.drop(columns=['Ratio', 'Region_Name'])
]).sort_values(by=['Region_Code']).drop_duplicates().reset_index(drop=True)
grid_alloc[['Row', 'Column']] = grid_alloc[['Row', 'Column']] - 1
row_indices, col_indices = zip(*grid_alloc[['Row', 'Column']].values)

ctrl = pd.read_csv('/mnt/dsk1/yhlee/workdir/cmaqnet/datasets/control_matrix.csv', index_col=0)
conc_path = [sorted(glob.glob(f'/dataset/npy/conc/hourly_new_split/RSM_{i}/*.npy')) for i in range(1, 120)]

# conc_dataset = []
# for i, rsm_path in enumerate(conc_path):
#     conc_day = []
#     for t, path in enumerate(rsm_path):
#         print(f"Reading {i} | {path.split('/')[-1]}" + ' ' * 10, end='\r')
#         conc = np.load(path)[:, :18].sum(axis=1)[:, :, :, np.newaxis]
#         conc_day.append(conc)
#     conc_dataset.append(np.concatenate(conc_day, axis=0))
# conc_dataset = np.array(conc_dataset)

def get_ctrl_map(X):
    return X
    X = X.reshape(-1, 17, 7)
    ctrl_map = np.zeros((X.shape[0], 82, 67, 7))
    ctrl_map[:, :, :, 6] = 1.0
    for i, key in enumerate(REGION_CODE.keys()):
        index = grid_alloc.loc[grid_alloc.Region_Code==key, ['Row', 'Column']]
        index = index.drop_duplicates().values - 1
        row, col = zip(*index)
        ctrl_map[:, row, col, :] = X[:, i:i+1, :]
    return ctrl_map

# def get_time():
#     start_0 = 0
#     start_1 = (start_0+41*24)+(2013081-2013031)*24
#     start_2 = (start_1+40*24)+(2013172-2013120)*24
#     start_3 = (start_2+41*24)+(2013264-2013212)*24
#     a = list(range(
#         start_0,
#         start_0+41*24))
#     b = list(range(
#         start_1,
#         start_1+40*24))
#     c = list(range(
#         start_2,
#         start_2+41*24))
#     d = list(range(
#         start_3,
#         start_3+41*24))
#     return a+b+c+d

# ctrl_dataset = []
# time_dataset = []
# date = get_time()
# for i in range(119):
#     time_dataset.append(date.copy())
#     ctrl_map = get_ctrl_map(ctrl.values[i])
#     ctrl_dataset.append([ctrl_map for _ in range(len(date))])
# ctrl_dataset = np.array(ctrl_dataset).squeeze()
# time_dataset = np.array(time_dataset).astype(np.float32)

INDEX = int(os.environ['INDEX'])
slices = [slice(0, 41*24), slice(41*24, 81*24), slice(81*24, 122*24), slice(122*24, 163*24)]
timelength = [41*24, 40*24, 41*24, 41*24]
conc_dataset = np.load('/mnt/dsk1/yhlee/workdir/cmaqnet/datasets/conc_o3_total.npy')
conc_dataset = conc_dataset[:, slices[INDEX]]

ctrl_dataset = []
time_dataset = []
date = list(range(timelength[INDEX]))
for i in range(119):
    time_dataset.append(date.copy())
    ctrl_map = get_ctrl_map(ctrl.values[i])
    ctrl_dataset.append([ctrl_map for _ in range(len(date))])
ctrl_dataset = np.array(ctrl_dataset).squeeze()
time_dataset = np.array(time_dataset).astype(np.float32)

X_ctrl_train, X_ctrl_test, X_time_train, X_time_test, y_train, y_test = train_test_split(
    ctrl_dataset, time_dataset, conc_dataset, test_size=19, random_state=42, shuffle=True)

# X_ctrl_train_hour = X_ctrl_train.reshape(-1, 82, 67, 7)
# X_ctrl_test_hour = X_ctrl_test.reshape(-1, 82, 67, 7)
X_ctrl_train_hour = X_ctrl_train.reshape(-1, 119)
X_ctrl_test_hour = X_ctrl_test.reshape(-1, 119)
X_time_train_hour = X_time_train.reshape(-1)
X_time_test_hour = X_time_test.reshape(-1)
y_train_hour = y_train.reshape(-1, 82, 67, 1)
y_test_hour = y_test.reshape(-1, 82, 67, 1)



class Gridding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.coordinates  = [
            (grid_alloc
            .loc[grid_alloc.Region_Code==key, ['Row', 'Column']]
            .drop_duplicates()
            .values.tolist())
            for key in REGION_CODE.keys()
        ]
        
    def build(self, input_shape):
        pass

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Reshape inputs to (number of regions, 7)
        reshaped = tf.reshape(inputs, (batch_size, 17, 7))
        # Create a zero tensor of the target shape
        output_tensor = tf.zeros((batch_size, 82, 67, 7))
        
        # Assign the values to the corresponding coordinates
        for i, region_coords in enumerate(self.coordinates):
            for (x, y) in region_coords:
                indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], x), tf.fill([batch_size], y)], axis=1)
                updates = reshaped[:, i, :]
                output_tensor = tf.tensor_scatter_nd_update(output_tensor, indices, updates)

        return output_tensor

    def get_config(self):
        config = super().get_config()
        config.update({
            "coordinates": self.coordinates
        })
        return config


def get_unet_model(
    hidden_size:tuple=(128, 96),
    in_filters:int=20,
    kernel:int=12,
    activation:str='silu',
    dropout:float=0.0) -> tf.keras.Model:
    def time_embedding(t, dim:int=128):
        half_dim = dim // 2
        emb = tf.math.log(10000.) / (half_dim - 1)
        emb = tf.math.exp(-tf.range(half_dim, delta=emb))
        emb = t * emb[None, :]
        emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1)
        emb = tf.keras.layers.Dense(dim)(emb)
        emb = tf.keras.layers.Activation(activation)(emb)
        return emb

    def encoder_block(x_map, t_emb=None, filters:int=1, kernel:int=3, dropout:float=0.0):
        x_map = tf.keras.layers.Conv2D(filters, kernel, padding='same')(x_map)
        x_map = tf.keras.layers.BatchNormalization()(x_map)
        x_map = tf.keras.layers.Activation(activation)(x_map)
        x_param = None
        if t_emb is not None:
            t_emb = tf.keras.layers.Dense(filters)(t_emb)
            t_emb = tf.keras.layers.Activation(activation)(t_emb)
            t_emb = tf.keras.layers.Reshape((1, 1, filters))(t_emb)
            x_param = tf.keras.layers.Multiply()([x_map, t_emb])
        x_out = tf.keras.layers.Conv2D(filters, kernel, padding='same')(x_map)
        if x_param is not None:
            x_out = tf.keras.layers.Add()([x_out, x_param])
        x_out = tf.keras.layers.BatchNormalization()(x_out)
        x_out = tf.keras.layers.Activation(activation)(x_out)
        x_out = tf.keras.layers.Dropout(dropout)(x_out)
        return x_out
    
    def decoder_block(x0, x1, filters:int=1, kernel:int=3, dropout:float=0.0, t_emb=None):
        x0 = tf.keras.layers.Conv2DTranspose(
            filters, kernel, strides=2, padding='same')(x0)
        x0 = tf.keras.layers.BatchNormalization()(x0)
        x0 = tf.keras.layers.Activation(activation)(x0)
        x_param = None
        if t_emb is not None:
            t_emb = tf.keras.layers.Dense(filters)(t_emb)
            t_emb = tf.keras.layers.Activation(activation)(t_emb)
            t_emb = tf.keras.layers.Reshape((1, 1, filters))(t_emb)
            x_param = tf.keras.layers.Multiply()([x0, t_emb])
        x0 = tf.keras.layers.Concatenate()([x0, x1])
        x0 = tf.keras.layers.Conv2D(filters, kernel, padding='same')(x0)
        if x_param is not None:
            x0 = tf.keras.layers.Add()([x0, x_param])
        x0 = tf.keras.layers.BatchNormalization()(x0)
        x0 = tf.keras.layers.Activation(activation)(x0)
        x0 = tf.keras.layers.Dropout(dropout)(x0)
        return x0
    
    ctrl_inputs = tf.keras.Input(shape=(119,))
    ctrl_maps = Gridding()(ctrl_inputs)
    time_inputs = tf.keras.Input(shape=(1,))
    init_map = tf.keras.layers.Resizing(*hidden_size)(ctrl_maps)

    t_emb = time_embedding(time_inputs, 128)
    
    x = x0 = encoder_block(init_map, t_emb, in_filters, kernel, dropout)
    x = tf.keras.layers.MaxPool2D()(x)
    x = x1 = encoder_block(x, t_emb, in_filters*2, kernel, dropout)
    x = tf.keras.layers.MaxPool2D()(x)
    x = x2 = encoder_block(x, t_emb, in_filters*4, kernel, dropout)
    x = tf.keras.layers.MaxPool2D()(x)
    x = x3 = encoder_block(x, t_emb, in_filters*8, kernel, dropout)
    x = tf.keras.layers.MaxPool2D()(x)
    
    x = encoder_block(x, t_emb, in_filters*16, kernel, dropout)
    x = decoder_block(x, x3, in_filters*8, kernel, dropout)
    x = decoder_block(x, x2, in_filters*4, kernel, dropout)
    x = decoder_block(x, x1, in_filters*2, kernel, dropout)
    x = decoder_block(x, x0, in_filters, kernel, dropout)
    
    x = tf.keras.layers.Resizing(82, 67)(x)
    x = tf.keras.layers.Conv2D(1, 1)(x)

    model = tf.keras.Model(inputs=[ctrl_inputs, time_inputs], outputs=x)
    return model

EPOCHS=int(os.environ['EPOCHS'])
conc_yearly_unet_model = get_unet_model(kernel=3)
conc_yearly_unet_model.compile(
    optimizer=tf.keras.optimizers.AdamW(1e-3),
    loss=tf.keras.losses.MeanSquaredError(),)
conc_yearly_unet_model.summary()
history = conc_yearly_unet_model.fit(
    [X_ctrl_train_hour, X_time_train_hour], y_train_hour,
    validation_data=[
        [X_ctrl_test_hour, X_time_test_hour], y_test_hour
    ], batch_size=512, epochs=EPOCHS, verbose=1)

conc_yearly_unet_model.save(f'/home/user/workdir/yhlee/models/cond_unet_o3_{INDEX}')
