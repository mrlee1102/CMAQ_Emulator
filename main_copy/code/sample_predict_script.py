"""README
Control matrix information (17*5 features):
    - Precursor (5 features):
        X_NOx_ALL, X_SOx_ALL, X_VOC_ALL, X_NH3_ALL, X_PM25_ALL
    - Region X (17 features):
        A: Seoul City,   B: Incheon City, C: Busan City,   D: Daegu City,
        E: Gwangju City, F: Gyeonggi-do,  G: Gangwon-do,   H: Chungbuk-do,
        I: Chungnam-do,  J: Gyeongbuk-do, K: Gyeongnam-do, L: Jeonbuk-do,
        M: Jeonnam-do,   N: Jeju-do,      O: Daejeon City, P: Ulsan City,
        Q: Sejong City

Concentration information:
    - PM2.5 (micrograms/m**3)

Map information:
    - Grid resolution: 9 km x 9 km
    - Grid size: 82 x 67
    - Total grid: 5,494
"""
import pandas as pd
import numpy as np
import tensorflow as tf

import os
import netCDF4 as nc

def get_korea_city_info():
    city_names = [
        'Seoul', 'Incheon', 'Busan', 'Daegu', 'Gwangju', 'Gyeonggi',
        'Gangwon', 'Chung-Buk', 'Chung-Nam', 'Gyeong-Buk', 'Gyeong-Nam',
        'Jeon-Buk', 'Jeon-Nam', 'Jeju', 'Daejeon', 'Ulsan', 'Sejong']
    # grid_info = pd.read_csv('./geoinfo/grid_allocation.csv')
    grid_info = pd.read_csv('/home/user/workdir/main/resources/geom/grid_allocation.csv')
    city_info_grid = {}
    for city in city_names:
        region =  grid_info[grid_info['Region_Name']==city]
        region_grid = region.loc[:, ['Row', 'Column']] - 1
        city_info_grid[city] = region_grid.values
    return city_info_grid

def get_annual_mean(preds, city_index):
    preds_arr = np.array(preds).reshape(-1, 82, 67)
    city_mean = {city: None for city in city_index}
    for city, grid_index in city_index.items():
        row, col = grid_index.T
        city_conc = preds_arr[:, row, col].mean(axis=1)
        city_mean[city] = city_conc
    return city_mean

# Test Data Load 
emis_ctrl_2019_05 = pd.read_csv(
    '/home/user/workdir/main/resources/ctrl/precursor_control_2013.csv', index_col=0)
emis_ctrl_2019_05['Timestep'] = 0.0
emis_ctrl_2019_05['Boundary'] = 0.5

emis_ctrl_2019_10 = emis_ctrl_2019_05.copy()
emis_ctrl_2019_10['Boundary'] = 1.0
ctrl_data = pd.concat([
    emis_ctrl_2019_05,
    emis_ctrl_2019_10
], axis=0)
ctrl_data = ctrl_data.reset_index(drop=True).values
emis_data, time_data, boundary_data = ctrl_data[:, :85], ctrl_data[:, 85], ctrl_data[:, 86]

base_path_2019 = '/home/user/workdir/main/datasets/concentration/2019'

conc_path = []
for i in range(1, 120): conc_path.append(os.path.join(base_path_2019, '0.50', f'ACONC.{i}'))
for i in range(1, 120): conc_path.append(os.path.join(base_path_2019, '1.00', f'ACONC.{i}'))

conc_data = []
for path in conc_path:
    with nc.Dataset(path) as f:
        conc_data.append(f.variables['PM2_5'][:].data.squeeze())
conc_data = np.array(conc_data).reshape(len(conc_path), 82, 67, 1)

if __name__ == '__main__':
    city_index = get_korea_city_info()
    
    model = tf.keras.models.load_model('/home/user/workdir/checkpoints/final_model')
    # working directory ==> /home/user/workdir/
    # ctrl = np.ones((10, 17*5), dtype=np.float32) # dummy control matrix
    # bnd = np.linspace(0.5, 1.0, 10, dtype=np.float32) # dummy boundary condition

    y_preds = model.predict([emis_data, time_data, boundary_data])
    y_preds_mean = get_annual_mean(y_preds, city_index)
    print(f"Columns in emis_data: {emis_data.shape}")
    print(f"Columns in time_data: {time_data.shape}")
    print(f"Columns in boundary_data: {boundary_data.shape}")
    print()
    y_preds_mean_df = pd.DataFrame(y_preds_mean).T
    print(f"Columns in y_preds_mean_df: {y_preds_mean_df.shape}")
    print(f"Columns in y_preds_mean_df[1]: {y_preds_mean_df.shape[1]}")
    print(f"Length of new column names: {len([f'bnd_{i:.2f}' for i in boundary_data])}")
    y_preds_mean_df.columns = [f'bnd_{i:.2f}' for i in boundary_data]
    y_preds_mean_df.to_csv('/home/user/workdir/main/result/pred_annual_mean.csv')
