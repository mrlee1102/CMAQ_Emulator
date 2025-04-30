import os
import glob
import time
import numpy as np
import netCDF4 as nc
import tqdm

from multiprocessing import Pool, freeze_support, RLock, current_process
freeze_support()

pm_list = [
    'ASO4J','ASO4I','ANH4J','ANH4I','ANO3J','ANO3I','AALKJ',
    'AXYL1J','AXYL2J','AXYL3J','ATOL1J','ATOL2J','ATOL3J',
    'ABNZ1J','ABNZ2J','ABNZ3J','ATRP1J','ATRP2J','AISO1J',
    'AISO2J','AISO3J','ASQTJ','AORGCJ','AORGPAJ','AORGPAI',
    'AECJ','AECI','A25J','ANAJ','ACLJ','ACLI','AOLGAJ','AOLGBJ'
]
o3_list = [
    'O3'
]
month_idx = [
    ('2012357', '2013031'),
    ('2013081', '2013120'),
    ('2013172', '2013212'),
    ('2013264', '2013304')
]

rsm_path = glob.glob('/mnt/dsk0/bggo/CMAQ_dataset/nc/**')
rsm_path = [path for path in rsm_path if os.path.isdir(path)]
rsm_path = [path for path in rsm_path if 'RSM' in path]
rsm_path = sorted(rsm_path, key=lambda x: int(x.split('_')[-1]))

conc_path = {}
for path in rsm_path:
    month_conc = {}
    total_path = sorted(glob.glob(path + '/*.ncf'))
    # split by month
    for idx, (start, end) in enumerate(month_idx):
        month_conc[idx] = [
            path for path in total_path if start <= path.split('.')[-2] <= end
        ]
    conc_path[path.split('/')[-1]] = month_conc

keys = list(conc_path.keys())

prefix = conc_path[keys[0]][0][0].split('/')[-1]
prefix = prefix.split('.')[:-2]
prefix = '.'.join(prefix)

# for key in keys[86:]:
for key in keys:
    pm25_map, o3_map = [], []
    start_time = time.time()
    for season, path_list in conc_path[key].items():
        daily_pm25, daily_o3 = [], []
        pbar = tqdm.tqdm(total=len(path_list[:40]), desc=f'{key}: {month_idx[season][0]}/{month_idx[season][1]}')
        for path in path_list[:40]:
            pbar.set_description(f"{key}: {path.split('.')[-2]}/{month_idx[season][1]}")
            
            tmp_pm25, tmp_o3 = [], []
            dataset = nc.Dataset(path, 'r')
            for speices in pm_list:
                tmp_pm25.append(dataset[speices][:].squeeze())
                
            for speices in o3_list:
                tmp_o3.append(dataset[speices][:].squeeze())
                
            tmp_pm25 = np.transpose(tmp_pm25, (1, 0, 2, 3))
            tmp_o3 = np.transpose(tmp_o3, (1, 0, 2, 3))
            
            daily_pm25.append(tmp_pm25)
            daily_o3.append(tmp_o3)
            
            pbar.update(1)
        pbar.close()
        
        daily_pm25 = np.concatenate(daily_pm25, axis=0)
        daily_o3 = np.concatenate(daily_o3, axis=0)
        
        pm25_map.append(daily_pm25)
        o3_map.append(daily_o3)
    
    print(f'Processing {key}...', end=' ')
    pm25_map = np.array(pm25_map, dtype=np.float32)
    pm25_map = np.sum(pm25_map, axis=2)
    o3_map = np.array(o3_map, dtype=np.float32)
    
    np.savez_compressed(
        f'/home/user/workdir/CMAQ_Emulator/ncf_dataset/npy/{key}.npz',
        pm25_map=pm25_map,
        o3_map=o3_map
    )
    end_time = time.time()
    print(f'Done!, Elapsed time: {end_time - start_time:.2f}s\n')
