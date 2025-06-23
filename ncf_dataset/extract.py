import glob
import tqdm
import numpy as np
import netCDF4 as nc

pm25_sp = [
    "ASO4J", "ASO4I", "ANH4J", "ANH4I", "ANO3J",
    "ANO3I", "AALKJ", "AXYL1J", "AXYL2J", "AXYL3J",
    "ATOL1J", "ATOL2J", "ATOL3J", "ABNZ1J", "ABNZ2J",
    "ABNZ3J", "ATRP1J", "ATRP2J", "AISO1J", "AISO2J",
    "AISO3J", "ASQTJ", "AORGCJ", "AORGPAJ", "AORGPAI",
    "AECJ", "AECI", "A25J", "ANAJ", "ACLJ", "ACLI",
    "AOLGAJ", "AOLGBJ"
]

no2_sp = ["NO2"]
o3_sp = ["O3", "O3P"]
so2_sp = ["SO2"]
co_sp = ["CO"]

pass_date = ['2013031', '2013212', '2013304']
outpath = "/home/user/workdir/CMAQ_Emulator/ncf_dataset/rsm_total_pm25_conc_2sc.npy"
conc_data = []

print(f"Start extracting data for {outpath}")
for i in range(1, 2):
    rsm_path = sorted(glob.glob(f"/mnt/dsk1/yhlee/workdir/cmaq_dataset/nc/RSM_{i}/*.ncf"))
    rsm_path = list(filter(lambda x: not any([date in x for date in pass_date]), rsm_path))
    rsm_data = []
    for path in tqdm.tqdm(rsm_path, desc=f"RSM_{i}"):
        data = nc.Dataset(path)
        conc_map = np.zeros((24, 1, 82, 67))
        for key in pm25_sp:
            conc_map += data.variables[key][...]
        # no2_map = data.variables["NO2"][...]
        # o3_map = np.zeros((24, 1, 82, 67))
        # for key in o3_sp:
        #     o3_map += data.variables[key][...]
        # so2_map = data.variables["SO2"][...]
        # co_map = data.variables["CO"][...]
        # conc_map = np.concatenate((pm25_map, no2_map, o3_map, so2_map, co_map), axis=1)
        rsm_data.append(conc_map)
    rsm_data = np.array(rsm_data).reshape(4, 40, 24, 1, 82, 67)
    # np.save(f'/workdir/datasets/hourly_0/RSM_{i}.npy', rsm_data.astype(np.float32))
    conc_data.append(rsm_data)
conc_data = np.array(conc_data).astype(np.float32)
np.save(outpath, conc_data)
