import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/mnt/dsk1/yhlee/workdir'))))

import glob
import warnings
# os.chdir('/workdir')
warnings.filterwarnings('ignore', category=UserWarning)

import torch, torchvision
from torch.distributed import init_process_group, destroy_process_group
import bitsandbytes as bnb

import tqdm
import netCDF4 as nc
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split

class Embedding(torch.nn.Module):
    def __init__(self, dim:int=128) -> None:
        super().__init__()
        self.dim = dim//2
        self.register_buffer('param', torch.tensor(10000))

    def forward(self, time:torch.Tensor, train:bool=True) -> torch.Tensor:
        device = time.device
        t_emb = torch.log(self.param.to(device)) / (self.dim - 1) 
        t_emb = torch.exp(torch.arange(self.dim, device=device) * -t_emb)
        t_emb = time[:, None] * t_emb[None, :]
        t_emb = torch.cat((t_emb.sin(), t_emb.cos()), dim=-1)
        return t_emb

class Encoder(torch.nn.Module):
    def __init__(
        self,
        filters:int=1,
        kernel_size:int=3,
        dropout:float=0.0) -> None:
        super().__init__()
        self.layer_act = torch.nn.SiLU()
        self.layer_conv_1 = torch.nn.LazyConv2d(
            out_channels=filters,
            kernel_size=kernel_size,
            padding='same')
        self.layer_norm_1 = torch.nn.LazyBatchNorm2d()
        self.layer_fc_1 = torch.nn.LazyLinear(filters)
        self.layer_conv_2 = torch.nn.LazyConv2d(
            out_channels=filters,
            kernel_size=kernel_size,
            padding='same')
        self.layer_dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        X_ctrl:torch.Tensor,
        X_time:torch.Tensor,
        train:bool=True) -> torch.Tensor:
        batch_size = X_ctrl.shape[0]
        X_map = self.layer_conv_1(X_ctrl)
        X_map = self.layer_norm_1(X_map)
        X_map = self.layer_act(X_map)
        X_time = self.layer_fc_1(X_time)
        X_time = self.layer_act(X_time)
        X_time = X_time.reshape(batch_size, -1, 1, 1)
        X_param = X_map * X_time
        X_out = self.layer_conv_2(X_ctrl)
        X_out = X_out + X_param
        X_out = self.layer_norm_1(X_out)
        X_out = self.layer_act(X_out)
        if train:
            X_out = self.layer_dropout(X_out)
        return X_out

class Decoder(torch.nn.Module):
    def __init__(
        self,
        filters:int=1,
        kernel_size:int=3,
        dropout:float=0.0) -> None:
        super().__init__()
        self.layer_act = torch.nn.SiLU()
        self.layer_conv_1 = torch.nn.LazyConvTranspose2d(
            out_channels=filters,
            kernel_size=kernel_size,
            stride=2,
            output_padding=1,
            padding=1)
        self.layer_norm_1 = torch.nn.LazyBatchNorm2d()
        self.layer_fc_1 = torch.nn.LazyLinear(filters)
        self.layer_conv_2 = torch.nn.LazyConv2d(
            out_channels=filters,
            kernel_size=kernel_size,
            padding='same')
        self.layer_dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        X_ctrl_0:torch.Tensor,
        X_ctrl_1:torch.Tensor,
        X_time:torch.Tensor,
        train:bool=True) -> torch.Tensor:
        batch_size = X_ctrl_0.shape[0]
        X_ctrl_0 = self.layer_conv_1(X_ctrl_0)
        X_ctrl_0 = self.layer_norm_1(X_ctrl_0)
        X_ctrl_0 = self.layer_act(X_ctrl_0)
        X_time = self.layer_fc_1(X_time)
        X_time = self.layer_act(X_time)
        X_time = X_time.reshape(batch_size, -1, 1, 1)
        X_param = X_ctrl_0 * X_time
        X_out = torch.concat((X_ctrl_0, X_ctrl_1), dim=1)
        X_out = self.layer_conv_2(X_out)
        X_out = X_out + X_param
        X_out = self.layer_norm_1(X_out)
        X_out = self.layer_act(X_out)
        if train:
            X_out = self.layer_dropout(X_out)
        return X_out

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
    path = '/workdir/datasets/geoinfo/ctp_rvn.shp'
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

    gdf_joined_loc = ['CTPRVN_CD', 'CTP_ENG_NM', 'CTP_KOR_NM', 'index_right']
    gdf_joined = gpd.sjoin(ctprvn, grid_data.to_crs(5179), predicate='contains')

    indices = gpd.GeoDataFrame(pd.merge(
        left=grid_data, right=gdf_joined.loc[:,gdf_joined_loc], 
        how='left', left_on='index', right_on='index_right'
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

grid_alloc = pd.read_csv('/workdir/datasets/grid_allocation.csv')
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

INDEX = int(os.environ['INDEX'])
slices = [slice(0, 41*24), slice(41*24, 81*24), slice(81*24, 122*24), slice(122*24, 163*24)]
timelength = [41*24, 40*24, 41*24, 41*24]

ctrl = pd.read_csv('/workdir/datasets/control_matrix.csv', index_col=0)
# conc_path = [sorted(glob.glob(f'/dataset/npy/conc/hourly_new_split/RSM_{i}/*.npy'))[slices[INDEX]] for i in range(1, 120)]
# conc_dataset = []
# for i, rsm_path in enumerate(conc_path):
#     conc_day = []
#     for t, path in enumerate(rsm_path):
#         print(f"Reading {i} | {path.split('/')[-1]}" + ' ' * 10, end='\r')
#         conc = np.load(path)[:, :18].sum(axis=1)[:, :, :, np.newaxis]
#         conc_day.append(conc)
#     conc_dataset.append(np.concatenate(conc_day, axis=0))
# conc_dataset = np.array(conc_dataset)
conc_dataset = np.load('/workdir/datasets/conc_o3_total.npy')
conc_dataset = conc_dataset[:, slices[INDEX]]

def get_ctrl_map(X):
    X = X.reshape(-1, 17, 7)
    ctrl_map = np.zeros((X.shape[0], 82, 67, 7))
    ctrl_map[:, :, :, 6] = 0.0
    for i, key in enumerate(REGION_CODE.keys()):
        index = grid_alloc.loc[grid_alloc.Region_Code==key, ['Row', 'Column']]
        index = index.drop_duplicates().values - 1
        row, col = zip(*index)
        ctrl_map[:, row, col, :] = X[:, i:i+1, :]
    return ctrl_map

def get_time():
    start_0 = 0
    start_1 = (start_0+41*24)+(2013081-2013031)*24
    start_2 = (start_1+40*24)+(2013172-2013120)*24
    start_3 = (start_2+41*24)+(2013264-2013212)*24
    a = list(range(
        start_0,
        start_0+41*24))
    b = list(range(
        start_1,
        start_1+40*24))
    c = list(range(
        start_2,
        start_2+41*24))
    d = list(range(
        start_3,
        start_3+41*24))
    return a+b+c+d

ctrl_dataset = []
time_dataset = []
# date = list(range(timelength[INDEX]))
date = get_time()
for i in range(119):
    time_dataset.append(date.copy())
    ctrl_dataset.append([ctrl.values[i] for _ in range(len(date))])
ctrl_dataset = np.array(ctrl_dataset).squeeze()
time_dataset = np.array(time_dataset).astype(np.float32)

X_ctrl_train, X_ctrl_test, X_time_train, X_time_test, y_train, y_test = train_test_split(
    ctrl_dataset, time_dataset, conc_dataset, train_size=80, random_state=42, shuffle=True)

X_ctrl_train_hour = X_ctrl_train.reshape(-1, 119)
X_time_train_hour = X_time_train.reshape(-1)
y_train_hour = y_train.reshape(-1, 1, 82, 67)

X_ctrl_test_hour = X_ctrl_test.reshape(-1, 119)
X_time_test_hour = X_time_test.reshape(-1)
y_test_hour = y_test.reshape(-1, 1, 82, 67)

# X_ctrl_train, X_ctrl_val, X_time_train, X_time_val, y_train, y_val = train_test_split(
#     X_ctrl_train_hour, X_time_train_hour, y_train_hour,
#     test_size=0.1, random_state=42, shuffle=True
# )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

indicies = [
    (grid_alloc
     .loc[grid_alloc.Region_Code==key, ['Row', 'Column']]
     .drop_duplicates()
     .values.tolist())
    for key in REGION_CODE.keys()
]
class CMAQNet(torch.nn.Module):
    def __init__(
        self,
        in_filters:int=20,
        kernel_size:int=3,
        dropout:float=0.0):
        super().__init__()
        for i, index in enumerate(indicies):
            self.register_buffer(
                f'region_{i}', torch.tensor(index, dtype=torch.int64))
        self.layer_resize_1 = torchvision.transforms.Resize((128, 96))
        self.layer_resize_2 = torchvision.transforms.Resize((82, 67))
        self.block_embedding = Embedding(128)
        self.layer_conv_out = torch.nn.LazyConv2d(
            out_channels=1, kernel_size=1, padding='same')
        self.block_encoder_1 = Encoder(in_filters, kernel_size, dropout)
        self.block_encoder_2 = Encoder(in_filters*2, kernel_size, dropout)
        self.block_encoder_3 = Encoder(in_filters*4, kernel_size, dropout)
        self.block_encoder_4 = Encoder(in_filters*8, kernel_size, dropout)
        self.block_encoder_5 = Encoder(in_filters*16, kernel_size, dropout)
        self.block_decoder_4 = Decoder(in_filters*8, kernel_size, dropout)
        self.block_decoder_3 = Decoder(in_filters*4, kernel_size, dropout)
        self.block_decoder_2 = Decoder(in_filters*2, kernel_size, dropout)
        self.block_decoder_1 = Decoder(in_filters, kernel_size, dropout)

    def const_ctrl_map(self, ctrl_mat:torch.Tensor)->torch.Tensor:
        batch_size = ctrl_mat.shape[0]
        ctrl_mat = ctrl_mat.reshape(-1, 17, 7).transpose(2, 1)
        ctrl_map = torch.zeros((batch_size, 7, 82, 67), device=ctrl_mat.device)
        for i in range(17):
            row, col = zip(*getattr(self, f'region_{i}'))
            ctrl_map[:, :, row, col] = ctrl_mat[:, :, i:i+1]
        return ctrl_map

    def forward(
        self,
        X_ctrl:torch.Tensor,
        X_time:torch.Tensor,
        train:bool=True) -> torch.Tensor:
        X_ctrl = self.const_ctrl_map(X_ctrl)
        X_ctrl = self.layer_resize_1(X_ctrl)
        X_time = self.block_embedding(X_time, train)
        X_ctrl = X_ctrl_0 = self.block_encoder_1(X_ctrl, X_time, train)
        X_ctrl = torch.nn.functional.max_pool2d(X_ctrl, 2)
        X_ctrl = X_ctrl_1 = self.block_encoder_2(X_ctrl, X_time, train)
        X_ctrl = torch.nn.functional.max_pool2d(X_ctrl, 2)
        X_ctrl = X_ctrl_2 = self.block_encoder_3(X_ctrl, X_time, train)
        X_ctrl = torch.nn.functional.max_pool2d(X_ctrl, 2)
        X_ctrl = X_ctrl_3 = self.block_encoder_4(X_ctrl, X_time, train)
        X_ctrl = torch.nn.functional.max_pool2d(X_ctrl, 2)
        X_ctrl = self.block_encoder_5(X_ctrl, X_time, train)
        X_ctrl = self.block_decoder_4(X_ctrl, X_ctrl_3, X_time, train)
        X_ctrl = self.block_decoder_3(X_ctrl, X_ctrl_2, X_time, train)
        X_ctrl = self.block_decoder_2(X_ctrl, X_ctrl_1, X_time, train)
        X_ctrl = self.block_decoder_1(X_ctrl, X_ctrl_0, X_time, train)
        X_out = self.layer_conv_out(X_ctrl)
        X_out = self.layer_resize_2(X_out)
        return X_out

model = CMAQNet()
x = torch.rand(1, 17*7).float()
t = torch.rand(1, 1).float()
with torch.no_grad():
    y = model(x, t) # Init LazyModule
# model = torch.nn.parallel.DataParallel(model)
model.cuda()

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, X_ctrl, X_time, y):
        self.X_ctrl = X_ctrl
        self.X_time = X_time
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_ctrl = torch.tensor(self.X_ctrl[idx], dtype=torch.float32)
        X_time = torch.tensor(self.X_time[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X_ctrl, X_time, y

train_dataset = TrainDataset(
    X_ctrl_train_hour, X_time_train_hour, y_train_hour)
val_dataset = TrainDataset(
    X_ctrl_test_hour, X_time_test_hour, y_test_hour)

# train_sampler = torch.utils.data.distributed.DistributedSampler(
#     dataset=train_dataset, shuffle=True)
# val_sampler = torch.utils.data.distributed.DistributedSampler(
#     dataset=train_dataset, shuffle=False)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=16,)
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=4096,
    shuffle=False,
    num_workers=16,)

criterion = torch.nn.MSELoss()
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3)
# scaler = torch.cuda.amp.GradScaler()

best_loss = float('inf')
best_model = None

EPOCHS=int(os.environ['EPOCHS'])
print(f'Train loader size: {len(train_loader)}')
print(f'Validation laoder size: {len(val_loader)}')
for epoch in range(EPOCHS):
    train_loss = []
    val_loss = []
    pbar = tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{EPOCHS}')
    model.train()
    for i, (X_ctrl, X_time, y) in enumerate(train_loader):
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            X_ctrl = X_ctrl.to(device)
            X_time = X_time.to(device)
            y = y.to(device)
            y_pred = model(X_ctrl, X_time)
            loss = criterion(y_pred, y)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        pbar.set_postfix({
            'train_loss': np.array(train_loss).mean(),
            'best_loss': best_loss})
        pbar.update(1)

    model.eval()
    with torch.no_grad():
        for j, (X_ctrl, X_time, y) in enumerate(val_loader):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                X_ctrl = X_ctrl.to(device)
                X_time = X_time.to(device)
                y = y.to(device)
                y_pred = model(X_ctrl, X_time)
                loss = criterion(y_pred, y)
                val_loss.append(loss.item())
    if np.array(val_loss).mean() < best_loss:
        best_loss = np.array(val_loss).mean()
        best_model = model

    pbar.set_postfix({
        'train_loss': np.array(train_loss).mean(),
        'val_loss': np.array(val_loss).mean(),
        'best_loss': best_loss})
    pbar.close()

x = torch.rand(1, 17*7).float()
t = torch.rand(1, 1).float()

best_model = best_model.cpu()
with torch.no_grad():
    trace = torch.jit.trace(best_model, (x, t))
torch.jit.save(trace, f"/workdir/paper/models/cond_unet_o3_{INDEX}.pth")
