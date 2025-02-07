import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/home/user/workdir/main/src/'))))

import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset

from sklearn.model_selection import train_test_split
import tensorflow as tf
from src.model.cmaqnet_cond_unet import build_model
from sklearn.metrics import r2_score

import geopandas as gpd
from shapely.geometry import Point
import matplotlib as mpl
import matplotlib.pyplot as plt

# 저장 경로 설정 및 디렉토리 생성 (없으면 생성)
output_dir = '/home/user/workdir/main/nitrate_training/bootstrapping_result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

'''
Prediction Data Preparation & Model Loading
'''
# 1. 입력 데이터 처리
# --- 배출 제어 데이터 (emission & boundary) ---
emis_ctrl_2019_10 = pd.read_csv('/home/user/workdir/main/resources/ctrl/precursor_control_2019_4input_scaled_o3.csv')
emis_ctrl_2019_10['Boundary'] = 1.0

ctrl_data = pd.concat([emis_ctrl_2019_10], axis=0)
ctrl_data = ctrl_data.reset_index(drop=True).values
pred_emis_data = ctrl_data[:, :17*5]
pred_boundary_data = ctrl_data[:, 17*5]

# --- 농도 데이터 (netCDF 파일에서 Nitrate 값 추출) ---
base_path_2019 = "/home/user/workdir/main/datasets/concentration/2019/"

conc_path = []
for i in range(1, 120):
    conc_path.append(os.path.join(base_path_2019, '1.00', f'ACONC.{i}'))

conc_data = []
for path in conc_path:
    with nc.Dataset(path) as f:
        conc_data.append(f.variables['Nitrate'][:].data.squeeze())
conc_data = np.array(conc_data).reshape(len(conc_path), 82, 67, 1)

# 2. 사전 학습된 모델 불러오기 (모델은 이미 학습된 상태)
model_path = '/home/user/workdir/main/src/model/nitrate/scaled_o3/input5/final_model'
model = tf.keras.models.load_model(model_path)

# =========================================================================================

'''
마스킹 및 지리정보 처리
'''
# 지도 관련 파라미터
proj = '+proj=lcc +lat_1=30 +lat_2=60 +lon_1=126 +lat_0=38 +lon_0=126 +ellps=GRS80 +units=m'
atob = {
    0: 'G', 1: 'F', 2: 'K', 3: 'J', 4: 'E', 5: 'D',
    6: 'O', 7: 'C', 8: 'A', 9: 'Q', 10: 'P', 11: 'B',
    12: 'M', 13: 'L', 14: 'N', 15: 'I', 16: 'H'}
region_columns = {
    'A': 'Seoul City', 'B': 'Incheon City', 'C': 'Busan City', 'D': 'Daegu City',
    'E': 'Gwangju City', 'F': 'Gyeonggi-do', 'G': 'Gangwon-do', 'H': 'Chungbuk-do',
    'I': 'Chungnam-do', 'J': 'Gyeongbuk-do', 'K': 'Gyeongnam-do', 'L': 'Jeonbuk-do',
    'M': 'Jeonnam-do', 'N': 'Jeju-do', 'O': 'Daejeon City', 'P': 'Ulsan City', 'Q': 'Sejong City'}

# grid allocation 데이터 로드 및 마스킹 처리
grid_alloc = (
    pd.read_csv('/home/user/workdir/main/resources/geom/grid_allocation.csv')
    .sort_values(by=['Row', 'Column', 'Ratio'], ascending=[True, True, False])
    .drop_duplicates(subset=['Row', 'Column'], keep='first')
    .reset_index(drop=True)
)

def get_ctprvn_map() -> gpd.GeoDataFrame:
    path = '/home/user/workdir/main/resources/geom/ctp_rvn.shp'
    ctprvn = gpd.GeoDataFrame.from_file(path, encoding='cp949')
    ctprvn.crs = 'EPSG:5179'
    return ctprvn

def get_base_raster(ctprvn: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = [Point(i, j)
              for i in range(-180000, -180000 + 9000 * 67, 9000)
              for j in range(-585000, -585000 + 9000 * 82, 9000)]
    grid_data = gpd.GeoDataFrame(points, geometry='geometry', columns=['geometry'])
    grid_data.crs = ctprvn.to_crs(proj).crs
    grid_data.loc[:, 'x_m'] = grid_data.geometry.x
    grid_data.loc[:, 'y_m'] = grid_data.geometry.y
    grid_data.loc[:, 'value'] = 0
    grid_data.loc[:, 'index'] = grid_data.index
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
    gdf_joined = gpd.sjoin(get_ctprvn_map(), grid_data.to_crs(5179), predicate='contains')
    indices = gpd.GeoDataFrame(pd.merge(
        left=grid_data, right=gdf_joined.loc[:, gdf_joined_loc],
        how='left', left_on='index', right_on='index_right0'
    ), geometry='geometry').dropna()
    pixel_indices = [
        [(idx % 82, idx // 82) for idx in indices.loc[indices.CTP_KOR_NM == cities[region]].index.tolist()]
        for region, _ in cities.items()
    ]
    return pixel_indices

pixel_indices = get_region_pixel_indices()
total_index = []
for idx, grids in enumerate(pixel_indices):
    for grid in grids:
        total_index.append([grid[1], grid[0], 100.0, atob[idx], region_columns[atob[idx]]])
total_index = pd.DataFrame(total_index, columns=grid_alloc.columns)
grid_alloc = pd.concat([
    grid_alloc.drop(columns=['Ratio', 'Region_Name']),
    total_index.drop(columns=['Ratio', 'Region_Name'])
]).sort_values(by=['Region_Code']).drop_duplicates().reset_index(drop=True)
grid_alloc[['Row', 'Column']] = grid_alloc[['Row', 'Column']] - 1

row_indices, col_indices = zip(*grid_alloc[['Row', 'Column']].values)

mask = np.ones((82, 67))
mask[row_indices, col_indices] = 1

# =====================

n_iterations = 20   # 예를 들어, 20번의 반복 (원하는 만큼 늘릴 수 있음)
nmae_list = []      # 각 반복에서의 NMAE를 저장할 리스트
all_pred_list = []
all_true_list = []

for i in range(n_iterations):
    # 매 반복마다 다른 랜덤 시드로 테스트 셋 분할 (전체 데이터셋에서 무작위로 추출)
    _, X_emis_test, _, X_boundary_test, _, y_test = train_test_split(
        pred_emis_data, pred_boundary_data, conc_data,
        test_size=0.4, random_state=(i))  # random_state를 i로 변경
    
    # 모델 예측 수행
    y_preds = model.predict([X_emis_test, X_boundary_test])
    y_pred = y_preds.squeeze()
    y_true = y_test.squeeze()

    mask_expanded = np.repeat(mask[np.newaxis, :, :], repeats=y_true.shape[0], axis=0)
    pred_conc_map = np.where(mask_expanded == 1, y_pred, 0)
    true_conc_map = np.where(mask_expanded == 1, y_true, 0)
    
    # 전체 픽셀 단위로 NMAE 계산
    pred_flat = pred_conc_map.reshape(-1)
    true_flat = true_conc_map.reshape(-1)
    mae = np.mean(np.abs(true_flat - pred_flat))
    mean_true = np.mean(true_flat)
    nmae = mae / mean_true if mean_true != 0 else np.nan
    nmae_list.append(nmae)
    all_pred_list.append(pred_flat)
    all_true_list.append(true_flat)
    
    print(f"Iteration {i+1}: NMAE = {nmae:.3f}")

nmae_array = np.array(nmae_list)
global_mean = np.mean(nmae_array)
ci_lower = np.percentile(nmae_array, 2.5)
ci_upper = np.percentile(nmae_array, 97.5)

print("\nGlobal Performance over all iterations:")
print(f"Mean NMAE: {global_mean:.3f}")
print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
# -----------------------------------------------------------------------------
# Visualization 1: Bar Chart with Error Bar (각 반복의 NMAE 및 글로벌 평균, CI 오버레이)
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
iterations = np.arange(1, n_iterations+1)
plt.bar(iterations, nmae_array, color='skyblue', edgecolor='black', label='Iteration NMAE')
plt.axhline(global_mean, color='red', linestyle='--', linewidth=2, label=f'Global Mean: {global_mean:.4f}')
# 전체 반복 구간을 CI로 표시 (x축 전체에 걸쳐 수평 영역으로 표시)
plt.fill_between(iterations, ci_lower, ci_upper, color='green', alpha=0.2, label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.3f}]')
plt.xlabel('Iteration')
plt.ylabel('NMAE')
plt.title('Global NMAE over Multiple Test Splits')
plt.legend()
plt.savefig(os.path.join(output_dir, 'global_nmae_bar.png'), dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# Visualization 2: Histogram of NMAE Distribution
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.hist(nmae_array, bins=10, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(global_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {global_mean:.4f}')
plt.axvline(ci_lower, color='green', linestyle='dashed', linewidth=2, label=f'2.5%: {ci_lower:.5f}')
plt.axvline(ci_upper, color='green', linestyle='dashed', linewidth=2, label=f'97.5%: {ci_upper:.5f}')
plt.xlabel('NMAE')
plt.ylabel('Density')
plt.title('Histogram of Global NMAE')
plt.legend()
plt.savefig(os.path.join(output_dir, 'global_nmae_histogram.png'), dpi=300, bbox_inches='tight')
plt.close()

def plot_scatter(ax, y_true, y_pred):
    # 선형 회귀선을 np.polyfit으로 계산 (rcond 조정하여 안정화)
    r_x, r_y = np.polyfit(y_true, y_pred, 1, rcond=1e-5)
    # 2D 히스토그램을 그려서 데이터 밀도를 표시
    h = ax.hist2d(
        y_true, y_pred,
        bins=150, cmap='jet', cmin=1,
        norm=mpl.colors.LogNorm(vmin=1, vmax=1000),
    )
    # 회귀선을 추가
    ax.plot(
        y_true, r_x*y_true + r_y,
        color='red', label=f"y={r_x:.4f}x+{r_y:.4f}"
    )
    # R² 스코어 계산
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.text(
        0.05, 0.95, f"$R^2={r2:.4f}$\nSlope={r_x:.4f}\nIntercept={r_y:.4f}",
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )
    ax.grid(alpha=0.25)
    return ax

def plot_residuals(ax, residuals):
    ax.hist(residuals, bins=50, density=True, alpha=0.7, color='salmon', edgecolor='black')
    ax.set_xlabel('Residuals (True - Predicted)')
    ax.set_ylabel('Density')
    ax.set_title('Residuals Distribution')
    ax.grid(alpha=0.25)
    return ax

# --- Visualization 3: Scatter Plot of All Predicted vs. True Values ---

# 모든 반복에서 flatten된 예측값과 실제값을 하나의 배열로 합침
all_pred = np.concatenate(all_pred_list, axis=0)
all_true = np.concatenate(all_true_list, axis=0)

fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
ax = plot_scatter(ax, all_true, all_pred)
ax.set_title("Scatter Plot of All Predicted vs. True Values")
ax.set_xlabel('True Nitrate [$\mu g/m^3$]')
ax.set_ylabel('Predicted Nitrate [$\mu g/m^3$]')
# inset colorbar 추가
cbaxes = ax.inset_axes([0.5, 0.2, 0.35, 0.03])
cb = plt.colorbar(
    mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=1, vmax=1000), cmap='jet'),
    cax=cbaxes, orientation='horizontal'
)
cb.set_label(label='Number of samples', fontsize=10)
plt.savefig(os.path.join(output_dir, "global_scatter_all.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- Visualization 4: Residuals Analysis ---

# residuals 계산 (모든 반복에서의 예측값과 실제값)
residuals = all_true - all_pred

fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
ax = plot_residuals(ax, residuals)
plt.savefig(os.path.join(output_dir, "global_residuals_histogram.png"), dpi=300, bbox_inches='tight')
plt.close()