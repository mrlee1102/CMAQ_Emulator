''' Import '''
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib as mpl
import tensorflow as tf
import matplotlib.pyplot as plt
import sys, os

def resource_path(relative_path):
    """
    실행 파일 내부 또는 개발 환경에서 파일 경로를 올바르게 반환합니다.
    PyInstaller 사용 시 sys._MEIPASS에 압축해제된 임시 경로가 설정됩니다.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

''' Local Setup: 리소스 경로 설정 '''
ctprvn_shp = resource_path('resources/geom/ctp_rvn.shp')
emis_csv = resource_path('resources/ctrl/precursor_control_2019_4input_scaled_o3.csv')
grid_alloc_csv = resource_path('resources/geom/grid_allocation.csv')
model_path = resource_path('final_model')

''' GPU Setup (필요시 활성화) '''
'''
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs")
    except RuntimeError as e:
        print(e)
'''

''' Function Definitions '''
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

def get_ctprvn_map() -> gpd.GeoDataFrame:
    """
    행정구역 경계 정보를 포함한 Shapefile을 로드하여 GeoDataFrame으로 반환합니다.
    """
    # resource_path를 사용하여 동적으로 파일 경로를 처리합니다.
    path = ctprvn_shp
    ctprvn = gpd.read_file(path, encoding='cp949')
    ctprvn.crs = 'EPSG:5179'
    return ctprvn

def get_base_raster(ctprvn: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    기본 그리드(래스터)를 생성하여 GeoDataFrame으로 반환합니다.
    """
    points = [Point(i, j)
              for i in range(-180000, -180000 + 9000 * 67, 9000)
              for j in range(-585000, -585000 + 9000 * 82, 9000)]
    grid_data = gpd.GeoDataFrame(points, geometry='geometry', columns=['geometry'])
    grid_data.crs = ctprvn.to_crs(proj).crs
    grid_data['x_m'] = grid_data.geometry.x
    grid_data['y_m'] = grid_data.geometry.y
    grid_data['value'] = 0
    grid_data['index'] = grid_data.index
    return grid_data

def get_region_pixel_indices() -> list:
    """
    행정구역 정보와 그리드를 결합하여 각 지역별 픽셀 인덱스 리스트를 반환합니다.
    """
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
    merged = pd.merge(left=grid_data, right=gdf_joined[gdf_joined_loc],
                      how='left', left_on='index', right_on='index_right0')
    indices = gpd.GeoDataFrame(merged, geometry='geometry').dropna()
    pixel_indices = [
        [(idx % 82, idx // 82) for idx in indices.loc[indices.CTP_KOR_NM == cities[region]].index.tolist()]
        for region, _ in cities.items()
    ]
    return pixel_indices

''' Load Administrative Boundaries and Grid Allocation Data '''
ctprvn = get_ctprvn_map()
ctprvn_proj = ctprvn.to_crs(proj)

grid_alloc = (
    pd.read_csv(grid_alloc_csv)
    .sort_values(by=['Row', 'Column', 'Ratio'], ascending=[True, True, False])
    .drop_duplicates(subset=['Row', 'Column'], keep='first')
    .reset_index(drop=True)
)

pixel_indices = get_region_pixel_indices()
total_index = []
for idx, grids in enumerate(pixel_indices):
    for grid in grids:
        total_index.append([
            grid[1], grid[0], 100.0, atob[idx], region_columns[atob[idx]]
        ])
total_index = pd.DataFrame(total_index, columns=grid_alloc.columns)

grid_alloc = pd.concat([
    grid_alloc.drop(columns=['Ratio', 'Region_Name']),
    total_index.drop(columns=['Ratio', 'Region_Name'])
]).sort_values(by=['Region_Code']).drop_duplicates().reset_index(drop=True)
grid_alloc[['Row', 'Column']] = grid_alloc[['Row', 'Column']] - 1

row_indices, col_indices = zip(*grid_alloc[['Row', 'Column']].values)
offset_x, offset_y = 4500, 4500

mask = np.zeros((82, 67))
mask[list(row_indices), list(col_indices)] = 1

cmap_white = mpl.colormaps['jet']
cmap_white.set_under('white')

''' ===== Load Nitrate Prediction Model ===== '''
model = tf.keras.models.load_model(model_path)

''' ===== Input Data Setting ===== '''
emis_ctrl_2019_10 = pd.read_csv(emis_csv)
emis_ctrl_2019_10['Boundary'] = 1.0

ctrl_data = pd.concat([emis_ctrl_2019_10], axis=0)
ctrl_data = ctrl_data.reset_index(drop=True).values

input_emis_data = ctrl_data[:, :17*5]
input_boundary_data = ctrl_data[:, 17*5]

''' ===== Prediction & Result ===== '''
pred_nitrate = model.predict([input_emis_data, input_boundary_data])
pred_nitrate = pred_nitrate.squeeze()

mask_expanded = np.repeat(mask[np.newaxis, :, :], repeats=pred_nitrate.shape[0], axis=0)
pred_conc_map = np.where(mask_expanded == 1, pred_nitrate, 0)

''' 1. Predicted Nitrate Concentration Map '''
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.imshow(
    pred_conc_map[::-1],
    cmap=cmap_white,
    vmin=0.001,
    extent=(-180000+offset_x, 414000+offset_x, -585000+offset_y, 144000+offset_y),
    interpolation='none'
)
# 행정구역 경계 표시
ctprvn_proj.boundary.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, alpha=0.25)
ax.set_xlim(-180000, 414000)
ax.set_ylim(-585000, 144000)
ax.set_xlabel('Longitude [°]')
ax.set_ylabel('Latitude [°]')
ax.set_title("Conc. Map of Nitrate Prediction")
ax.grid(alpha=0.25, color='silver')
ax.set_xticklabels([f"{i}" for i in range(124, 132, 1)])
ax.set_yticklabels([f"{i}" for i in range(32, 40, 1)])
cbaxes = ax.inset_axes([0.6, 0.15, 0.35, 0.03])
cb = plt.colorbar(ax.images[0], cax=cbaxes, orientation='horizontal', label='Nitrate ($\mu g/m^3$)', extend='min')
cb.set_label(label='conc. [$\mu \mathrm{g}/\mathrm{m}^3$]')
plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(wspace=-0.4, hspace=0.2)
plt.savefig('savefig_edgecolor.png', facecolor='#eeeeee', edgecolor='blue')
# plt.show()
