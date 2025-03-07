import os
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# 데이터 로드 및 전처리
# -------------------------------

# 배출량 데이터 (ctrl) CSV 파일 로드
dataset_2013 = pd.read_csv('/home/user/workdir/CMAQ_Emulator/main/resources/ctrl/dataset_for_o3_interaction_v1.csv')
ctrl_data = pd.concat([dataset_2013], axis=0)
ctrl_data = ctrl_data.reset_index(drop=True).values  # numpy array로 변환
emis_data = ctrl_data[:, :17*9]  # 입력 특성 (예: 17*9 차원)

# netCDF 파일로부터 O3 농도 데이터 로드
label_path_2013 = '/home/user/workdir/CMAQ_Emulator/main/datasets/concentration/2013'
label_path = []
for i in range(1, 120): 
    label_path.append(os.path.join(label_path_2013, '1.00', f'ACONC.{i}'))
label_data = []
for path in label_path:
    with nc.Dataset(path) as f:
        # O3 농도 데이터 로드 (82x67x1)
        label_data.append(f.variables['O3'][:].data.squeeze())
label_data = np.array(label_data).reshape(len(label_data), 82, 67, 1)

# 여기서 목표는 각 배출 시나리오에 따른 전체 공간 분포 예측이므로,
# label 데이터는 그대로 사용 (평균 내지 않음)
# Train/Test 분할
test_split = 0.4  # 테스트 데이터 비율 (예: 40%)
random_seed = 42  # 랜덤 시드
X_train, X_test, y_train, y_test = train_test_split(emis_data, label_data, test_size=test_split, random_state=random_seed)

# 다중 출력 회귀를 위해 타겟 데이터를 2차원 배열로 reshape (n_samples, 82*67)
y_train_reshaped = y_train.reshape(y_train.shape[0], -1)
y_test_reshaped = y_test.reshape(y_test.shape[0], -1)

# -------------------------------
# 모델 1: Lasso 회귀 (MultiOutputRegressor 사용)
# -------------------------------
print("=== Lasso Regression ===")
lasso_alpha = 0.1  # alpha 값은 튜닝 필요
base_lasso = Lasso(alpha=lasso_alpha, random_state=random_seed)
lasso_model = MultiOutputRegressor(base_lasso)
lasso_model.fit(X_train, y_train_reshaped)
y_pred_lasso = lasso_model.predict(X_test)
# 예측 결과를 원래의 4차원 형태로 복원
y_pred_lasso = y_pred_lasso.reshape(y_test.shape)

print("Lasso MAE:", mean_absolute_error(y_test.ravel(), y_pred_lasso.ravel()))
print("Lasso R^2:", r2_score(y_test.ravel(), y_pred_lasso.ravel()))

# -------------------------------
# 모델 2: Decision Tree 회귀
# -------------------------------
print("\n=== Decision Tree Regression ===")
tree_max_depth = 5  # 예시: 최대 깊이 5 (튜닝 필요)
tree_model = DecisionTreeRegressor(max_depth=tree_max_depth, random_state=random_seed)
tree_model.fit(X_train, y_train_reshaped)
y_pred_tree = tree_model.predict(X_test)
y_pred_tree = y_pred_tree.reshape(y_test.shape)

print("Decision Tree MAE:", mean_absolute_error(y_test.ravel(), y_pred_tree.ravel()))
print("Decision Tree R^2:", r2_score(y_test.ravel(), y_pred_tree.ravel()))

# -------------------------------
# 결과 시각화: 산점도 플롯 (예측 결과를 1차원으로 flatten하여 비교)
# -------------------------------
def plot_scatter(y_true, y_pred, title, save_path):
    plt.figure(figsize=(6, 6))
    # flatten하여 1차원 배열로 산점도 생성
    plt.scatter(y_true.ravel(), y_pred.ravel(), alpha=0.7, color='blue', edgecolors='k')
    # 이상적인 예측 대각선
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel('True O3 [µg/m³]')
    plt.ylabel('Predicted O3 [µg/m³]')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Lasso Regression 결과 산점도 저장
plot_scatter(y_test, y_pred_lasso, 
             "Lasso Regression: True vs. Predicted Spatial O3 Distribution", 
             "lasso_result.png")

# Decision Tree Regression 결과 산점도 저장
plot_scatter(y_test, y_pred_tree, 
             "Decision Tree Regression: True vs. Predicted Spatial O3 Distribution", 
             "DT_result.png")
