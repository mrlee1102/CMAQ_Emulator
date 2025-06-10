import re, glob, pandas as pd

# 1) 헤더 파일 정보
headers = sorted(glob.glob('pm25_model_compile/params/layer_*_params_*.h'))
header_info = []
for path in headers:
    m = re.search(r'layer_(\d+)_params_([^\.]+)\.h', path)
    if not m: continue
    header_info.append({
        'index': int(m.group(1)),
        'name':  m.group(2),
        'header': path
    })
df_h = pd.DataFrame(header_info)

# 2) forward.c 에서 쓰는 변수 목록 (weight, bias, gamma, beta, mean, variance)
uses = open('uses_vars.txt').read().splitlines()
use_info = []
for u in uses:
    m = re.match(
        r'layer_(\d+)_([a-z0-9_]+)_(weight|bias|gamma|beta|mean|variance)$', 
        u
    )
    if not m: 
        continue
    idx, name, typ = int(m.group(1)), m.group(2), m.group(3)
    use_info.append({'index': idx, 'name': name, typ: u})
df_u = pd.DataFrame(use_info)

# 3) 머지 후 pivot
df = df_h.merge(df_u, on=['index','name'], how='outer')

df_pivot = (df
    .pivot_table(
        index=['index','name','header'],
        values=['weight','bias','gamma','beta','mean','variance'],
        aggfunc='first'
    )
    .reset_index()
    .sort_values(['index','name'])
)

# 4) CSV 출력 & 화면에 보기
df_pivot.to_csv('mapping_table_full.csv', index=False)
print(df_pivot.to_string(index=False))
