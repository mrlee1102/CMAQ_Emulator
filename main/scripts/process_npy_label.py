#!/usr/bin/env python3
import os
import glob
import numpy as np
from tqdm import tqdm

# 0) 경로 설정
IN_DIR   = "/mnt/dsk1/mrlee/ncf_dataset/filtered"   # 필터된 NPZ (o3_map 기준)
OUT_DIR  = "/mnt/dsk1/mrlee/ncf_dataset/labels"     # 실제 라벨 NPZ
LINK_DIR = "/home/user/workdir/CMAQ_Emulator/ncf_dataset/labvels"  # 심볼릭 링크

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LINK_DIR, exist_ok=True)

# 1) 8h 윈도우 시작 시간
WINDOW_STARTS = [0, 6, 12, 18]

# 2) (24,H,W) 블록에서 4개 윈도우 평균 맵 계산
def compute_avg_maps(daily_block):
    # daily_block: (24,H,W)
    return [ daily_block[s:s+8].mean(axis=0) for s in WINDOW_STARTS ]

# 3) 모든 RSM NPZ 순회
for npz_fp in tqdm(sorted(glob.glob(os.path.join(IN_DIR, "RSM_*.npz")))):
    key = os.path.splitext(os.path.basename(npz_fp))[0]

    data = np.load(npz_fp, allow_pickle=True)
    o3_map = data['o3_map']  # object array or numeric array

    # object array 면 각 season별 원소로 가져오기
    if o3_map.dtype == object:
        seasons_blocks = list(o3_map)
    else:
        seasons_blocks = [o3_map[i] for i in range(o3_map.shape[0])]

    labels = []
    # ---- 디버깅용 출력 ----
    print(f"\n>>> {key}: found {len(seasons_blocks)} seasons_blocks:")
    for i, sb in enumerate(seasons_blocks):
        print(f"    season[{i}] type={type(sb)}, len={len(sb) if hasattr(sb,'__len__') else 'N/A'}, sb.dtype={getattr(sb, 'dtype', None)}")
    print(">>> 시작 shape 디버깅 완료\n")
    # -----------------------

    # 4) 시즌별로 가변 길이 day -> label 선택
    for season_idx, season_block in enumerate(seasons_blocks):
        # 만약 list 형태라면 array 로 바꿔 보자
        if isinstance(season_block, list):
            print(f"DEBUG: season[{season_idx}] is list, stacking into array...")
            try:
                season_block = np.stack(season_block, axis=0)
                print(f"DEBUG: season[{season_idx}] new shape = {season_block.shape}")
            except Exception as e:
                print(f"ERROR stacking season[{season_idx}]: {e}")
                continue

        # 이제 season_block 은 numpy array
        # shape 확인
        print(f"DEBUG: season[{season_idx}] final array shape = {season_block.shape}")

        # season_block: (days,24,H,W) or (days,H,W)
        n_days = season_block.shape[0]
        season_labels = []

        for d in range(n_days):
            blk = season_block[d]
            if blk.ndim == 3:  # (24,H,W)
                avg_maps = compute_avg_maps(blk)
            else:             # (H,W) 이미 일평균인 경우
                avg_maps = [ blk ]

            # 전체 그리드 평균
            means = [ m.mean() for m in avg_maps ]
            best_idx = int(np.argmax(means))
            season_labels.append(avg_maps[best_idx])

        # (n_days,H,W) 배열로
        labels.append(np.stack(season_labels, axis=0).astype(np.float32))

    # 5) 시즌별 배열을 하나의 NPZ 로 저장 (key: label_s0, label_s1...)
    out_fp = os.path.join(OUT_DIR, f"{key}_label.npz")
    savez_kwargs = { f"label_s{i}": labels[i] for i in range(len(labels)) }
    np.savez_compressed(out_fp, **savez_kwargs)

    # 6) 심볼릭 링크
    link_fp = os.path.join(LINK_DIR, os.path.basename(out_fp))
    if os.path.islink(link_fp) or os.path.exists(link_fp):
        os.remove(link_fp)
    os.symlink(out_fp, link_fp)

    tqdm.write(f"{key}: saved labels → {out_fp} (link → {link_fp})")
