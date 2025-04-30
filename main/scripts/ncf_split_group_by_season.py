#!/usr/bin/env python3
import os
import re
import glob

# 1) 원본 RSM 디렉터리들이 있는 상위 경로
BASE_DIR = "/mnt/dsk0/bggo/CMAQ_dataset/npy/conc/hourly_new"

# 2) 최종 계절별 저장 경로
DEST_BASE = "/home/user/workdir/CMAQ_Emulator/ncf_dataset/season"

# 3) Julian day 범위로 정의한 4개 계절 구간 (start, end)
SEASONS = [
    (2012357, 2013031),  # Season 1: 12/22 ~  1/31
    (2013081, 2013120),  # Season 2:  3/22 ~  4/30
    (2013172, 2013212),  # Season 3:  6/21 ~  7/31
    (2013264, 2013304),  # Season 4:  9/21 ~ 10/31
]

# 4) 계절별 디렉터리 및 RSM 서브디렉터리 생성
for season_idx in range(1, len(SEASONS) + 1):
    for rsm_idx in range(1, 120):
        season_rsm_dir = os.path.join(DEST_BASE,
                                       f"season_{season_idx}",
                                       f"RSM_{rsm_idx}")
        os.makedirs(season_rsm_dir, exist_ok=True)

# 5) RSM_1 ~ RSM_119 순회 및 심볼릭 링크 생성
for rsm_idx in range(1, 120):
    src_dir = os.path.join(BASE_DIR, f"RSM_{rsm_idx}")
    if not os.path.isdir(src_dir):
        continue

    # 모든 .npy 파일을 찾아서
    for src_fp in glob.glob(os.path.join(src_dir, "*.npy")):
        fname = os.path.basename(src_fp)
        # 파일명에서 Julian day 추출 (예: ...ACONC.2012357.npy)
        m = re.search(r"\.(\d{7})\.npy$", fname)
        if not m:
            print(f"  → 건너뜀: 패턴 불일치 {fname}")
            continue

        jday = int(m.group(1))

        # 어느 계절에 속하는지 판별
        for season_idx, (start, end) in enumerate(SEASONS, 1):
            if start <= end:
                in_range = (start <= jday <= end)
            else:
                # 연도 경계(12월→다음해 1월) 처리
                in_range = (jday >= start) or (jday <= end)

            if in_range:
                # season_n/RSM_idx 서브폴더에 심볼릭 링크 생성
                dest_dir = os.path.join(DEST_BASE,
                                         f"season_{season_idx}",
                                         f"RSM_{rsm_idx}")
                dest_fp = os.path.join(dest_dir, fname)
                # 이미 링크가 없을 경우만 생성
                if not os.path.exists(dest_fp):
                    os.symlink(src_fp, dest_fp)
                break