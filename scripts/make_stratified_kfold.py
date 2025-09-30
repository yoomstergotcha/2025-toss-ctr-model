"""
RID(행 고유 ID)를 Polars로 생성하고, clicked 비율을 유지하는 Stratified K-Fold를 만듭니다.

생성물:
- data/splits/kfold_v1/rid_clicked.parquet   # RID↔clicked 매핑
- data/splits/kfold_v1/folds.parquet         # 각 RID의 fold 번호

예시 실행:
  python scripts/make_stratified_kfold.py \
    --train_parquet data/raw/train.parquet \
    --output_dir data/splits/kfold_v1 \
    --n_splits 5 \
    --seed 42
"""
import argparse
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedKFold

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_parquet", type=str, required=True, help="data/raw/train.parquet 경로")
    p.add_argument("--output_dir", type=str, required=True, help="split 파일을 저장할 디렉토리")
    p.add_argument("--n_splits", type=int, default=5, help="K-Fold 개수 (기본 5)")
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 라벨만 lazy로 로딩 (메모리 안전)
    lazy = pl.scan_parquet(args.train_parquet)
    if "clicked" not in lazy.columns:
        raise SystemExit("`clicked` 컬럼을 찾을 수 없습니다. train.parquet에 존재해야 합니다.")

    # 2) collect()로 materialize 후, 안정적 surrogate ID(RID) 부여
    labels = lazy.select(pl.col("clicked")).collect()
    labels = labels.with_row_index("RID")  # 0..N-1

    # 3) RID↔clicked 저장
    rid_clicked_path = out_dir / "rid_clicked.parquet"
    labels.select(["RID", "clicked"]).write_parquet(rid_clicked_path)

    # 4) Stratified K-Fold 수행 (scikit-learn)
    rid = labels.get_column("RID").to_numpy()
    y = labels.get_column("clicked").cast(pl.Int8).to_numpy()

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_assign = np.full(rid.shape[0], fill_value=-1, dtype=np.int16)
    for k, (_, val_idx) in enumerate(skf.split(rid, y)):
        fold_assign[val_idx] = k

    # 5) folds.parquet 저장 (RID별 fold 번호)
    folds_path = out_dir / "folds.parquet"
    pl.DataFrame({"RID": rid, "fold": fold_assign}).write_parquet(folds_path)

    print(f"[OK] Saved:\n - {rid_clicked_path}\n - {folds_path}")

if __name__ == "__main__":
    main()