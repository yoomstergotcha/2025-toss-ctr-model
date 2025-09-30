"""
folds.parquet 기반으로 train/val 데이터를 나눠 parquet로 저장합니다.

사용 예시:
  python scripts/apply_split.py \
    --train_parquet data/raw/train.parquet \
    --folds data/splits/kfold_v1/folds.parquet \
    --fold 0 \
    --out_dir data/processed/fold0
"""
import argparse
from pathlib import Path
import polars as pl

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_parquet", required=True)
    p.add_argument("--folds", required=True)           # folds.parquet
    p.add_argument("--fold", type=int, required=True)  # e.g., 0
    p.add_argument("--out_dir", required=True)
    p.add_argument("--compression", default="zstd")
    p.add_argument("--no_statistics", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # validation RID들만 (작은 테이블로 유지)
    val_ids = (
        pl.scan_parquet(args.folds)
          .filter(pl.col("fold") == args.fold)
          .select("RID")
    )

    # 원본을 lazy로 스캔 + 글로벌 RID 부여 (단일 파일일 때 권장)
    train_lazy = pl.scan_parquet(args.train_parquet).with_row_index("RID")

    # validation: semi join (RID 있는 행만 통과)
    val_lazy = train_lazy.join(val_ids, on="RID", how="semi")
    # train: anti join (RID 없는 행만 통과)
    train_lazy = train_lazy.join(val_ids, on="RID", how="anti")

    # 저장 (streaming)
    kwargs = {"compression": args.compression, "statistics": (not args.no_statistics)}
    (out / "val.parquet").unlink(missing_ok=True)
    (out / "train.parquet").unlink(missing_ok=True)
    val_lazy.sink_parquet(out / "val.parquet", **kwargs)
    train_lazy.sink_parquet(out / "train.parquet", **kwargs)

    print(f"[OK] saved -> {out}/train.parquet, {out}/val.parquet")

if __name__ == "__main__":
    main()