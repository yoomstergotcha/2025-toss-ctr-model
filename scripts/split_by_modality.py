#!/usr/bin/env python3
"""
Split a large Parquet dataset into modality-specific shards:
- cont_*.parquet        (scaled continuous)
- cat_hash_*.parquet    (hashed categorical indices)
- cat_vocab_*.parquet   (vocab-encoded integer IDs)
- seq_*.parquet         (sequence features as arrow lists of int32)

Two-pass pipeline:
  Pass 1: compute scaling stats + build vocabs
  Pass 2: transform + write sharded outputs

예시 실행: (맥북용임)
python scripts/split_by_modality.py \
  --input data/raw/train_9M.parquet \ 
  --output-dir data/processed/train_modality \
  --config configs/config.yaml
  
테스트셋일 경우:
python scripts/split_by_modality.py \
  --input data/raw/test.parquet \
  --output-dir data/processed/test_modality \
  --config configs/config_test.yaml
  """

import argparse, json
from pathlib import Path
from typing import Dict, List, Any
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import yaml

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def stable_hash_array(arr: pa.Array, seed: int) -> pa.Array:
    # uint64 hash -> 범주형 해시 버킷 인덱싱에 사용
    h = pc.hash_array(arr, seed=seed)  # uint64
    return pc.fill_null(h, 0)

def to_int_list_array(arr: pa.Array) -> pa.Array:
    # 시퀀스 칼럼 
    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        py = arr.to_pylist()
        py2 = [None if v is None else [int(x) for x in v] for v in py]
        return pa.array(py2, type=pa.list_(pa.int32()))
    if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        py = arr.to_pylist()
        out = []
        for s in py:
            if s is None or s == "": out.append(None)
            else:
                parts = [p for p in str(s).split(",") if p != ""]
                out.append([int(float(p)) for p in parts])
        return pa.array(out, type=pa.list_(pa.int32()))
    py = arr.to_pylist()
    out = [None if v is None else [int(v)] for v in py]
    return pa.array(out, type=pa.list_(pa.int32()))

# 연속형 스케일링에 필요한 통계치 계산
def compute_cont_stats(table: pa.Table, plan: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for spec in plan:
        col, method = spec["name"], spec["method"].lower()
        if col not in table.column_names: continue
        x = pc.cast(table[col], pa.float64(), safe=False)
        stats = {}
        if "standard" in method:
            stats["mean"] = float(pc.mean(x).as_py() or 0.0)
            stats["std"]  = float(pc.stddev(x, ddof=0).as_py() or 1.0) or 1.0
        if "robust" in method:
            med = pc.quantile(x, q=0.5).as_py()
            q1  = pc.quantile(x, q=0.25).as_py()
            q3  = pc.quantile(x, q=0.75).as_py()
            stats["median"] = float(med or 0.0)
            stats["iqr"]    = float((q3 - q1) if (q3 is not None and q1 is not None) else 1.0) or 1.0
        if "log1p" in method:
            mn = pc.min(x).as_py()
            stats["min"] = float(mn if mn is not None else 0.0)
        out[col] = stats
    return out

# 연속형 스케일 변환 수행
def transform_cont(arr: pa.Array, method: str, stats: Dict[str, float]) -> pa.Array:
    x = pc.cast(arr, pa.float64(), safe=False)
    if "log1p" in method:
        minv = stats.get("min", 0.0)
        shift = -minv + 1e-6 if minv is not None and minv <= 0 else 0.0
        x = pc.log1p(pc.add(x, pa.scalar(shift)))
    if "standard" in method:
        mean, std = stats.get("mean", 0.0), stats.get("std", 1.0) or 1.0
        x = pc.divide(pc.subtract(x, pa.scalar(mean)), pa.scalar(std))
    elif "robust" in method:
        med, iqr = stats.get("median", 0.0), stats.get("iqr", 1.0) or 1.0
        x = pc.divide(pc.subtract(x, pa.scalar(med)), pa.scalar(iqr))
    return x.cast(pa.float32())

def clip_lists(lst_arr: pa.Array, max_len: int, keep: str = "tail") -> pa.Array:
    py = lst_arr.to_pylist()
    tail = (keep.lower() != "head")
    out = []
    for v in py:
        if v is None:
                out.append(None)
        else:
                out.append(v[-max_len:] if tail and len(v) > max_len else v[:max_len])
    return pa.array(out, type=pa.list_(pa.int32()))

def hash_list_tokens(lst_arr: pa.Array, seed: int, buckets: int) -> pa.Array:
    py = lst_arr.to_pylist()
    out = []
    for v in py:
        if v is None:
            out.append(None)
        else:
            hv = []
            for t in v:
                h = pc.hash_array(pa.array([int(t)]), seed=seed)[0].as_py()  # uint64
                hv.append(int(h % buckets))
            out.append(hv)
    return pa.array(out, type=pa.list_(pa.int32()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Parquet file or dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    rid_col   = cfg["rid_col"]
    label_col = cfg.get("label_col") or None
    rows_per_chunk = int(cfg.get("rows_per_chunk", 400_000))
    hash_seed      = int(cfg.get("hash_seed", 17))
    default_buckets= int(cfg.get("hash_num_buckets", 1_048_576))

    out_names = cfg["out_names"]
    out_root = Path(args.output_dir)
    cont_dir  = out_root / out_names["cont"]
    hash_dir  = out_root / out_names["cat_hash"]
    vocab_dir = out_root / out_names["cat_vocab"]
    seq_dir   = out_root / out_names["seq"]
    meta_dir  = out_root / out_names["meta"]
    for d in (cont_dir, hash_dir, vocab_dir, seq_dir, meta_dir): ensure_dir(d)

    cont_plan  = cfg.get("continuous", [])
    hash_plan  = cfg.get("categorical_hash", [])
    vocab_cols = cfg.get("categorical_vocab", [])

    vocab_cfg = cfg.get("vocab_encoding", {})
    default_mode = vocab_cfg.get("default", "auto")
    per_col_mode = vocab_cfg.get("per_column", {})

    raw_seq_cfg    = cfg.get("sequence", {})
    if isinstance(raw_seq_cfg, dict):
        seq_columns   = raw_seq_cfg.get("columns", []) or []
        seq_max_len   = int(raw_seq_cfg.get("max_len", 6000))
        seq_keep      = str(raw_seq_cfg.get("keep", "tail"))
        seq_buckets   = int(raw_seq_cfg.get("buckets", default_buckets)) if "buckets" in raw_seq_cfg else None
    elif isinstance(raw_seq_cfg, (list, tuple)):
        seq_columns   = list(raw_seq_cfg)
        seq_max_len   = 6000
        seq_keep      = "tail"
        seq_buckets   = None
    elif isinstance(raw_seq_cfg, str):
        seq_columns   = [raw_seq_cfg]
        seq_max_len   = 6000
        seq_keep      = "tail"
        seq_buckets   = None
    else:
        seq_columns   = []
        seq_max_len   = 6000
        seq_keep      = "tail"
        seq_buckets   = None

    dataset = ds.dataset(args.input, format="parquet")

    # ---- PASS 1: stats/vocab (full-scan but column-pruned) ----
    # ---- 연속형 통계치 계산 (필요 열만), categorical_vocab auto 모드인 경우 vocab_maps 생성 ----
    cols_for_stats = [s["name"] for s in cont_plan if s["name"] in dataset.schema.names]
    cont_stats = {}
    if cols_for_stats:
        stats_tbl = dataset.to_table(columns=cols_for_stats)
        cont_stats = compute_cont_stats(stats_tbl, cont_plan)

    # auto 모드가 있는 컬럼만 vocab_maps 생성
    need_vocab = any(per_col_mode.get(col, default_mode) == "auto" for col in vocab_cols)
    vocab_maps = {}
    if need_vocab:
        for col in vocab_cols:
            if col not in dataset.schema.names: 
                continue
            if per_col_mode.get(col, default_mode) != "auto":
                continue
            uniq = pc.unique(dataset.to_table(columns=[col])[col])
            py_vals = [v.as_py() for v in uniq if v is not None]
            vocab_maps[col] = {str(v): i+1 for i, v in enumerate(py_vals)}  # 0=OOV/NA

    # 스케일링 통계, vocab 맵, seed, 버킷, 청크크기 기록
    manifest = {
        "rid_col": rid_col,
        "label_col": label_col,
        "cont_stats": cont_stats,
        "vocab_maps": vocab_maps,
        "hash_seed": hash_seed,
        "default_buckets": default_buckets,
        "plans": {
            "continuous": cont_plan,
            "categorical_hash": hash_plan,
            "categorical_vocab": vocab_cols,
            "sequence": {
                "columns": seq_columns,
                "max_len": seq_max_len,
                "keep": seq_keep,
                "buckets": seq_buckets,
            },
        },
        "source": str(args.input),
        "rows_per_chunk": rows_per_chunk,
    }
    (meta_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


    # ---- PASS 2: transform + write (streaming) ----
    # ---- 스트리밍 변환/저장, 필요한 열만 스캔 후 batch 단위 처리 ----
    needed = [rid_col] + ([label_col] if (label_col and label_col in dataset.schema.names) else [])
    needed += [s["name"] for s in cont_plan if s["name"] in dataset.schema.names]
    needed += [s["name"] for s in hash_plan if s["name"] in dataset.schema.names]
    needed += [c for c in vocab_cols if c in dataset.schema.names]
    needed += [c for c in seq_columns if c in dataset.schema.names]
    needed = list(dict.fromkeys(needed))

    scanner = dataset.scanner(columns=needed)
    shard = 0
    for batch in scanner.to_batches(max_chunksize=rows_per_chunk):
        shard += 1
        tbl = pa.Table.from_batches([batch])
        base = {rid_col: tbl[rid_col]}
        if label_col and label_col in tbl.column_names:
            base[label_col] = tbl[label_col]

        # continuous
        cont_cols = []
        for spec in cont_plan:
            col = spec["name"]
            if col not in tbl.column_names: continue
            cont_cols.append(transform_cont(tbl[col], spec["method"].lower(), cont_stats.get(col, {})).rename(col))
        if cont_cols:
            out = pa.table(base)
            for a in cont_cols: out = out.append_column(a.field(0), a)
            pq.write_table(out, cont_dir / f"cont_{shard:05d}.parquet", compression="zstd")

        # categorical_hash
        hash_cols = []
        for spec in hash_plan:
            col = spec["name"]
            if col not in tbl.column_names: continue
            buckets = int(spec.get("buckets", default_buckets))
            h = stable_hash_array(tbl[col], seed=hash_seed)
            mod = pc.mod(h, pa.scalar(buckets, pa.uint64()))
            hash_cols.append(pc.cast(mod, pa.int32()).rename(col))
        if hash_cols:
            out = pa.table(base)
            for a in hash_cols: out = out.append_column(a.field(0), a)
            pq.write_table(out, hash_dir / f"cat_hash_{shard:05d}.parquet", compression="zstd")


        # ----- categorical_vocab (with modes) -----

        voc_cols = []
        for col in vocab_cols:
            if col not in tbl.column_names:
                continue
            mode = per_col_mode.get(col, default_mode)

            if mode == "auto":
                # 기존 동작: Pass1에서 만든 vocab_maps 사용 (0=OOV/NA)
                vocab = vocab_maps.get(col, {})
                py = tbl[col].to_pylist()
                ids = [int(vocab.get(str(v), 0)) if v is not None else 0 for v in py]
                voc_cols.append(pa.array(ids, type=pa.int32()).rename(col))

            elif mode == "identity":
                # 값 그대로 사용, 결측은 0
                arr = pc.fill_null(pc.cast(tbl[col], pa.int64(), safe=False), 0)
                voc_cols.append(pc.cast(arr, pa.int32()).rename(col))

            elif mode == "identity_plus1":
                # 값+1, 결측은 0  →  0을 PAD/OOV로 예약 가능
                # (결측을 -1로 채운 뒤 +1 → 0; 그 외는 +1)
                arr = pc.fill_null(pc.cast(tbl[col], pa.int64(), safe=False), -1)
                arr = pc.add(arr, pa.scalar(1, pa.int64()))
                arr = pc.max(arr, pa.scalar(0, pa.int64()))  # 음수 방지
                voc_cols.append(pc.cast(arr, pa.int32()).rename(col))

            else:
                raise ValueError(f"Unknown vocab_encoding mode: {mode} for col={col}")

        if voc_cols:
            out = pa.table(base)
            for a in voc_cols:
                out = out.append_column(a.field(0), a)
            pq.write_table(out, vocab_dir / f"cat_vocab_{shard:05d}.parquet", compression="zstd")


        # ----- sequence: clip + hash -----

        raw_cols, seq_hash_cols = [], []
        for col in seq_columns:
            if col not in tbl.column_names:
                continue
            lst = to_int_list_array(tbl[col])
            lst = clip_lists(lst, max_len=seq_max_len, keep=seq_keep)
            raw_cols.append((col, lst))
            if seq_buckets is not None:
                hashed = hash_list_tokens(lst, seed=hash_seed, buckets=seq_buckets)
                seq_hash_cols.append((col, hashed))

        if raw_cols:
            out = pa.table(base)
            for col, a in raw_cols:
                out = out.append_column(col, a)
            pq.write_table(out, seq_dir / f"seq_{shard:05d}.parquet", compression="zstd")

        # 해시 시퀀스도 저장하려면 out_names.seq_hash 지정 필요
        if seq_hash_cols and "seq_hash" in out_names:
            seqh_dir = out_root / out_names["seq_hash"]
            ensure_dir(seqh_dir)
            out = pa.table(base)
            for col, a in seq_hash_cols:
                out = out.append_column(col, a)
            pq.write_table(out, seqh_dir / f"seq_hash_{shard:05d}.parquet", compression="zstd")

    print("Done:", args.output_dir)

if __name__ == "__main__":
    main()
