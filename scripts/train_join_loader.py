#!/usr/bin/env python3
"""
Join modality shards by RID at training time.
- ShardedJoiner.load_modalities(shard_idx, use=[...]) -> pd.DataFrame
- CTRShardedDataset: minimal PyTorch dataset over joined shards
"""

from pathlib import Path
import json, glob
from typing import List, Optional
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def _glob_sorted(pat: str) -> List[str]:
    files = sorted(glob.glob(pat))
    return files

def _read_df(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas(types_mapper=pd.ArrowDtype)

class ShardedJoiner:
    def __init__(self, out_root: str):
        self.root = Path(out_root)
        self.meta  = json.loads((self.root / "meta" / "manifest.json").read_text())
        self.rid   = self.meta["rid_col"]
        self.label = self.meta.get("label_col")

        self.cont     = _glob_sorted(str(self.root / "cont"     / "cont_*.parquet"))
        self.cat_hash = _glob_sorted(str(self.root / "cat_hash" / "cat_hash_*.parquet"))
        self.cat_vocab= _glob_sorted(str(self.root / "cat_vocab"/ "cat_vocab_*.parquet"))
        self.seq      = _glob_sorted(str(self.root / "seq"      / "seq_*.parquet"))

        self.groups = { "cont": self.cont, "cat_hash": self.cat_hash, "cat_vocab": self.cat_vocab, "seq": self.seq }
        self.max_shards = max((len(v) for v in self.groups.values() if v), default=0)

    def num_shards(self) -> int:
        return self.max_shards

    def load_modalities(self, shard_idx: int, use: List[str]) -> pd.DataFrame:
        dfs = []
        for g in use:
            files = self.groups.get(g, [])
            if not files: continue
            dfs.append(_read_df(files[shard_idx]))
        if not dfs: raise ValueError("no modalities found")
        out = dfs[0]
        for d in dfs[1:]:
            out = out.merge(d, on=self.rid, how="inner")
        return out

class CTRShardedDataset(Dataset):
    def __init__(self, out_root: str, use: List[str], batch_cache: bool = False):
        self.joiner = ShardedJoiner(out_root)
        self.use = use
        self.rid = self.joiner.rid
        self.label = self.joiner.label
        self.shards = self.joiner.num_shards()
        # discover feature columns
        df0 = self.joiner.load_modalities(0, use)
        drop = [self.rid] + ([self.label] if self.label and self.label in df0.columns else [])
        self.feat_cols = [c for c in df0.columns if c not in drop]
        # sizes
        self.sizes = [len(self.joiner.load_modalities(i, use)) for i in range(self.shards)]
        self.cum = np.cumsum([0] + self.sizes)
        self.cache = {} if batch_cache else None

    def __len__(self): return int(self.cum[-1])

    def _loc(self, idx: int):
        shard = int(np.searchsorted(self.cum, idx, side="right") - 1)
        local = idx - self.cum[shard]
        return shard, local

    def __getitem__(self, idx: int):
        shard, local = self._loc(idx)
        if self.cache is not None and shard in self.cache:
            df = self.cache[shard]
        else:
            df = self.joiner.load_modalities(shard, self.use)
            if self.cache is not None: self.cache[shard] = df
        row = df.iloc[local]
        x = row[self.feat_cols]
        x = np.array([np.nan if pd.isna(v) else v for v in x], dtype=np.float32)
        if self.label and self.label in df.columns:
            y = np.float32(row[self.label])
            return x, y
        return x
