"""Small long-table resume/cache helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PRIMARY_KEYS = {
    "characterization": ("run_id", "config_hash", "star_id"),
    "injection": ("run_id", "config_hash", "star_id", "injection_id"),
    "preservation": ("run_id", "config_hash", "star_id", "injection_id", "treatment"),
    "detection": ("run_id", "config_hash", "star_id", "injection_id", "treatment", "detector", "score_definition"),
    "null_score": ("run_id", "config_hash", "star_id", "trial", "treatment", "detector", "score_definition"),
}


class LongTableStore:
    """CSV-backed append-only store that deduplicates by config-aware keys."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, table_name: str) -> Path:
        return self.root / f"{table_name}.csv"

    def read(self, table_name: str) -> pd.DataFrame:
        path = self.path(table_name)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def completed_keys(self, table_name: str, *, config_hash: str) -> set[tuple]:
        frame = self.read(table_name)
        keys = PRIMARY_KEYS[table_name]
        if frame.empty or any(key not in frame.columns for key in keys):
            return set()
        current = frame[frame["config_hash"].astype(str).eq(str(config_hash))]
        return {tuple(row[key] for key in keys) for _, row in current.iterrows()}

    def append_rows(self, table_name: str, rows: Iterable[dict]) -> pd.DataFrame:
        incoming = pd.DataFrame(list(rows))
        if incoming.empty:
            return self.read(table_name)
        existing = self.read(table_name)
        combined = incoming if existing.empty else pd.concat([existing, incoming], ignore_index=True)
        keys = [key for key in PRIMARY_KEYS[table_name] if key in combined.columns]
        if keys:
            combined = combined.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
        path = self.path(table_name)
        combined.to_csv(path, index=False)
        return combined
