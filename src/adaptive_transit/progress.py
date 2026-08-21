"""Incremental SQLite progress store and event log for benchmark runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
from typing import Iterable, Mapping

import pandas as pd

from adaptive_transit.resume import PRIMARY_KEYS
from adaptive_transit.schemas import LONG_TABLE_SCHEMAS, json_ready


FAP_THRESHOLD_COLUMNS = (
    "run_id",
    "config_hash",
    "star_id",
    "treatment",
    "detector",
    "score_name",
    "score_definition",
    "fap_level",
    "fap_threshold",
    "null_trial_count",
)

PROGRESS_PRIMARY_KEYS = {
    **PRIMARY_KEYS,
    "fap_thresholds": (
        "run_id",
        "config_hash",
        "star_id",
        "treatment",
        "detector",
        "score_definition",
        "fap_level",
    ),
    "run_status": ("run_id", "config_hash", "star_id", "stage", "injection_id", "treatment", "detector", "null_trial"),
}

TABLE_COLUMNS = {
    **{name: tuple(columns) for name, columns in LONG_TABLE_SCHEMAS.items()},
    "fap_thresholds": FAP_THRESHOLD_COLUMNS,
    "run_status": (
        "run_id",
        "config_hash",
        "star_id",
        "stage",
        "injection_id",
        "treatment",
        "detector",
        "null_trial",
        "status",
        "runtime_seconds",
        "updated_at",
        "error",
    ),
}


class LiveBenchmarkStore:
    """Small SQLite/WAL store with stable keys for live progress and resume."""

    def __init__(self, root: Path, *, db_name: str = "run_live.sqlite", event_name: str = "run_events.jsonl") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / db_name
        self.event_path = self.root / event_name
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def close(self) -> None:
        self.connection.close()

    def _create_tables(self) -> None:
        for table_name, columns in TABLE_COLUMNS.items():
            keys = PROGRESS_PRIMARY_KEYS[table_name]
            extras = ("error", "diagnostics") if table_name not in {"run_status"} else ()
            all_columns = tuple(dict.fromkeys((*columns, *extras)))
            column_sql = ", ".join(f'"{column}"' for column in all_columns)
            key_sql = ", ".join(f'"{key}"' for key in keys)
            self.connection.execute(
                f'CREATE TABLE IF NOT EXISTS "{table_name}" ({column_sql}, PRIMARY KEY ({key_sql}))'
            )
            existing = set(self._columns(table_name))
            for column in all_columns:
                if column not in existing:
                    self.connection.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{column}"')
        self.connection.commit()

    def _columns(self, table_name: str) -> tuple[str, ...]:
        rows = self.connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        return tuple(row[1] for row in rows)

    def upsert_rows(self, table_name: str, rows: Iterable[Mapping]) -> None:
        materialized = [dict(row) for row in rows]
        if not materialized:
            return
        columns = self._columns(table_name)
        placeholders = ", ".join("?" for _ in columns)
        column_sql = ", ".join(f'"{column}"' for column in columns)
        values = [tuple(self._sqlite_value(row.get(column)) for column in columns) for row in materialized]
        self.connection.executemany(
            f'INSERT OR REPLACE INTO "{table_name}" ({column_sql}) VALUES ({placeholders})',
            values,
        )
        self.connection.commit()

    def upsert_row(self, table_name: str, row: Mapping) -> None:
        self.upsert_rows(table_name, [row])

    def read_table(self, table_name: str) -> pd.DataFrame:
        return pd.read_sql_query(f'SELECT * FROM "{table_name}"', self.connection)

    def read(self, table_name: str) -> pd.DataFrame:
        return self.read_table(table_name)

    def completed_keys(self, table_name: str, *, run_id: str | None = None, config_hash: str | None = None) -> set[tuple]:
        keys = PROGRESS_PRIMARY_KEYS[table_name]
        clauses = []
        params = []
        if run_id is not None:
            clauses.append('"run_id" = ?')
            params.append(str(run_id))
        if config_hash is not None:
            clauses.append('"config_hash" = ?')
            params.append(str(config_hash))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        key_sql = ", ".join(f'"{key}"' for key in keys)
        rows = self.connection.execute(f'SELECT {key_sql} FROM "{table_name}"{where}', params).fetchall()
        return {tuple(row) for row in rows}

    def has_key(self, table_name: str, key_values: Mapping) -> bool:
        keys = PROGRESS_PRIMARY_KEYS[table_name]
        clauses = [f'"{key}" = ?' for key in keys]
        params = [self._sqlite_value(key_values.get(key)) for key in keys]
        row = self.connection.execute(
            f'SELECT 1 FROM "{table_name}" WHERE {" AND ".join(clauses)} LIMIT 1',
            params,
        ).fetchone()
        return row is not None

    def record_status(self, **fields) -> None:
        row = {
            "run_id": fields.get("run_id"),
            "config_hash": fields.get("config_hash"),
            "star_id": fields.get("star_id", ""),
            "stage": fields.get("stage", ""),
            "injection_id": fields.get("injection_id", ""),
            "treatment": fields.get("treatment", ""),
            "detector": fields.get("detector", ""),
            "null_trial": fields.get("null_trial", ""),
            "status": fields.get("status", ""),
            "runtime_seconds": fields.get("runtime_seconds"),
            "updated_at": utc_now(),
            "error": fields.get("error", ""),
        }
        self.upsert_row("run_status", row)
        self.append_event(row)

    def append_event(self, fields: Mapping) -> None:
        payload = {"timestamp": utc_now(), **dict(fields)}
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_ready(payload), sort_keys=True) + "\n")

    def export_csvs(
        self,
        table_names: Iterable[str] | None = None,
        *,
        run_id: str | None = None,
        config_hash: str | None = None,
        preserve_incompatible: bool = True,
    ) -> dict[str, Path]:
        names = tuple(
            table_names
            or (
                "characterization",
                "treatment",
                "injection",
                "preservation",
                "detection",
                "null_score",
                "fap_thresholds",
                "run_status",
            )
        )
        written = {}
        for table_name in names:
            frame = self.read_table(table_name)
            path = self.root / f"{table_name}.csv"
            if preserve_incompatible and path.exists() and (run_id is not None or config_hash is not None):
                historical = self._incompatible_existing_rows(path, run_id=run_id, config_hash=config_hash)
                if not historical.empty:
                    frame = pd.concat([historical, frame], ignore_index=True, sort=False)
                    keys = [key for key in PROGRESS_PRIMARY_KEYS[table_name] if key in frame.columns]
                    if keys:
                        frame = frame.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
            if frame.empty:
                continue
            frame.to_csv(path, index=False)
            written[table_name] = path
        return written

    def import_existing_csvs(
        self,
        *,
        run_id: str | None = None,
        config_hash: str | None = None,
        compatible_only: bool = True,
    ) -> dict[str, int]:
        imported = {}
        for table_name in (
            "characterization",
            "treatment",
            "injection",
            "preservation",
            "detection",
            "null_score",
            "fap_thresholds",
            "run_status",
        ):
            path = self.root / f"{table_name}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if compatible_only:
                frame = self._compatible_existing_rows(frame, run_id=run_id, config_hash=config_hash)
            if not frame.empty:
                self.upsert_rows(table_name, frame.to_dict(orient="records"))
                imported[table_name] = int(len(frame))
        return imported

    def _compatible_existing_rows(
        self,
        frame: pd.DataFrame,
        *,
        run_id: str | None,
        config_hash: str | None,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        current = frame.copy()
        if config_hash is not None:
            if "config_hash" not in current.columns:
                return current.iloc[0:0].copy()
            current = current[current["config_hash"].astype(str).eq(str(config_hash))]
        if run_id is not None:
            if "run_id" not in current.columns:
                return current.iloc[0:0].copy()
            current = current[current["run_id"].astype(str).eq(str(run_id))]
        return current.copy()

    def _incompatible_existing_rows(
        self,
        path: Path,
        *,
        run_id: str | None,
        config_hash: str | None,
    ) -> pd.DataFrame:
        try:
            frame = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        if frame.empty:
            return frame
        mask = pd.Series(False, index=frame.index)
        if config_hash is not None and "config_hash" in frame.columns:
            mask |= ~frame["config_hash"].astype(str).eq(str(config_hash))
        if run_id is not None and "run_id" in frame.columns:
            mask |= ~frame["run_id"].astype(str).eq(str(run_id))
        if not mask.any():
            return frame.iloc[0:0].copy()
        return frame[mask].copy()

    def __enter__(self) -> "LiveBenchmarkStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @staticmethod
    def _sqlite_value(value):
        ready = json_ready(value)
        if isinstance(ready, (dict, list, tuple)):
            return json.dumps(ready, sort_keys=True)
        if isinstance(ready, bool):
            return int(ready)
        return ready


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
