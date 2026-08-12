"""Build a catalog-clean Kepler target pool and a stratified injection-benchmark manifest.

"Catalog-clean" means no match in the catalog snapshots used here for KOIs,
DR25 TCEs, confirmed Kepler names, or the Kepler eclipsing-binary catalog. It
does not mean planet-free.
"""
import argparse
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/target_selection"
CACHE_ROOT = OUTPUT_ROOT / "catalog_cache"
POOL_PATH = OUTPUT_ROOT / "kepler_catalog_clean_pool.csv"
CANDIDATE_PATH = OUTPUT_ROOT / "kepler_catalog_clean_candidates_q5.csv"
FEATURE_PATH = OUTPUT_ROOT / "kepler_catalog_clean_candidate_features.csv"
FINAL_MANIFEST_PATH = PROJECT_ROOT / "configs/kepler_clean_background_manifest.csv"
CLEAN_SELECTION_GROUP = "catalog_clean_background"

TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
EB_URL = "https://archive.stsci.edu/kepler/eclipsing_binaries.html"


def tap_url(query):
    return f"{TAP_BASE}?query={quote_plus(query)}&format=csv"


CATALOG_SOURCES = {
    "kepler_stellar": tap_url("select distinct kepid from q1_q17_dr25_ks where kepid is not null"),
    "koi": tap_url("select distinct kepid from q1_q17_dr25_koi where kepid is not null"),
    "tce": tap_url("select distinct kepid from q1_q17_dr25_tce where kepid is not null"),
    "confirmed_kepler": tap_url("select distinct kepid from keplernames where kepid is not null"),
    "eclipsing_binary": EB_URL,
}


class HtmlCells(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_cell = False
        self.cell_parts = []
        self.row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"td", "th"}:
            self.in_cell = True
            self.cell_parts = []

    def handle_data(self, data):
        if self.in_cell:
            self.cell_parts.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"td", "th"} and self.in_cell:
            self.row.append(" ".join("".join(self.cell_parts).split()))
            self.in_cell = False
            self.cell_parts = []
        elif tag == "tr":
            if self.row:
                self.rows.append(self.row)
            self.row = []


def normalize_kepids(values):
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype("int64")
    return set(int(value) for value in numeric if int(value) > 0)


def fetch_bytes(url, cache_path, refresh=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "multimodel-transit-search target-selection/1.0"})
    with urlopen(request, timeout=120) as response:
        payload = response.read()
    cache_path.write_bytes(payload)
    return payload


def fetch_csv_ids(name, url, cache_root, refresh=False):
    path = Path(cache_root) / f"{name}.csv"
    payload = fetch_bytes(url, path, refresh=refresh)
    frame = pd.read_csv(path)
    if "kepid" not in frame.columns:
        raise ValueError(f"Catalog {name!r} does not contain a kepid column; columns={list(frame.columns)}")
    return frame, normalize_kepids(frame["kepid"])


def fetch_eb_ids(cache_root, refresh=False):
    path = Path(cache_root) / "kepler_eclipsing_binaries.html"
    payload = fetch_bytes(EB_URL, path, refresh=refresh)
    parser = HtmlCells()
    parser.feed(payload.decode("utf-8", errors="replace"))
    ids = set()
    for row in parser.rows:
        for cell in row[:2]:
            value = cell.replace(",", "").strip()
            if re.fullmatch(r"\d{6,9}", value):
                ids.add(int(value))
                break
    if len(ids) < 1000:
        raise ValueError(f"Parsed only {len(ids)} eclipsing-binary KepIDs; refusing to build a clean manifest from an incomplete EB parse.")
    return ids


def build_pool(args):
    stellar, stellar_ids = fetch_csv_ids("kepler_stellar", CATALOG_SOURCES["kepler_stellar"], args.catalog_cache_dir, args.refresh_catalogs)
    _, koi_ids = fetch_csv_ids("koi", CATALOG_SOURCES["koi"], args.catalog_cache_dir, args.refresh_catalogs)
    _, tce_ids = fetch_csv_ids("tce", CATALOG_SOURCES["tce"], args.catalog_cache_dir, args.refresh_catalogs)
    _, confirmed_ids = fetch_csv_ids("confirmed_kepler", CATALOG_SOURCES["confirmed_kepler"], args.catalog_cache_dir, args.refresh_catalogs)
    eb_ids = fetch_eb_ids(args.catalog_cache_dir, args.refresh_catalogs)

    kepids = sorted(stellar_ids)
    pool = pd.DataFrame({"target_id": [str(value) for value in kepids]})
    integer_ids = pool["target_id"].astype("int64")
    pool["quarter"] = int(args.quarter)
    pool["selection_group"] = CLEAN_SELECTION_GROUP
    pool["sample_stratum"] = "uncharacterized"
    pool["koi_flag"] = integer_ids.isin(koi_ids)
    pool["tce_flag"] = integer_ids.isin(tce_ids)
    pool["confirmed_planet_flag"] = integer_ids.isin(confirmed_ids)
    pool["eb_flag"] = integer_ids.isin(eb_ids)
    pool["catalog_clean"] = ~pool[["koi_flag", "tce_flag", "confirmed_planet_flag", "eb_flag"]].any(axis=1)
    pool["provenance"] = "NASA Exoplanet Archive DR25 stellar/KOI/TCE + Kepler confirmed names + MAST Kepler EB Revision 3"
    clean = pool.loc[pool["catalog_clean"]].copy().reset_index(drop=True)

    args.pool_output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.pool_output, index=False)
    n_candidates = min(int(args.candidate_limit), len(clean))
    candidates = clean.sample(n=n_candidates, random_state=int(args.seed), replace=False).sort_values("target_id").reset_index(drop=True)
    candidates.to_csv(args.candidate_output, index=False)
    source_record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "No catalog match in KOI, DR25 TCE, confirmed Kepler names, or Kepler EB Revision 3 snapshots used by this script; not a claim of being planet-free.",
        "sources": CATALOG_SOURCES,
        "quarter_requested_for_follow-up": int(args.quarter),
        "stellar_target_count": int(len(pool)),
        "catalog_clean_count": int(len(clean)),
        "candidate_count": int(len(candidates)),
        "seed": int(args.seed),
    }
    (args.pool_output.parent / "catalog_clean_selection_sources.json").write_text(json.dumps(source_record, indent=2) + "\n")
    print(f"Catalog-clean pool: {len(clean)} -> {args.pool_output}")
    print(f"Characterization candidates: {len(candidates)} -> {args.candidate_output}")
    return clean, candidates


def add_extreme(selected, selected_keys, frame, column, ascending, quota, stratum):
    if column not in frame.columns or quota <= 0:
        return
    ordered = frame.sort_values(column, ascending=ascending, na_position="last")
    added = 0
    for row in ordered.to_dict(orient="records"):
        key = (str(row["target_id"]), int(row["quarter"]))
        if key in selected_keys:
            continue
        if not np.isfinite(pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]):
            continue
        payload = dict(row)
        payload["sample_stratum"] = stratum
        selected.append(payload)
        selected_keys.add(key)
        added += 1
        if added >= quota:
            break


def build_final_manifest(args):
    if args.feature_path is None:
        return None
    if not Path(args.feature_path).exists():
        raise FileNotFoundError(f"Feature file does not exist: {args.feature_path}")
    features = pd.read_csv(args.feature_path, dtype={"target_id": str})
    required = {"target_id", "quarter", "status", "robust_flux_scatter_ppm", "gap_fraction"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Feature file is missing required columns: {sorted(missing)}")
    usable = features.loc[features["status"].astype(str).eq("success")].copy()
    for column in ("koi_flag", "tce_flag", "confirmed_planet_flag", "eb_flag"):
        if column not in usable.columns:
            raise ValueError(f"Feature file must preserve catalog flag {column!r}.")
        if usable[column].fillna(False).astype(bool).any():
            raise ValueError(f"Feature file contains catalog contamination in {column!r}.")
    usable["selection_group"] = CLEAN_SELECTION_GROUP
    final_size = min(int(args.final_size), len(usable))
    if final_size < int(args.final_size):
        raise ValueError(f"Only {len(usable)} characterized clean targets are available; requested {args.final_size}. Increase --candidate-limit and characterize more targets.")

    criteria = [
        ("robust_flux_scatter_ppm", True, "quiet_low_scatter"),
        ("robust_flux_scatter_ppm", False, "high_scatter"),
        ("background_tau_integrated_positive_acf_days", False, "long_memory"),
        ("rolling_background_to_short_scatter_ratio", False, "smooth_background_dominant"),
        ("gap_fraction", False, "gap_heavy"),
    ]
    base_quota, remainder = divmod(final_size, len(criteria))
    selected, selected_keys = [], set()
    for index, (column, ascending, label) in enumerate(criteria):
        quota = base_quota + int(index < remainder)
        add_extreme(selected, selected_keys, usable, column, ascending, quota, label)
    if len(selected) < final_size:
        for row in usable.sort_values(["target_id", "quarter"]).to_dict(orient="records"):
            key = (str(row["target_id"]), int(row["quarter"]))
            if key in selected_keys:
                continue
            payload = dict(row)
            payload["sample_stratum"] = "mixed_fill"
            selected.append(payload)
            selected_keys.add(key)
            if len(selected) >= final_size:
                break
    manifest = pd.DataFrame(selected[:final_size])
    keep = ["target_id", "quarter", "selection_group", "sample_stratum", "koi_flag", "tce_flag", "confirmed_planet_flag", "eb_flag", "provenance"]
    for column in keep:
        if column not in manifest.columns:
            manifest[column] = "" if column in {"sample_stratum", "provenance"} else False
    manifest = manifest[keep].copy()
    manifest["target_id"] = manifest["target_id"].astype(str)
    manifest = manifest.sort_values(["sample_stratum", "target_id"]).reset_index(drop=True)
    args.final_output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.final_output, index=False)
    print(f"Final clean benchmark manifest: {len(manifest)} -> {args.final_output}")
    print(manifest["sample_stratum"].value_counts().sort_index().to_string())
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build a reproducible catalog-clean Kepler background cohort for injection recovery.")
    parser.add_argument("--quarter", type=int, default=5)
    parser.add_argument("--catalog-cache-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--refresh-catalogs", action="store_true")
    parser.add_argument("--pool-output", type=Path, default=POOL_PATH)
    parser.add_argument("--candidate-output", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--candidate-limit", type=int, default=250)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--feature-path", type=Path)
    parser.add_argument("--final-output", type=Path, default=FINAL_MANIFEST_PATH)
    parser.add_argument("--final-size", type=int, default=50)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    build_pool(args)
    build_final_manifest(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
