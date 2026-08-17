import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bridge", ROOT / "scripts/build_canonical_pipeline_bridge.py")
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def test_source_columns_are_resolved_to_all_frozen_canonical_names():
    frame = pd.DataFrame({
        "target_id": ["KIC 1"],
        "flux_robust_scale": [1.0],
        "flux_skewness": [0.1],
        "flux_outlier_fraction": [0.02],
        "v2_acf_lag_1": [0.3],
        "v2_acf_decay_e_days": [2.0],
        "original_series_stationarity_conclusion": ["stationary_supported"],
        "v2_spectral_concentration": [0.4],
        "v2_spectral_harmonic_power_ratio": [0.5],
        "v2_ls_dominant_period_days": [7.0],
        "v2_ls_acf_period_relative_error": [0.08],
        "v2_segment_scale_relative_mad": [0.12],
    })
    out = bridge.canonicalize_feature_frame(frame)
    assert set(bridge.CANONICAL_FEATURES).issubset(out.columns)
    assert out.loc[0, "target_id"] == "1"
    assert out.loc[0, "segment_scale_variability"] == 0.12


def test_spearman_output_is_descriptive_and_branch_specific():
    rows = []
    for i in range(5):
        row = {name: float(i + 1) for name in bridge.CONTINUOUS_FEATURES}
        row.update({
            "stationarity_state": "stationary_supported",
            "target_id": str(i),
            "branch": "gp",
            "median_template_correlation": float(i + 1),
            "oracle_snr_gain_vs_raw": float(i + 1),
            "whitening_gain_vs_raw": float(i + 1),
            "tls_exact_gain_vs_raw": float(i + 1),
        })
        rows.append(row)
    out = bridge.spearman_associations(pd.DataFrame(rows))
    hit = out[(out["feature"] == "robust_scatter") & (out["outcome"] == "oracle_snr_gain_vs_raw")].iloc[0]
    assert np.isclose(hit["spearman_rho"], 1.0)
    assert hit["n_stars"] == 5
    assert "descriptive" in hit["interpretation_scope"]
