import numpy as np
import pandas as pd
import pytest
import requests

from adaptive_transit.data import kepler_io
from adaptive_transit.data.kepler_io import (
    KeplerFetchPolicy,
    KeplerLightCurveFetchError,
    bounded_mast_requests,
    light_curve_cache_path,
    load_cached_kepler_pdcsap_frame,
    load_kepler_pdcsap,
)


class _Value:
    def __init__(self, values):
        self.value = np.asarray(values)


class _DownloadedLightCurve:
    time = _Value([1.0, 2.0, 3.0])
    pdcsap_flux = _Value([100.0, 101.0, 99.0])
    pdcsap_flux_err = _Value([0.1, 0.1, 0.1])
    quality = _Value([0, 0, 0])
    cadenceno = _Value([10, 11, 12])


class _SearchResult:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self

    def __str__(self):
        return "unit-search-result"

    def download(self, quality_bitmask):
        assert quality_bitmask == "none"
        return _DownloadedLightCurve()


def test_bounded_mast_requests_injects_connect_read_timeout(monkeypatch):
    from astroquery.query import BaseQuery

    captured = []

    def fake_request(self, *args, **kwargs):
        captured.append(kwargs.get("timeout"))
        return object()

    monkeypatch.setattr(BaseQuery, "_request", fake_request)
    policy = KeplerFetchPolicy(connect_timeout_seconds=1.5, read_timeout_seconds=2.5)
    with bounded_mast_requests(policy):
        BaseQuery()._request("GET", "https://example.invalid")

    assert captured == [(1.5, 2.5)]
    assert BaseQuery._request is fake_request


def test_bounded_mast_requests_replaces_positional_none_timeout(monkeypatch):
    from astroquery.query import BaseQuery

    captured = []

    def fake_request(self, *args, **kwargs):
        captured.append(args[8])
        return object()

    monkeypatch.setattr(BaseQuery, "_request", fake_request)
    policy = KeplerFetchPolicy(connect_timeout_seconds=1.5, read_timeout_seconds=2.5)
    positional_args = (
        "GET",
        "https://example.invalid",
        None,
        None,
        None,
        None,
        False,
        "",
        None,
    )
    with bounded_mast_requests(policy):
        BaseQuery()._request(*positional_args)

    assert captured == [(1.5, 2.5)]


def test_bounded_mast_requests_injects_download_file_timeout(monkeypatch):
    from astroquery.query import BaseQuery

    captured = []

    def fake_download_file(self, url, local_filepath, timeout=None, **kwargs):
        captured.append(timeout)
        return local_filepath

    monkeypatch.setattr(BaseQuery, "_download_file", fake_download_file)
    policy = KeplerFetchPolicy(connect_timeout_seconds=1.5, read_timeout_seconds=2.5)
    with bounded_mast_requests(policy):
        BaseQuery()._download_file("https://example.invalid/file.fits", "file.fits")

    assert captured == [(1.5, 2.5)]
    assert BaseQuery._download_file is fake_download_file


def test_load_kepler_pdcsap_retries_transient_mast_errors(monkeypatch):
    attempts = []
    waits = []
    events = []

    def fake_search_lightcurve(*args, **kwargs):
        attempts.append((args, kwargs))
        if len(attempts) < 3:
            raise requests.exceptions.ReadTimeout("slow MAST read")
        return _SearchResult()

    monkeypatch.setattr(kepler_io.lk, "search_lightcurve", fake_search_lightcurve)
    policy = KeplerFetchPolicy(
        connect_timeout_seconds=1.0,
        read_timeout_seconds=2.0,
        max_attempts=3,
        initial_wait_seconds=0.0,
    )
    light_curve = load_kepler_pdcsap(
        "1",
        5,
        fetch_policy=policy,
        progress_callback=events.append,
        sleep_fn=waits.append,
    )

    assert light_curve.target_id == "KIC 1"
    assert len(attempts) == 3
    assert waits == [0.0, 0.0]
    assert [event["status"] for event in events] == [
        "attempt",
        "retrying",
        "attempt",
        "retrying",
        "attempt",
        "complete",
    ]


def test_load_kepler_pdcsap_records_retry_limit(monkeypatch):
    def fake_search_lightcurve(*_args, **_kwargs):
        raise requests.exceptions.ConnectTimeout("cannot connect")

    monkeypatch.setattr(kepler_io.lk, "search_lightcurve", fake_search_lightcurve)
    policy = KeplerFetchPolicy(
        connect_timeout_seconds=1.0,
        read_timeout_seconds=2.0,
        max_attempts=3,
        initial_wait_seconds=0.0,
    )
    events = []

    with pytest.raises(KeplerLightCurveFetchError, match="failed after 3/3 attempts"):
        load_kepler_pdcsap("1", 5, fetch_policy=policy, progress_callback=events.append, sleep_fn=lambda _seconds: None)

    assert [event["status"] for event in events][-2:] == ["attempt", "failed"]


def test_cached_frame_loads_without_mast_when_no_download(monkeypatch, tmp_path):
    path = light_curve_cache_path(tmp_path, "1", 5)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "flux": [100.0, 101.0],
            "flux_error": [0.1, 0.1],
            "quality": [0, 0],
            "cadenceno": [10, 11],
        }
    )
    expected.to_parquet(path, index=False)

    def fail_search(*_args, **_kwargs):
        raise AssertionError("cache hit should not call MAST")

    monkeypatch.setattr(kepler_io.lk, "search_lightcurve", fail_search)
    frame, cache_hit = load_cached_kepler_pdcsap_frame("KIC 1", 5, cache_dir=tmp_path, allow_download=False)

    assert cache_hit is True
    pd.testing.assert_frame_equal(frame, expected)
