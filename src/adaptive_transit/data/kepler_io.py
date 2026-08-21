"""Kepler light-curve loading utilities.

This module loads PDCSAP flux for one target and one quarter at a time. It is
not a full-archive ingestion system.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any
import lightkurve as lk
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIGHT_CURVE_CACHE_DIR = PROJECT_ROOT / "outputs/cache/kepler_light_curves"
DEFAULT_MAST_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAST_READ_TIMEOUT_SECONDS = 120.0
DEFAULT_MAST_MAX_ATTEMPTS = 3
DEFAULT_MAST_INITIAL_WAIT_SECONDS = 10.0
DEFAULT_MAST_BACKOFF_FACTOR = 2.0
LIGHT_CURVE_FRAME_COLUMNS = ("time", "flux", "flux_error", "quality", "cadenceno")


@dataclass(frozen=True)
class KeplerFetchPolicy:
    """Operational policy for bounded MAST reads."""

    connect_timeout_seconds: float = DEFAULT_MAST_CONNECT_TIMEOUT_SECONDS
    read_timeout_seconds: float = DEFAULT_MAST_READ_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAST_MAX_ATTEMPTS
    initial_wait_seconds: float = DEFAULT_MAST_INITIAL_WAIT_SECONDS
    backoff_factor: float = DEFAULT_MAST_BACKOFF_FACTOR

    @property
    def timeout(self) -> tuple[float, float]:
        return (float(self.connect_timeout_seconds), float(self.read_timeout_seconds))

    def validate(self) -> None:
        if float(self.connect_timeout_seconds) <= 0:
            raise ValueError("connect_timeout_seconds must be positive.")
        if float(self.read_timeout_seconds) <= 0:
            raise ValueError("read_timeout_seconds must be positive.")
        if int(self.max_attempts) < 1:
            raise ValueError("max_attempts must be at least 1.")
        if float(self.initial_wait_seconds) < 0:
            raise ValueError("initial_wait_seconds must be non-negative.")
        if float(self.backoff_factor) < 1:
            raise ValueError("backoff_factor must be at least 1.")

    def retry_delay(self, completed_attempt: int) -> float:
        return float(self.initial_wait_seconds) * float(self.backoff_factor) ** max(0, int(completed_attempt) - 1)


class KeplerLightCurveFetchError(RuntimeError):
    """Raised when a transient MAST fetch fails after the retry budget."""


@dataclass(frozen=True)
class KeplerLightCurve:
    """In-memory representation of one Kepler PDCSAP light curve."""

    target_id: str
    quarter: int
    time: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray
    quality: np.ndarray
    cadenceno: np.ndarray
    metadata: dict[str, Any]

    def to_dataframe(self):
        return pd.DataFrame(
            {
                "time": self.time,
                "flux": self.flux,
                "flux_error": self.flux_error,
                "quality": self.quality,
                "cadenceno": self.cadenceno,
            }
        )


def format_kepler_target(target_id):
    """Normalize a target identifier into the format expected by Lightkurve."""
    target = str(target_id).strip()
    if target.upper().startswith("KIC"):
        target = target[3:].strip()
    return f"KIC {target}"


def normalize_kepler_target_id(target_id) -> str:
    """Normalize a target identifier for local cache filenames."""

    return str(target_id).upper().replace("KIC", "").strip()


def light_curve_cache_path(cache_dir: Path | str, target_id, quarter) -> Path:
    """Return the project-standard cache path for one Kepler target-quarter."""

    return Path(cache_dir) / f"kic_{normalize_kepler_target_id(target_id)}_q{int(quarter)}_pdcsap.parquet"


def is_transient_mast_error(exc: Exception) -> bool:
    """Return whether a fetch exception is worth retrying."""

    if isinstance(exc, TimeoutError):
        return True
    try:
        import requests

        if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True
    except Exception:
        pass
    message = f"{type(exc).__name__}: {exc}"
    if "No Kepler light curve found" in message:
        return False
    markers = (
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectionError",
        "HTTPSConnectionPool",
        "Max retries exceeded",
        "Temporary failure",
        "temporarily unavailable",
        "Connection reset",
        "RemoteDisconnected",
        "mast.stsci.edu",
        "Timeout limit",
    )
    return any(marker in message for marker in markers)


@contextmanager
def bounded_mast_requests(policy: KeplerFetchPolicy):
    """Apply finite request timeouts to Astroquery calls made by Lightkurve."""

    policy.validate()
    from astroquery.query import BaseQuery

    try:
        from astroquery.mast.discovery_portal import PortalAPI
    except Exception:
        PortalAPI = None

    original_request = BaseQuery._request
    original_download_file = BaseQuery._download_file
    original_portal_timeout = getattr(PortalAPI, "TIMEOUT", None) if PortalAPI is not None else None

    def bounded_request(self, *args, **kwargs):
        if len(args) >= 9:
            mutable_args = list(args)
            if mutable_args[8] is None:
                mutable_args[8] = policy.timeout
            args = tuple(mutable_args)
        elif kwargs.get("timeout") is None:
            kwargs["timeout"] = policy.timeout
        return original_request(self, *args, **kwargs)

    def bounded_download_file(self, url, local_filepath, timeout=None, *args, **kwargs):
        if timeout is None:
            timeout = policy.timeout
        return original_download_file(self, url, local_filepath, timeout=timeout, *args, **kwargs)

    BaseQuery._request = bounded_request
    BaseQuery._download_file = bounded_download_file
    if PortalAPI is not None:
        PortalAPI.TIMEOUT = float(max(policy.connect_timeout_seconds, policy.read_timeout_seconds))
    try:
        yield
    finally:
        BaseQuery._request = original_request
        BaseQuery._download_file = original_download_file
        if PortalAPI is not None:
            PortalAPI.TIMEOUT = original_portal_timeout


def _emit_fetch_event(progress_callback, **payload) -> None:
    if progress_callback is None:
        return
    progress_callback(dict(payload))


def _load_kepler_pdcsap_once(target_id, quarter, *, author="Kepler", cadence="long", fetch_policy=None):
    """Download one Kepler PDCSAP light curve with Lightkurve.

    No quality masking, gap handling, cleaning, or normalization is performed here.
    Those operations are deferred to `preprocess_pdcsap_light_curve` so the preprocessing choices remain explicit and testable.
    """
    query = format_kepler_target(target_id)
    policy = fetch_policy or KeplerFetchPolicy()
    with bounded_mast_requests(policy):
        search = lk.search_lightcurve(query, mission="Kepler", quarter=quarter, author=author, cadence=cadence)
    if len(search) == 0:
        raise RuntimeError(f"No Kepler light curve found for target={query!r}, quarter={quarter}, author={author!r}, cadence={cadence!r}.")
    with bounded_mast_requests(policy):
        light_curve = search[0].download(quality_bitmask="none")
    if light_curve is None:
        raise RuntimeError(f"Lightkurve did not download a light curve for {query}.")
    # Cast FITS integer columns to native int64 for downstream NumPy/pandas operations.
    return KeplerLightCurve(
        target_id=query,
        quarter=quarter,
        time=np.asarray(light_curve.time.value, dtype=float),
        flux=np.asarray(light_curve.pdcsap_flux.value, dtype=float),
        flux_error=np.asarray(light_curve.pdcsap_flux_err.value, dtype=float),
        quality=np.asarray(light_curve.quality.value, dtype=np.int64),
        cadenceno=np.asarray(light_curve.cadenceno.value, dtype=np.int64),
        metadata={"author": author, "cadence": cadence, "mission": "Kepler", "search_result": str(search[0])},
    )


def load_kepler_pdcsap(
    target_id,
    quarter,
    *,
    author="Kepler",
    cadence="long",
    fetch_policy: KeplerFetchPolicy | None = None,
    progress_callback=None,
    sleep_fn=sleep,
):
    """Download one Kepler PDCSAP light curve with bounded MAST retries."""

    policy = fetch_policy or KeplerFetchPolicy()
    policy.validate()
    failures: list[str] = []
    for attempt in range(1, int(policy.max_attempts) + 1):
        _emit_fetch_event(
            progress_callback,
            status="attempt",
            target_id=normalize_kepler_target_id(target_id),
            quarter=int(quarter),
            attempt=attempt,
            max_attempts=int(policy.max_attempts),
            timeout_seconds=policy.timeout,
        )
        try:
            result = _load_kepler_pdcsap_once(
                target_id,
                quarter,
                author=author,
                cadence=cadence,
                fetch_policy=policy,
            )
            _emit_fetch_event(
                progress_callback,
                status="complete",
                target_id=normalize_kepler_target_id(target_id),
                quarter=int(quarter),
                attempt=attempt,
                max_attempts=int(policy.max_attempts),
            )
            return result
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            failures.append(failure)
            if attempt >= int(policy.max_attempts) or not is_transient_mast_error(exc):
                _emit_fetch_event(
                    progress_callback,
                    status="failed",
                    target_id=normalize_kepler_target_id(target_id),
                    quarter=int(quarter),
                    attempt=attempt,
                    max_attempts=int(policy.max_attempts),
                    error=failure,
                )
                if is_transient_mast_error(exc) and attempt >= int(policy.max_attempts):
                    raise KeplerLightCurveFetchError(
                        "Kepler MAST fetch failed after "
                        f"{attempt}/{int(policy.max_attempts)} attempts for "
                        f"{format_kepler_target(target_id)} Q{int(quarter)}: {failure}"
                    ) from exc
                raise
            wait_seconds = policy.retry_delay(attempt)
            _emit_fetch_event(
                progress_callback,
                status="retrying",
                target_id=normalize_kepler_target_id(target_id),
                quarter=int(quarter),
                attempt=attempt,
                max_attempts=int(policy.max_attempts),
                wait_seconds=wait_seconds,
                error=failure,
            )
            sleep_fn(wait_seconds)
    raise KeplerLightCurveFetchError(
        "Kepler MAST fetch retry loop ended unexpectedly for "
        f"{format_kepler_target(target_id)} Q{int(quarter)}. Failures: {failures}"
    )


def _validate_light_curve_frame(frame: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    missing = sorted(set(LIGHT_CURVE_FRAME_COLUMNS).difference(frame.columns))
    if missing:
        location = "" if path is None else f" at {path}"
        raise ValueError(f"Cached light-curve frame{location} is missing columns: {missing}")
    return frame.loc[:, LIGHT_CURVE_FRAME_COLUMNS].copy()


def load_cached_kepler_pdcsap_frame(
    target_id,
    quarter,
    *,
    cache_dir: Path | str = DEFAULT_LIGHT_CURVE_CACHE_DIR,
    allow_download: bool = True,
    author="Kepler",
    cadence="long",
    fetch_policy: KeplerFetchPolicy | None = None,
    progress_callback=None,
    sleep_fn=sleep,
) -> tuple[pd.DataFrame, bool]:
    """Load one PDCSAP frame from local cache, downloading and caching on miss."""

    path = light_curve_cache_path(cache_dir, target_id, quarter)
    if path.exists():
        _emit_fetch_event(
            progress_callback,
            status="cache_hit",
            target_id=normalize_kepler_target_id(target_id),
            quarter=int(quarter),
            path=str(path),
        )
        return _validate_light_curve_frame(pd.read_parquet(path), path), True
    _emit_fetch_event(
        progress_callback,
        status="cache_miss",
        target_id=normalize_kepler_target_id(target_id),
        quarter=int(quarter),
        path=str(path),
    )
    if not allow_download:
        raise FileNotFoundError(f"Cached light curve is missing: {path}. Re-run without --no-download to fetch it.")
    light_curve = load_kepler_pdcsap(
        target_id,
        quarter,
        author=author,
        cadence=cadence,
        fetch_policy=fetch_policy,
        progress_callback=progress_callback,
        sleep_fn=sleep_fn,
    )
    frame = _validate_light_curve_frame(light_curve.to_dataframe())
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    _emit_fetch_event(
        progress_callback,
        status="cache_written",
        target_id=normalize_kepler_target_id(target_id),
        quarter=int(quarter),
        path=str(path),
    )
    return frame, False
