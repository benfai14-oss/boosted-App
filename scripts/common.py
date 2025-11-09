"""Common helper functions for I/O and auditing.

Throughout the pipeline we need to read and write tabular data to disk
in a flexible way.  This module centralises those operations and
provides a fallback when optional dependencies are missing.  It also
implements a simple audit logging mechanism that writes JSON lines
into the ``logs/`` directory, enabling post‑hoc inspection of the
pipeline's behaviour.
"""

from __future__ import annotations

import json
import os
import pathlib
import datetime as _dt
from typing import Any, Dict

import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
    import pyarrow as pa  # type: ignore
    _PARQUET_AVAILABLE = True
except Exception:
    _PARQUET_AVAILABLE = False


def ensure_dir(path: str) -> None:
    """Ensure that the directory exists."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to disk, using Parquet if available.

    The directory will be created if it does not exist.  If PyArrow is
    installed and the file extension is ``.parquet`` or ``.parq``, the
    function writes a Parquet file.  Otherwise it writes a CSV.
    """
    ensure_dir(os.path.dirname(path))
    suffix = pathlib.Path(path).suffix.lower()
    if _PARQUET_AVAILABLE and suffix in {".parquet", ".parq"}:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)
    else:
        df.to_csv(path, index=False)


def read_table(path: str) -> pd.DataFrame:
    """Read a DataFrame from a Parquet or CSV file."""
    suffix = pathlib.Path(path).suffix.lower()
    if _PARQUET_AVAILABLE and suffix in {".parquet", ".parq"}:
        table = pq.read_table(path)
        return table.to_pandas()
    else:
        return pd.read_csv(path, parse_dates=["date"])  # assume 'date' exists


def log_event(layer: str, action: str, payload: Dict[str, Any]) -> None:
    """Append an event to the daily audit log.

    Logs are stored under ``logs/audit_YYYY‑MM‑DD.ndjson``.  Each line
    contains a JSON object with ``timestamp``, ``layer``, ``action`` and
    ``payload`` keys.  The timestamp is in UTC ISO8601 format.
    """
    ensure_dir("logs")
    ts = _dt.datetime.utcnow().isoformat() + "Z"
    event = {
        "timestamp": ts,
        "layer": layer,
        "action": action,
        "payload": payload,
    }
    fname = f"logs/audit_{_dt.date.today().isoformat()}.ndjson"
    with open(fname, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

# --- Compatibility shim ---
def write_audit_log(*args, **kwargs):
    """Placeholder for audit logging; does nothing if real logger is absent."""
    pass  # or: print(f"[AUDIT LOG ignored] args={args}, kwargs={kwargs}")