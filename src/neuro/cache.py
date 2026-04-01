"""Simulation cache: content-addressed storage backed by SQLite.

Hashes all physics-relevant Params fields to produce a deterministic
fingerprint.  If a matching run exists in the registry, its saved
parquet files are returned instead of re-running the simulation.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import fields
from pathlib import Path

from neuro.sim import Params, simulate

# Fields that do NOT affect simulation dynamics and are excluded from the hash.
_EXCLUDED_FROM_HASH = frozenset({"record_every"})

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS runs (
    hash         TEXT PRIMARY KEY,
    params_json  TEXT NOT NULL,
    record_every REAL NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime')),
    duration_s   REAL,
    parquet_path TEXT NOT NULL,
    spikes_path  TEXT NOT NULL
)
"""


def params_hash(p: Params) -> str:
    """Deterministic SHA-256 hex digest of all simulation-relevant fields."""
    d: dict = {}
    for f in fields(p):
        if f.name in _EXCLUDED_FROM_HASH:
            continue
        val = getattr(p, f.name)
        # tuples → lists for canonical JSON
        if isinstance(val, tuple):
            val = list(val)
        d[f.name] = val
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def lookup_run(db_path: Path, hash_hex: str) -> dict | None:
    conn = _init_db(db_path)
    try:
        row = conn.execute(
            "SELECT created_at, parquet_path, spikes_path, record_every FROM runs WHERE hash = ?",
            (hash_hex,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "created_at": row[0],
        "parquet_path": row[1],
        "spikes_path": row[2],
        "record_every": row[3],
    }


def register_run(
    db_path: Path,
    hash_hex: str,
    p: Params,
    parquet_path: str,
    spikes_path: str,
    duration_s: float,
) -> None:
    d: dict = {}
    for f in fields(p):
        val = getattr(p, f.name)
        if isinstance(val, tuple):
            val = list(val)
        d[f.name] = val
    params_json = json.dumps(d, sort_keys=True, separators=(",", ":"))

    conn = _init_db(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO runs (hash, params_json, record_every, duration_s, parquet_path, spikes_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (hash_hex, params_json, p.record_every, duration_s, parquet_path, spikes_path),
        )
        conn.commit()
    finally:
        conn.close()


def cached_simulate(
    p: Params,
    *,
    cache_dir: Path = Path("output"),
    chunk_rows: int = 100_000,
    force: bool = False,
) -> dict:
    """Run simulation with content-addressed caching.

    If *force* is True, skip the cache lookup and always rerun (but still
    save the result to the cache, replacing any prior entry).

    Returns the same dict as ``simulate()`` when given a parquet_path:
    ``{"parquet_path", "parquet_spikes_path", "rows_written", "spikes_written"}``.
    """
    hash_hex = params_hash(p)
    short = hash_hex[:12]
    db_path = cache_dir / "runs.db"

    if not force:
        hit = lookup_run(db_path, hash_hex)
        if hit is not None:
            pq = Path(hit["parquet_path"])
            sp = Path(hit["spikes_path"])
            if pq.exists() and sp.exists():
                print(f"Cache hit: {short} (run from {hit['created_at']})")
                return {
                    "parquet_path": str(pq),
                    "parquet_spikes_path": str(sp),
                }
            # Stale entry — files were deleted
            _delete_run(db_path, hash_hex)

    # Force rerun: delete old entry so INSERT OR REPLACE works cleanly
    if force:
        _delete_run(db_path, hash_hex)

    cache_dir.mkdir(parents=True, exist_ok=True)
    pq_path = str(cache_dir / f"{short}.parquet")

    t0 = time.monotonic()
    rec = simulate(p, parquet_path=pq_path, chunk_rows=chunk_rows)
    elapsed = time.monotonic() - t0

    spk_path = rec.get("parquet_spikes_path", pq_path.replace(".parquet", ".spikes.parquet"))
    register_run(db_path, hash_hex, p, pq_path, spk_path, elapsed)
    print(f"Cached as {short} ({elapsed:.1f}s)")
    return rec


def _delete_run(db_path: Path, hash_hex: str) -> None:
    conn = _init_db(db_path)
    try:
        conn.execute("DELETE FROM runs WHERE hash = ?", (hash_hex,))
        conn.commit()
    finally:
        conn.close()
