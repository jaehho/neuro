"""Simulation cache: content-addressed storage backed by SQLite.

Hashes all physics-relevant Params fields (plus any optional early-stop
criterion) to produce a deterministic fingerprint.  If a matching run
exists in the registry, its saved parquet files are returned instead of
re-running the simulation.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, fields
from pathlib import Path

from neuro.convergence import ConvergenceCriterion, StreamingConvergence
from neuro.params import Params
from neuro.simulate import simulate

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


def _early_stop_dict(early_stop: StreamingConvergence | None) -> dict | None:
    """Serialize the streaming detector's config + target for hashing.

    The runtime state (queues, streak) is not part of the cache key — only
    the criterion and target matter, since they are what determine when
    the loop breaks for a given Params trajectory.
    """
    if early_stop is None:
        return None
    return {"criterion": asdict(early_stop.criterion), "target": early_stop.target}


def params_hash(p: Params, early_stop: StreamingConvergence | None = None) -> str:
    """Deterministic SHA-256 hex digest of all simulation-relevant fields.

    Including ``early_stop`` (when non-None) folds the convergence
    criterion and target into the hash so cells with different stopping
    behavior get distinct cache entries.
    """
    d: dict = {}
    for f in fields(p):
        if f.name in _EXCLUDED_FROM_HASH:
            continue
        val = getattr(p, f.name)
        # tuples → lists for canonical JSON
        if isinstance(val, tuple):
            val = list(val)
        d[f.name] = val
    es = _early_stop_dict(early_stop)
    if es is not None:
        d["__early_stop"] = es
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
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
    early_stop: StreamingConvergence | None = None,
    quiet: bool = False,
) -> dict:
    """Run simulation with content-addressed caching.

    If *force* is True, skip the cache lookup and always rerun (but still
    save the result to the cache, replacing any prior entry).

    ``progress`` is forwarded to ``simulate()`` on cache miss; see that
    function's docstring for the contract.

    ``early_stop`` is forwarded to ``simulate()`` and its config (criterion
    + target) is included in the hash, so two runs that share Params but
    differ in early-stop config get distinct cache entries.

    Returns the same dict as ``simulate()`` when given a parquet_path:
    ``{"parquet_path", "parquet_spikes_path", "rows_written", "spikes_written"}``.
    """
    hash_hex = params_hash(p, early_stop=early_stop)
    short = hash_hex[:12]
    db_path = cache_dir / "runs.db"

    if not force:
        hit = lookup_run(db_path, hash_hex)
        if hit is not None:
            pq = Path(hit["parquet_path"])
            sp = Path(hit["spikes_path"])
            if pq.exists() and sp.exists():
                if not quiet:
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
    rec = simulate(
        p,
        parquet_path=pq_path,
        chunk_rows=chunk_rows,
        progress=progress,
        early_stop=early_stop,
    )
    elapsed = time.monotonic() - t0

    spk_path = rec.get("parquet_spikes_path", pq_path.replace(".parquet", ".spikes.parquet"))
    register_run(db_path, hash_hex, p, pq_path, spk_path, elapsed)
    if not quiet:
        print(f"Cached as {short} ({elapsed:.1f}s)")
    return rec


def _delete_run(db_path: Path, hash_hex: str) -> None:
    conn = _init_db(db_path)
    try:
        conn.execute("DELETE FROM runs WHERE hash = ?", (hash_hex,))
        conn.commit()
    finally:
        conn.close()


def merge_runs_db(local_db: Path, other_db: Path) -> dict[str, int]:
    """Merge rows from *other_db* into *local_db* using INSERT OR IGNORE.

    Returns a count of rows seen and rows inserted.  Hash is the primary
    key, so duplicates are dropped silently (every cached run is
    content-addressed, so two databases agreeing on a hash agree on the
    run).
    """
    if not other_db.exists():
        raise FileNotFoundError(other_db)
    conn = _init_db(local_db)
    try:
        rows_before = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.execute("ATTACH DATABASE ? AS other", (str(other_db),))
        conn.execute(
            "INSERT OR IGNORE INTO runs "
            "(hash, params_json, record_every, created_at, duration_s, parquet_path, spikes_path) "
            "SELECT hash, params_json, record_every, created_at, duration_s, parquet_path, spikes_path "
            "FROM other.runs"
        )
        seen = conn.execute("SELECT COUNT(*) FROM other.runs").fetchone()[0]
        conn.execute("DETACH DATABASE other")
        conn.commit()
        rows_after = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    finally:
        conn.close()
    return {"seen": int(seen), "inserted": int(rows_after - rows_before), "total": int(rows_after)}
