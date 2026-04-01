"""Tests for simulation cache (content-addressed storage)."""
from __future__ import annotations

from pathlib import Path

import pytest

from neuro.cache import (
    _EXCLUDED_FROM_HASH,
    _init_db,
    cached_simulate,
    lookup_run,
    params_hash,
    register_run,
)
from neuro.sim import Params

from dataclasses import fields


# ---------------------------------------------------------------------------
# params_hash
# ---------------------------------------------------------------------------

class TestParamsHash:
    def test_deterministic(self) -> None:
        p = Params(T=1.0, seed=1)
        assert params_hash(p) == params_hash(p)

    def test_changes_with_seed(self) -> None:
        assert params_hash(Params(seed=1)) != params_hash(Params(seed=2))

    def test_changes_with_physics(self) -> None:
        base = params_hash(Params())
        assert params_hash(Params(tau_m=0.05)) != base
        assert params_hash(Params(w0=5.0)) != base
        assert params_hash(Params(neuromod_type="gated")) != base

    def test_ignores_record_every(self) -> None:
        assert params_hash(Params(record_every=1e-4)) == params_hash(Params(record_every=1e-3))

    def test_scalar_tuple_equivalent(self) -> None:
        # __post_init__ normalizes scalars to tuples, so hash should match
        assert params_hash(Params(n_pre=1, w0=2.0)) == params_hash(Params(n_pre=1, w0=(2.0,)))

    def test_hash_length(self) -> None:
        assert len(params_hash(Params())) == 64  # SHA-256

    def test_all_fields_covered(self) -> None:
        all_names = {f.name for f in fields(Params)}
        hashed = all_names - _EXCLUDED_FROM_HASH
        # Every Params field is either hashed or explicitly excluded
        assert hashed | _EXCLUDED_FROM_HASH == all_names


# ---------------------------------------------------------------------------
# SQLite registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_init_creates_db(self, tmp_path: Path) -> None:
        db = tmp_path / "runs.db"
        conn = _init_db(db)
        conn.close()
        assert db.exists()

    def test_lookup_miss(self, tmp_path: Path) -> None:
        db = tmp_path / "runs.db"
        assert lookup_run(db, "abc123") is None

    def test_register_and_lookup(self, tmp_path: Path) -> None:
        db = tmp_path / "runs.db"
        p = Params(T=0.01)
        register_run(db, "abc123", p, "out.parquet", "out.spikes.parquet", 1.5)
        hit = lookup_run(db, "abc123")
        assert hit is not None
        assert hit["parquet_path"] == "out.parquet"
        assert hit["spikes_path"] == "out.spikes.parquet"
        assert hit["created_at"]  # non-empty timestamp

    def test_register_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "runs.db"
        p = Params(T=0.01)
        register_run(db, "abc123", p, "a.parquet", "a.spikes.parquet", 1.0)
        register_run(db, "abc123", p, "b.parquet", "b.spikes.parquet", 2.0)
        hit = lookup_run(db, "abc123")
        # INSERT OR IGNORE keeps the first
        assert hit["parquet_path"] == "a.parquet"


# ---------------------------------------------------------------------------
# cached_simulate integration
# ---------------------------------------------------------------------------

class TestCachedSimulate:
    def test_cache_miss_creates_files(self, tmp_path: Path) -> None:
        p = Params(T=0.01)
        rec = cached_simulate(p, cache_dir=tmp_path)
        assert Path(rec["parquet_path"]).exists()
        assert Path(rec["parquet_spikes_path"]).exists()

    def test_cache_hit_skips_simulation(self, tmp_path: Path) -> None:
        p = Params(T=0.01)
        rec1 = cached_simulate(p, cache_dir=tmp_path)
        rec2 = cached_simulate(p, cache_dir=tmp_path)
        assert rec1["parquet_path"] == rec2["parquet_path"]

    def test_different_params_miss(self, tmp_path: Path) -> None:
        rec1 = cached_simulate(Params(T=0.01, seed=1), cache_dir=tmp_path)
        rec2 = cached_simulate(Params(T=0.01, seed=2), cache_dir=tmp_path)
        assert rec1["parquet_path"] != rec2["parquet_path"]

    def test_stale_entry_reruns(self, tmp_path: Path) -> None:
        p = Params(T=0.01)
        rec = cached_simulate(p, cache_dir=tmp_path)
        # Delete the parquet files to simulate stale entry
        Path(rec["parquet_path"]).unlink()
        Path(rec["parquet_spikes_path"]).unlink()
        # Should re-run and recreate
        rec2 = cached_simulate(p, cache_dir=tmp_path)
        assert Path(rec2["parquet_path"]).exists()

    def test_cache_dir_created(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "nested" / "cache"
        cached_simulate(Params(T=0.01), cache_dir=new_dir)
        assert new_dir.exists()
