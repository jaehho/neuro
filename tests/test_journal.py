"""Journal tests: tag round-trip, back-compat, resolution precedence."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import polars as pl
import pytest

from neuro.journal import (
    AmbiguousTagError,
    SweepEntry,
    _load_sweep_entry,
    list_entries,
    load_by_tag,
    resolve,
    set_run_note,
    set_sweep_note,
    tag_run,
    tag_sweep,
    untag_run,
    untag_sweep,
)
from neuro.params import Params
from neuro.simulate import Run, load_run


# ── helpers ───────────────────────────────────────────────────────

def _write_run_sidecar(
    output_dir: Path,
    name: str,
    ts: str = "20260101_000000",
    *,
    tags: list[str] | None = None,
    note: str = "",
    include_keys: bool = True,
) -> Path:
    """Write a fake sidecar JSON and return the parquet path it points at."""
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    parquet = run_dir / f"{ts}.parquet"
    sidecar = run_dir / f"{ts}.json"
    p = Params()
    payload = {
        "name": name,
        "saved_at": ts,
        "duration_s": 1.0,
        "rows_written": 100,
        "spikes_written": 5,
        "converged_at": None,
        "params": asdict(p),
    }
    if include_keys:
        payload["tags"] = tags or []
        payload["note"] = note
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True))
    parquet.touch()  # rglob needs the file to exist; load_run doesn't read it
    return parquet


def _make_run(output_dir: Path, name: str, **kwargs) -> Run:
    return load_run(_write_run_sidecar(output_dir, name, **kwargs))


def _make_sweep(output_dir: Path, name: str, ts: str = "20260101_000000",
                *, tags: list[str] | None = None, note: str = "",
                summary: pl.DataFrame | None = None) -> SweepEntry:
    sweep_dir = output_dir / name / ts
    sweep_dir.mkdir(parents=True, exist_ok=True)
    p = Params()
    (sweep_dir / "base_params.json").write_text(
        json.dumps(asdict(p), indent=2, sort_keys=True)
    )
    if summary is None:
        summary = pl.DataFrame({
            "i": [0], "j": [0], "r_pre": [10.0], "r_target": [10.0],
            "r_post_late": [10.0], "abs_error": [0.0], "w_final": [1.0],
            "duration_s": [1.0],
        })
    summary.write_parquet(sweep_dir / "summary.parquet")
    if tags or note:
        (sweep_dir / "meta.json").write_text(
            json.dumps({"tags": tags or [], "note": note}, indent=2, sort_keys=True)
        )
    return _load_sweep_entry(sweep_dir)


# ── run tags ──────────────────────────────────────────────────────

def test_run_tag_roundtrip(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "output", "baseline")
    assert run.tags == []
    tag_run(run, "good", "demo")
    reloaded = load_run(run.parquet)
    assert reloaded.tags == ["demo", "good"]  # sorted + deduped


def test_run_tag_dedup_additive(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "output", "baseline", tags=["existing"])
    tag_run(run, "new", "existing")
    assert load_run(run.parquet).tags == ["existing", "new"]


def test_run_untag(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "output", "baseline", tags=["a", "b", "c"])
    untag_run(run, "b")
    assert load_run(run.parquet).tags == ["a", "c"]


def test_run_note_roundtrip(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "output", "baseline")
    set_run_note(run, "first stable")
    assert load_run(run.parquet).note == "first stable"


def test_old_sidecar_back_compat(tmp_path: Path) -> None:
    """Sidecar JSON missing `tags`/`note` reads as empty."""
    parquet = _write_run_sidecar(
        tmp_path / "output", "baseline", include_keys=False,
    )
    run = load_run(parquet)
    assert run.tags == []
    assert run.note == ""


def test_invalid_tag_rejected(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "output", "baseline")
    with pytest.raises(ValueError, match="Invalid tag"):
        tag_run(run, "has space")
    with pytest.raises(ValueError, match="Invalid tag"):
        tag_run(run, "has!punct")


# ── resolution ────────────────────────────────────────────────────

def test_resolve_by_name(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _make_run(out, "baseline")
    entry = resolve("baseline", output_dir=out)
    assert isinstance(entry, Run)
    assert entry.name == "baseline"


def test_resolve_by_tag(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _make_run(out, "baseline", tags=["good"])
    entry = resolve("good", output_dir=out)
    assert isinstance(entry, Run)
    assert "good" in entry.tags


def test_resolve_path_beats_name(tmp_path: Path) -> None:
    out = tmp_path / "output"
    run = _make_run(out, "baseline")
    entry = resolve(str(run.parquet), output_dir=out)
    assert isinstance(entry, Run)


def test_resolve_name_beats_tag(tmp_path: Path) -> None:
    """A run dir named "good" should win over a tag "good"."""
    out = tmp_path / "output"
    _make_run(out, "other", tags=["good"])
    _make_run(out, "good")
    entry = resolve("good", output_dir=out)
    assert isinstance(entry, Run)
    assert entry.name == "good"


def test_resolve_ambiguous_tag_raises(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _make_run(out, "baseline-a", tags=["dup"])
    _make_run(out, "baseline-b", tags=["dup"])
    with pytest.raises(AmbiguousTagError) as excinfo:
        resolve("dup", output_dir=out)
    assert len(excinfo.value.matches) == 2


def test_resolve_missing(tmp_path: Path) -> None:
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        resolve("nonexistent", output_dir=out)


def test_load_by_tag_single(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _make_run(out, "baseline", tags=["pick-me"])
    entry = load_by_tag("pick-me", output_dir=out)
    assert isinstance(entry, Run)
    assert "pick-me" in entry.tags


def test_load_by_tag_ambiguous(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _make_run(out, "a", tags=["dup"])
    _make_run(out, "b", tags=["dup"])
    with pytest.raises(AmbiguousTagError):
        load_by_tag("dup", output_dir=out)


# ── sweeps ────────────────────────────────────────────────────────

def test_sweep_tag_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "output"
    sweep = _make_sweep(out, "my-sweep")
    assert sweep.tags == []
    tag_sweep(sweep, "interesting")
    reloaded = resolve("my-sweep", output_dir=out)
    assert isinstance(reloaded, SweepEntry)
    assert reloaded.tags == ["interesting"]


def test_sweep_untag(tmp_path: Path) -> None:
    out = tmp_path / "output"
    sweep = _make_sweep(out, "my-sweep", tags=["a", "b"])
    untag_sweep(sweep, "a")
    reloaded = resolve("my-sweep", output_dir=out)
    assert isinstance(reloaded, SweepEntry)
    assert reloaded.tags == ["b"]


def test_sweep_note(tmp_path: Path) -> None:
    out = tmp_path / "output"
    sweep = _make_sweep(out, "my-sweep")
    set_sweep_note(sweep, "saturation cliff")
    reloaded = resolve("my-sweep", output_dir=out)
    assert isinstance(reloaded, SweepEntry)
    assert reloaded.note == "saturation cliff"


def test_sweep_swept_axes_detected(tmp_path: Path) -> None:
    """Columns that vary across cells are surfaced as swept axes; outputs aren't."""
    out = tmp_path / "output"
    summary = pl.DataFrame({
        "i":           [0, 0, 1, 1],
        "j":           [0, 1, 0, 1],
        "r_pre":       [5.0, 10.0, 5.0, 10.0],
        "r_target":    [3.0, 3.0, 8.0, 8.0],
        "r_post_late": [3.1, 9.0, 7.5, 10.0],
        "abs_error":   [0.1, 6.0, 0.5, 2.0],
        "w_final":     [1.0, 2.0, 1.5, 1.8],
        "duration_s":  [1.0, 1.0, 1.0, 1.0],
    })
    sweep = _make_sweep(out, "demo", summary=summary)
    cols = {a.column for a in sweep.swept_axes}
    # r_pre and r_target vary → axes. r_post_late/abs_error/w_final are outputs.
    # duration_s is a single value → not an axis.
    assert cols == {"r_pre", "r_target"}
    by_col = {a.column: a for a in sweep.swept_axes}
    assert by_col["r_pre"].min == 5.0
    assert by_col["r_pre"].max == 10.0
    assert by_col["r_pre"].n_unique == 2
    assert by_col["r_target"].n_unique == 2


def test_sweep_swept_axes_empty_for_trivial_sweep(tmp_path: Path) -> None:
    """A single-cell sweep with no varying columns has no axes."""
    out = tmp_path / "output"
    sweep = _make_sweep(out, "single")  # default summary has only 1 row
    assert sweep.swept_axes == []


def test_resolve_tag_matches_sweep(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _make_sweep(out, "ceiling", tags=["bright"])
    entry = resolve("bright", output_dir=out)
    assert isinstance(entry, SweepEntry)
    assert entry.name == "ceiling"


# ── listing ───────────────────────────────────────────────────────

def test_list_entries_separates(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _make_run(out, "loose-run")
    _make_sweep(out, "my-sweep")
    runs, sweeps = list_entries(output_dir=out)
    assert [r.name for r in runs] == ["loose-run"]
    assert [s.name for s in sweeps] == ["my-sweep"]


def test_list_entries_excludes_cell_runs(tmp_path: Path) -> None:
    """Cell runs under a sweep dir should not appear in the top-level run list."""
    out = tmp_path / "output"
    _make_sweep(out, "my-sweep")
    sweep_dir = out / "my-sweep" / "20260101_000000"
    cell = sweep_dir / "cell_00_00"
    _write_run_sidecar(cell.parent, "cell_00_00", ts="20260101_000001")
    runs, sweeps = list_entries(output_dir=out)
    assert runs == []
    assert len(sweeps) == 1
