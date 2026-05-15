"""Run/sweep journal — listing, tagging, resolution.

Tags + notes are persisted on disk:
- Run sidecar JSON gets ``tags: list[str]`` and ``note: str`` keys
  (see ``simulate._write_sidecar``).
- Sweep dirs get a sibling ``meta.json`` separate from ``base_params.json``
  (which is strict Params-only).

Old files lacking the keys read as empty.

The TUI is the human interface; this module is the script-level API:

    from neuro.journal import resolve, tag, list_entries, load_by_tag
    run = resolve("baseline")            # path → name → tag
    tag("baseline", "good-baseline")     # additive
    load_by_tag("good-baseline").serve() # bring back up
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from neuro.params import Params
from neuro.simulate import Run, _write_sidecar, load_latest, load_run

_TAG_RE = re.compile(r"^[\w-]+$")


# Columns in ``summary.parquet`` that are outputs/indices, not swept axes.
_SWEEP_OUTPUT_COLUMNS = frozenset({
    "i", "j", "r_post_late", "abs_error", "w_final", "duration_s",
})


@dataclass
class SweepAxis:
    """One swept axis detected from a ``summary.parquet``."""
    column: str
    min: float
    max: float
    n_unique: int

    def format(self) -> str:
        return f"{self.column}: {self.min:.3g} → {self.max:.3g}  ({self.n_unique} values)"


@dataclass
class SweepEntry:
    """A sweep run dir — ``output/<sweep>/<ts>/`` with ``summary.parquet``."""
    name: str
    sweep_ts: str
    path: Path
    summary_parquet: Path
    base_params: Params
    cell_count: int
    swept_axes: list[SweepAxis] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def saved_at(self) -> str:
        return self.sweep_ts

    @property
    def full_name(self) -> str:
        return f"{self.name}/{self.sweep_ts}"

    def serve(self, *, background: bool = False, **kwargs) -> str | None:
        """Open the Plotly heatmap + per-cell viewer in a browser.

        Blocking by default; pass ``background=True`` to run on a daemon
        thread and return the URL (the TUI uses this).
        """
        if background:
            from neuro.sweep_viewer import serve_sweep_background
            return serve_sweep_background(self.path, **kwargs)
        from neuro.sweep_viewer import serve_sweep
        serve_sweep(self.path, **kwargs)
        return None

    def summary(self) -> pl.DataFrame:
        return pl.read_parquet(str(self.summary_parquet))


class AmbiguousTagError(LookupError):
    """A tag matches more than one entry."""

    def __init__(self, tag: str, matches: list[Run | SweepEntry]):
        self.tag = tag
        self.matches = matches
        names = ", ".join(_display_name(m) for m in matches)
        super().__init__(f"Tag {tag!r} matches {len(matches)} entries: {names}")


def _display_name(entry: Run | SweepEntry) -> str:
    if isinstance(entry, Run):
        return f"{entry.name}/{entry.saved_at}"
    return entry.full_name


def _validate_tag(tag: str) -> str:
    if not _TAG_RE.match(tag):
        raise ValueError(f"Invalid tag {tag!r} — must match [\\w-]+")
    return tag


# ── Sweep meta.json ───────────────────────────────────────────────

def _sweep_meta_path(sweep_dir: Path) -> Path:
    return sweep_dir / "meta.json"


def _load_sweep_meta(sweep_dir: Path) -> tuple[list[str], str]:
    meta = _sweep_meta_path(sweep_dir)
    if not meta.exists():
        return ([], "")
    data = json.loads(meta.read_text())
    return (list(data.get("tags", [])), data.get("note", ""))


def _write_sweep_meta(sweep_dir: Path, tags: list[str], note: str) -> None:
    meta = _sweep_meta_path(sweep_dir)
    meta.write_text(
        json.dumps({"tags": list(tags), "note": note}, indent=2, sort_keys=True)
    )


def _detect_swept_axes(summary_pq: Path) -> list[SweepAxis]:
    """Identify columns that vary across cells. These are the swept axes.

    Excludes the standard output columns (i, j, r_post_late, abs_error,
    w_final, duration_s) and any column with a single unique value.
    """
    if not summary_pq.exists():
        return []
    try:
        df = pl.read_parquet(str(summary_pq))
    except (pl.exceptions.PolarsError, OSError):
        return []
    if df.is_empty():
        return []
    axes: list[SweepAxis] = []
    for col in df.columns:
        if col in _SWEEP_OUTPUT_COLUMNS:
            continue
        if not df[col].dtype.is_numeric():
            continue
        n_unique = df[col].n_unique()
        if n_unique <= 1:
            continue
        col_min = df.select(pl.col(col).min()).item()
        col_max = df.select(pl.col(col).max()).item()
        if col_min is None or col_max is None:
            continue
        axes.append(SweepAxis(
            column=col, min=float(col_min), max=float(col_max),
            n_unique=int(n_unique),
        ))
    return axes


def _load_sweep_entry(sweep_dir: Path) -> SweepEntry:
    base_params_path = sweep_dir / "base_params.json"
    if base_params_path.exists():
        base = Params(**json.loads(base_params_path.read_text()))
    else:
        base = Params()
    cell_count = sum(
        1 for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith("cell_")
    )
    tags, note = _load_sweep_meta(sweep_dir)
    return SweepEntry(
        name=sweep_dir.parent.name,
        sweep_ts=sweep_dir.name,
        path=sweep_dir,
        summary_parquet=sweep_dir / "summary.parquet",
        base_params=base,
        cell_count=cell_count,
        swept_axes=_detect_swept_axes(sweep_dir / "summary.parquet"),
        tags=tags,
        note=note,
    )


# ── Listing ───────────────────────────────────────────────────────

def _is_sweep_parent(name_dir: Path) -> bool:
    """A dir whose timestamped subdirs each have ``summary.parquet``."""
    try:
        children = list(name_dir.iterdir())
    except OSError:
        return False
    return any(
        d.is_dir() and (d / "summary.parquet").exists() for d in children
    )


def _try_load_run(parquet: Path) -> Run | None:
    """Best-effort load. Returns None if the sidecar is malformed/old-format."""
    try:
        return load_run(parquet)
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def list_entries(
    output_dir: str | Path = "output",
) -> tuple[list[Run], list[SweepEntry]]:
    """Top-level runs and sweeps (cell runs excluded). Newest first.

    Sidecar JSONs that don't conform to the current schema are silently
    skipped — they're typically pre-format leftovers in ``output/``.
    """
    root = Path(output_dir)
    runs: list[Run] = []
    sweeps: list[SweepEntry] = []
    if not root.is_dir():
        return runs, sweeps
    for name_dir in root.iterdir():
        if not name_dir.is_dir():
            continue
        if _is_sweep_parent(name_dir):
            for ts_dir in name_dir.iterdir():
                if ts_dir.is_dir() and (ts_dir / "summary.parquet").exists():
                    sweeps.append(_load_sweep_entry(ts_dir))
        else:
            for parquet in name_dir.iterdir():
                if parquet.suffix != ".parquet":
                    continue
                if parquet.name.endswith(".spikes.parquet"):
                    continue
                if not parquet.with_suffix(".json").exists():
                    continue
                run = _try_load_run(parquet)
                if run is not None:
                    runs.append(run)
    runs.sort(key=lambda r: r.saved_at, reverse=True)
    sweeps.sort(key=lambda s: s.sweep_ts, reverse=True)
    return runs, sweeps


def list_cell_runs(sweep: SweepEntry) -> list[Run]:
    """All cell runs inside a sweep, sorted by cell key."""
    out: list[Run] = []
    for cell_dir in sweep.path.iterdir():
        if not (cell_dir.is_dir() and cell_dir.name.startswith("cell_")):
            continue
        for parquet in cell_dir.iterdir():
            if parquet.suffix != ".parquet":
                continue
            if parquet.name.endswith(".spikes.parquet"):
                continue
            if parquet.with_suffix(".json").exists():
                out.append(load_run(parquet))
    out.sort(key=lambda r: r.name)
    return out


# ── Resolution ────────────────────────────────────────────────────

def _resolve_by_tag(
    tag: str, output_dir: str | Path = "output"
) -> list[Run | SweepEntry]:
    matches: list[Run | SweepEntry] = []
    root = Path(output_dir)
    if not root.is_dir():
        return matches
    for sidecar in root.rglob("*.json"):
        if sidecar.name == "base_params.json":
            continue
        try:
            data = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            continue
        if tag not in data.get("tags", []):
            continue
        if sidecar.name == "meta.json":
            sweep_dir = sidecar.parent
            if (sweep_dir / "summary.parquet").exists():
                matches.append(_load_sweep_entry(sweep_dir))
            continue
        parquet = sidecar.with_suffix(".parquet")
        if not parquet.exists():
            continue
        run = _try_load_run(parquet)
        if run is not None:
            matches.append(run)
    return matches


def resolve(
    target: str, output_dir: str | Path = "output"
) -> Run | SweepEntry:
    """Resolve a target. Tries: exact path → name under output/ → tag scan."""
    p = Path(target)
    if p.is_file() and p.suffix == ".parquet":
        return load_run(p)
    if p.is_dir() and (p / "summary.parquet").exists():
        return _load_sweep_entry(p)
    if p.is_dir():
        sweep_tsdirs = sorted(
            (d for d in p.iterdir()
             if d.is_dir() and (d / "summary.parquet").exists()),
            key=lambda d: d.name, reverse=True,
        )
        if sweep_tsdirs:
            return _load_sweep_entry(sweep_tsdirs[0])
    name_dir = Path(output_dir) / target
    if name_dir.is_dir():
        sweep_tsdirs = sorted(
            (d for d in name_dir.iterdir()
             if d.is_dir() and (d / "summary.parquet").exists()),
            key=lambda d: d.name, reverse=True,
        )
        if sweep_tsdirs:
            return _load_sweep_entry(sweep_tsdirs[0])
        try:
            return load_latest(target, output_dir=output_dir)
        except FileNotFoundError:
            pass
    matches = _resolve_by_tag(target, output_dir=output_dir)
    if not matches:
        raise FileNotFoundError(
            f"No path, run name, sweep name, or tag matches {target!r}"
        )
    if len(matches) > 1:
        raise AmbiguousTagError(target, matches)
    return matches[0]


def load_by_tag(
    tag: str, output_dir: str | Path = "output"
) -> Run | SweepEntry:
    """Resolve a tag to its single matching entry. Raises if 0 or >1 match."""
    matches = _resolve_by_tag(tag, output_dir=output_dir)
    if not matches:
        raise FileNotFoundError(f"No entry tagged {tag!r}")
    if len(matches) > 1:
        raise AmbiguousTagError(tag, matches)
    return matches[0]


# ── Tag/note mutations ────────────────────────────────────────────

def tag_run(run: Run, *tags: str) -> Run:
    """Add tags to a run (additive, deduped). Returns the updated Run."""
    new = sorted({*run.tags, *(_validate_tag(t) for t in tags)})
    run.tags = new
    _write_sidecar(run)
    return run


def untag_run(run: Run, *tags: str) -> Run:
    drop = set(tags)
    run.tags = [t for t in run.tags if t not in drop]
    _write_sidecar(run)
    return run


def set_run_note(run: Run, note: str) -> Run:
    run.note = note
    _write_sidecar(run)
    return run


def tag_sweep(sweep: SweepEntry, *tags: str) -> SweepEntry:
    new = sorted({*sweep.tags, *(_validate_tag(t) for t in tags)})
    sweep.tags = new
    _write_sweep_meta(sweep.path, sweep.tags, sweep.note)
    return sweep


def untag_sweep(sweep: SweepEntry, *tags: str) -> SweepEntry:
    drop = set(tags)
    sweep.tags = [t for t in sweep.tags if t not in drop]
    _write_sweep_meta(sweep.path, sweep.tags, sweep.note)
    return sweep


def set_sweep_note(sweep: SweepEntry, note: str) -> SweepEntry:
    sweep.note = note
    _write_sweep_meta(sweep.path, sweep.tags, sweep.note)
    return sweep


def tag(target: str, *tags: str, output_dir: str | Path = "output") -> Run | SweepEntry:
    """Tag whatever ``target`` resolves to."""
    entry = resolve(target, output_dir=output_dir)
    if isinstance(entry, Run):
        return tag_run(entry, *tags)
    return tag_sweep(entry, *tags)


def untag(target: str, *tags: str, output_dir: str | Path = "output") -> Run | SweepEntry:
    entry = resolve(target, output_dir=output_dir)
    if isinstance(entry, Run):
        return untag_run(entry, *tags)
    return untag_sweep(entry, *tags)


def set_note(target: str, note: str, output_dir: str | Path = "output") -> Run | SweepEntry:
    entry = resolve(target, output_dir=output_dir)
    if isinstance(entry, Run):
        return set_run_note(entry, note)
    return set_sweep_note(entry, note)
