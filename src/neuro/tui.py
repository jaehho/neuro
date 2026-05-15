"""Textual TUI: browse, tag, filter runs and sweeps.

The TUI is a catalog. The actual plot view is the Plotly HTTP viewer
in ``neuro.plotting`` / ``neuro.sweep_viewer`` — pressing ``enter`` on
a row launches the appropriate server on a daemon thread and opens
your browser.

Entry point is the ``neuro`` command (see ``pyproject.toml`` scripts).
``neuro`` with no args opens the TUI; ``neuro <target>`` skips the TUI
and serves the target directly (foreground, Ctrl-C to exit).

Keybinds (browse):
  enter — open in browser  t — tag           T — untag
  n — note                 / — filter        r — refresh    q — quit
"""
from __future__ import annotations

import sys

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from neuro.journal import (
    AmbiguousTagError,
    SweepEntry,
    list_entries,
    resolve,
    set_run_note,
    set_sweep_note,
    tag_run,
    tag_sweep,
    untag_run,
    untag_sweep,
)
from neuro.params import Params, diff_from_defaults
from neuro.simulate import Run


# ── helpers ────────────────────────────────────────────────────────

def _params_summary(p: Params) -> str:
    diff = diff_from_defaults(p)
    if not diff:
        return "params: (defaults)"
    width = max(len(k) for k in diff)
    lines = ["params (diff from defaults):"]
    for k, v in sorted(diff.items()):
        lines.append(f"  {k.ljust(width)} = {v!r}")
    return "\n".join(lines)


def _run_row(r: Run) -> tuple[str, str, str, str, str]:
    return ("run", r.name, r.saved_at, " ".join(r.tags), r.note)


def _sweep_row(s: SweepEntry) -> tuple[str, str, str, str, str]:
    return ("sweep", f"{s.name} ({s.cell_count})", s.sweep_ts,
            " ".join(s.tags), s.note)


def _entry_label(e: Run | SweepEntry) -> str:
    if isinstance(e, Run):
        return f"{e.name}/{e.saved_at}"
    return e.full_name


# ── text input modal ──────────────────────────────────────────────

class TextInputModal(ModalScreen[str | None]):
    """Single-line input modal. Returns the value, or ``None`` on cancel."""

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    DEFAULT_CSS = """
    TextInputModal {
        align: center middle;
    }
    TextInputModal > Vertical {
        width: 70;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: solid $primary;
    }
    TextInputModal Static {
        margin-bottom: 1;
    }
    """

    def __init__(self, prompt: str, value: str = "") -> None:
        super().__init__()
        self.prompt = prompt
        self.initial_value = value

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.prompt)
            yield Input(value=self.initial_value, id="modal-input")

    def on_mount(self) -> None:
        self.query_one("#modal-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── browse screen ─────────────────────────────────────────────────

class BrowseScreen(Screen):
    BINDINGS = [
        Binding("enter", "open", "open"),
        Binding("t", "tag", "tag"),
        Binding("T", "untag", "untag"),
        Binding("n", "note", "note"),
        Binding("r", "refresh", "refresh"),
        Binding("slash", "filter", "filter"),
        Binding("q", "quit", "quit"),
    ]

    DEFAULT_CSS = """
    DataTable { width: 60%; }
    #details {
        width: 40%;
        padding: 1;
        border-left: solid $primary;
    }
    """

    def __init__(self, filter_text: str = "") -> None:
        super().__init__()
        self.entries: list[Run | SweepEntry] = []
        self.filter_text = filter_text
        self.title = "neuro — runs & sweeps"
        # URLs of viewers we've launched, so the user can see them in the
        # status bar; daemon threads die with the TUI process.
        self._served: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DataTable(id="entries", cursor_type="row")
            yield Static(id="details")
        yield Footer()

    def on_mount(self) -> None:
        tbl = self.query_one("#entries", DataTable)
        tbl.add_columns("kind", "name", "saved_at", "tags", "note")
        self._refresh_entries()

    def _refresh_entries(self) -> None:
        runs, sweeps = list_entries()
        merged: list[Run | SweepEntry] = [*runs, *sweeps]
        merged.sort(key=lambda e: e.saved_at, reverse=True)
        ft = self.filter_text.lower().strip()
        if ft:
            merged = [e for e in merged
                      if ft in _entry_label(e).lower()
                      or any(ft in t.lower() for t in e.tags)
                      or ft in e.note.lower()]
        self.entries = merged
        tbl = self.query_one("#entries", DataTable)
        tbl.clear()
        for e in self.entries:
            row = _run_row(e) if isinstance(e, Run) else _sweep_row(e)
            tbl.add_row(*row)
        self._update_details()
        self._update_subtitle()

    def _update_subtitle(self) -> None:
        parts = [f"{len(self.entries)} entries"]
        if self.filter_text:
            parts.append(f"filter: {self.filter_text}")
        if self._served:
            parts.append(f"serving {len(self._served)}: {self._served[-1]}")
        self.sub_title = " · ".join(parts)

    def _selected(self) -> Run | SweepEntry | None:
        tbl = self.query_one("#entries", DataTable)
        if not self.entries:
            return None
        idx = tbl.cursor_row
        if 0 <= idx < len(self.entries):
            return self.entries[idx]
        return None

    def _update_details(self) -> None:
        d = self.query_one("#details", Static)
        e = self._selected()
        if e is None:
            d.update("(no entries — try `r` to refresh)")
            return
        if isinstance(e, Run):
            lines = [
                f"run: {e.name}",
                f"saved: {e.saved_at}",
                f"duration: {e.duration_s:.1f}s",
                f"rows: {e.rows_written:,}  spikes: {e.spikes_written:,}",
                f"tags: {', '.join(e.tags) or '—'}",
                f"note: {e.note or '—'}",
                "",
                _params_summary(e.params),
                "",
                f"parquet: {e.parquet}",
            ]
        else:
            lines = [
                f"sweep: {e.full_name}",
                f"cells: {e.cell_count}",
                f"tags: {', '.join(e.tags) or '—'}",
                f"note: {e.note or '—'}",
                "",
            ]
            if e.swept_axes:
                lines.append("swept axes:")
                for ax in e.swept_axes:
                    lines.append(f"  {ax.format()}")
            else:
                lines.append("swept axes: (none detected)")
            lines.extend([
                "",
                _params_summary(e.base_params),
                "",
                f"path: {e.path}",
            ])
        d.update("\n".join(lines))

    def on_data_table_row_highlighted(self, _event) -> None:
        self._update_details()

    def on_data_table_row_selected(self, _event) -> None:
        # DataTable consumes enter and emits RowSelected — route to `open`.
        self.action_open()

    # actions

    def action_open(self) -> None:
        e = self._selected()
        if e is None:
            return
        try:
            url = e.serve(background=True)
        except Exception as exc:  # noqa: BLE001 — surface any serve failure
            self.notify(f"serve failed: {exc}", severity="error")
            return
        if url:
            self._served.append(url)
            self.notify(f"{_entry_label(e)} → {url}", timeout=4)
            self._update_subtitle()

    def action_tag(self) -> None:
        e = self._selected()
        if e is None:
            return

        def cb(value: str | None) -> None:
            if not value:
                return
            tags = [t for t in value.split() if t]
            try:
                if isinstance(e, Run):
                    tag_run(e, *tags)
                else:
                    tag_sweep(e, *tags)
            except ValueError as exc:
                self.notify(str(exc), severity="error")
                return
            self._refresh_entries()

        self.app.push_screen(
            TextInputModal("add tag(s) (space-separated, [\\w-]+):"),
            cb,
        )

    def action_untag(self) -> None:
        e = self._selected()
        if e is None or not e.tags:
            return

        def cb(value: str | None) -> None:
            if not value:
                return
            tags = [t for t in value.split() if t]
            if isinstance(e, Run):
                untag_run(e, *tags)
            else:
                untag_sweep(e, *tags)
            self._refresh_entries()

        self.app.push_screen(
            TextInputModal(f"remove tag(s) from [{' '.join(e.tags)}]:"),
            cb,
        )

    def action_note(self) -> None:
        e = self._selected()
        if e is None:
            return

        def cb(value: str | None) -> None:
            if value is None:
                return
            if isinstance(e, Run):
                set_run_note(e, value)
            else:
                set_sweep_note(e, value)
            self._refresh_entries()

        self.app.push_screen(TextInputModal("note:", value=e.note), cb)

    def action_filter(self) -> None:
        def cb(value: str | None) -> None:
            if value is None:
                return
            self.filter_text = value
            self._refresh_entries()

        self.app.push_screen(
            TextInputModal("filter (matches name, tag, or note):",
                           value=self.filter_text),
            cb,
        )

    def action_refresh(self) -> None:
        self._refresh_entries()

    def action_quit(self) -> None:
        self.app.exit()


# ── app ───────────────────────────────────────────────────────────

class NeuroApp(App):
    def __init__(self, *, filter_text: str = "") -> None:
        super().__init__()
        self._filter_text = filter_text

    def on_mount(self) -> None:
        self.push_screen(BrowseScreen(filter_text=self._filter_text))


_HELP = """\
neuro — browse, tag, and view runs/sweeps

  neuro                 launch the TUI (browse all runs + sweeps)
  neuro <target>        skip the TUI and serve <target> in the browser
                        (Ctrl-C to stop). <target> is one of:
                          - parquet path  (output/baseline/<ts>.parquet)
                          - sweep dir     (output/ceiling-sweep/<ts>/)
                          - run name      (baseline)
                          - sweep name    (ceiling-sweep)
                          - tag           (good-baseline)

In the TUI:
  enter — open in browser (Plotly viewer on a daemon thread)
  t / T  — add / remove tag       n — note
  /      — filter                 r — refresh        q — quit
"""


def run() -> None:
    """``neuro`` command entry point."""
    args = sys.argv[1:]
    if args and args[0] in {"-h", "--help"}:
        print(_HELP)
        return
    if not args:
        NeuroApp().run()
        return
    target = " ".join(args)
    try:
        entry = resolve(target)
    except AmbiguousTagError as exc:
        print(f"neuro: {exc}", file=sys.stderr)
        print("  → opening browse view filtered to matches", file=sys.stderr)
        NeuroApp(filter_text=target).run()
        return
    except FileNotFoundError as exc:
        print(f"neuro: {exc}", file=sys.stderr)
        sys.exit(1)
    # Serve directly (foreground, Ctrl-C exits).
    entry.serve()
