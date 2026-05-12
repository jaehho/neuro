# neuro

Neuromodulated STDP simulation: pre-synaptic neurons → 1 post-synaptic LIF neuron with three-factor learning rules (Frémaux & Gerstner 2016).

```bash
uv sync
uv run python experiments/baseline.py            # 1-pre Fetz-style demo
uv run python experiments/credit_assignment.py   # 2-pre contingent reward
uv run python experiments/ceiling_sweep.py       # 14×14 sweep + heatmap
uv run pytest                                    # tests
```

Each experiment is a self-contained Python file: build a `Params`, call `simulate(p, name=...)`, and (optionally) open the zoom-adaptive viewer. Runs are written to `output/<name>/<timestamp>.parquet` with a JSON sidecar; reload with `load_latest("baseline")` and call `.serve()` to inspect later.

The unified write-up is `docs/main.typ`. See `CLAUDE.md` for project structure and conventions.
