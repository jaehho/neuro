# neuro

Neuromodulated STDP simulation: pre-synaptic neurons → 1 post-synaptic LIF neuron with three-factor learning rules (Frémaux & Gerstner 2016).

See `CLAUDE.md` for project structure. Quick start:

```bash
uv sync
uv run neuro --help          # discover subcommands
uv run neuro run             # interactive wizard for a single simulation
uv run neuro list            # browse cached runs
uv run neuro sweep run       # interactive 2D sweep
```
