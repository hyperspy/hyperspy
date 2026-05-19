<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# drawing

## Purpose
Matplotlib-based interactive plotting engine. Handles the `signal.plot()` infrastructure, including multi-panel figures, navigator/signal separation, interactive widgets, and the markers system.

## Key Files

| File | Description |
|------|-------------|
| `signal.py` | `SignalFigure` — top-level figure manager for any signal |
| `signal1d.py` | 1D signal panel rendering |
| `image.py` | 2D image panel rendering |
| `figure.py` | `BlittedFigure` — base figure with blit-based animation |
| `mpl_hse.py` | Hyperspy Signal Explorer figure layout |
| `mpl_hie.py` | Hyperspy Image Explorer figure layout |
| `mpl_he.py` | Hyperspy Explorer base |
| `markers.py` | Public markers entry point |
| `widget.py` | Navigator widget base |
| `widgets.py` | Concrete interactive widgets (crosshair, range, etc.) |
| `tiles.py` | Tiled signal display |
| `utils.py` | Plot utility helpers |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `_markers/` | Individual marker type implementations (Arrow, Circle, Line, etc.) |
| `_widgets/` | Low-level interactive widget implementations |

## For AI Agents

### Working In This Directory

- All rendering goes through `signal.plot()` which dispatches to the appropriate figure class based on signal dimension.
- Interactive updates rely on `events.py` — do not poll; connect/disconnect event handlers.
- Use blit (`figure.render_figure()`) for performance-critical animations.
- Markers are composable: each marker type in `_markers/` is independent; they are collected by `markers.py`.

### Testing Requirements

```bash
pytest hyperspy/tests/drawing/
```

Use `matplotlib.use('Agg')` or the `mpl_cleanup` fixture for non-interactive test rendering.

## Dependencies

### Internal
- `hyperspy/events.py` — reactive event system
- `hyperspy/roi.py` — ROI widgets connect to drawing widgets

### External
- `matplotlib` — entire rendering stack

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
