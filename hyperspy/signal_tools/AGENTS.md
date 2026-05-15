<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# signal_tools

## Purpose
Interactive signal editing dialogs and tools. These provide GUI-accessible operations (smoothing, background removal, spike removal, calibration, etc.) that wrap core signal processing methods.

## Key Files

| File | Description |
|------|-------------|
| `_smoothing.py` | Interactive smoothing tool |
| `_background_removal.py` | Interactive background removal |
| `_spikes_removal.py` | Cosmic ray / spike removal tool |
| `_calibration.py` | Axis calibration dialog |
| `_image_contrast_editor.py` | Image contrast/gamma editor |
| `_peaks_finder2d.py` | 2D peak finding tool |
| `_line.py` | Line profile tool |
| `_selector.py` | Region selector |
| `_io.py` | I/O helpers for tool dialogs |
| `_message.py` | Message/status display helpers |

## For AI Agents

### Working In This Directory

- All tools here are interactive (GUI) — they depend on matplotlib widgets and the HyperSpy event system.
- Tools are triggered from signal methods (e.g. `signal.smooth()` with no backend triggers the interactive tool).
- Backend-agnostic: avoid hard-coding `tk` or `qt` — use `matplotlib.widgets` only.

### Testing Requirements

```bash
pytest hyperspy/tests/
```

Interactive tools are tested with matplotlib's Agg backend; simulate user interaction by calling internal methods directly rather than via GUI events.

## Dependencies

### Internal
- `hyperspy/drawing/` — rendering and widget infrastructure
- `hyperspy/events.py` — reactive updates

### External
- `matplotlib.widgets`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
