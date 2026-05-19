<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# io_plugins

## Purpose
File format I/O plugins. HyperSpy delegates most format support to the `rsciio` package, which is imported here. Each plugin registers itself with HyperSpy's I/O dispatcher (`hyperspy/io.py`).

## For AI Agents

### Working In This Directory

- Most file format support is now in `rsciio` (a separate package), not here. Check `rsciio` first before adding format support directly.
- New plugins should follow the `rsciio` plugin protocol: implement `file_reader` and optionally `file_writer` functions, and register via `hyperspy_extension.yaml`.
- The load/save API entry point is `hyperspy/io.py` — do not call plugin functions directly.

### Adding a New Plugin

1. Check if `rsciio` already supports the format.
2. If not, create a new module here with `file_reader(filename, **kwargs)` returning a list of dicts.
3. Register the plugin in `hyperspy/hyperspy_extension.yaml`.
4. Add tests in `hyperspy/tests/` (see `test_io.py`).

### Testing Requirements

```bash
pytest hyperspy/tests/test_io.py
```

## Dependencies

### External
- `rsciio` — primary file format library
- Format-specific libraries (e.g. `h5py`, `tifffile`) loaded lazily per plugin

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
