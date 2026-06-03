Improve keyboard shortcut support for macOS:

- Add ``super`` (Command/Windows key) to the modifier key enum.
- On macOS, change default modifiers: ``modifier_dims_01`` uses ``alt``
  (Option, ⌥) instead of ``ctrl`` — avoids Mission Control capture and
  cross-backend modifier-name inconsistencies with the Command key.
  ``modifier_dims_45`` uses ``alt+shift`` instead of ``ctrl+alt``
  (both modifiers are consistent across all macOS backends).
- All previously hardcoded drawing shortcuts (``e``, ``h``, ``l``,
  ``+``, ``-``, ``x``, ``c``, ``y``, ``u``, ``PageUp``, ``PageDown``)
  are now configurable via plot preferences.
- Log-scale toggle (``l`` key) is now configurable independently of the
  matplotlib default (which HyperSpy already suppresses); users can remap
  it without affecting matplotlib behaviour.
- Add documentation for macOS shortcut defaults and how to configure
  shortcuts when connecting to a remote HyperSpy instance from macOS.
- Add keyboard shortcuts for 1D model plots: ``a`` toggles adjust
  position, ``s`` toggles component visibility, ``d`` toggles the
  residual line. All three are configurable via preferences.
- Add ``hs.show_keybindings()`` to display all keyboard shortcuts with
  their current key bindings, grouped by category.
