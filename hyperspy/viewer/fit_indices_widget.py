# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

"""Live-updating anywidget for FitIndices fitting progress.

Use :meth:`~hyperspy.fit_indices.FitIndices.display` instead of
constructing this class directly.
"""

import anywidget
import numpy as np
import traitlets


class FitIndicesWidget(anywidget.AnyWidget):
    """A live-updating canvas widget that shows fitting progress.

    Each cell in the canvas grid represents a block of navigation positions.
    Its shade encodes the fraction of positions in that block that have been
    fitted: light grey (0 %) → green (100 %).  The block containing
    :attr:`~hyperspy.fit_indices.FitIndices.current_index` is highlighted
    with an amber border.

    Do not construct this class directly — call
    :meth:`~hyperspy.fit_indices.FitIndices.display` instead.
    """

    # ------------------------------------------------------------------ #
    # Traitlets synced with JS                                            #
    # ------------------------------------------------------------------ #

    #: float32 bytes, row-major, shape (grid_rows, grid_cols).
    #: Each value is the fraction [0, 1] of positions fitted in that block.
    grid_data = traitlets.Bytes(b"").tag(sync=True)

    grid_rows = traitlets.Int(1).tag(sync=True)
    grid_cols = traitlets.Int(1).tag(sync=True)

    #: Block row/col of the current index; -1 when no fit is running.
    current_row = traitlets.Int(-1).tag(sync=True)
    current_col = traitlets.Int(-1).tag(sync=True)

    fitted_count = traitlets.Int(0).tag(sync=True)
    total_count = traitlets.Int(1).tag(sync=True)
    strategy = traitlets.Unicode("").tag(sync=True)
    current_index_str = traitlets.Unicode("").tag(sync=True)

    # ------------------------------------------------------------------ #
    # Inline ESM — canvas renderer                                        #
    # ------------------------------------------------------------------ #

    _esm = r"""
    function render({ model, el }) {
        const CELL   = 9;   // px per block cell
        const GAP    = 1;   // px gap between cells
        const STEP   = CELL + GAP;

        /* ── layout ─────────────────────────────────────────── */
        const root = document.createElement('div');
        root.style.cssText = 'display:inline-block; font-family:monospace;';

        const infoBar = document.createElement('div');
        infoBar.style.cssText =
            'font-size:12px; padding:3px 6px; background:#f0f0f0;' +
            'border:1px solid #ccc; border-bottom:none; border-radius:3px 3px 0 0;' +
            'white-space:nowrap; min-width:200px;';

        const canvas = document.createElement('canvas');
        canvas.style.cssText = 'display:block; border:1px solid #ccc; border-radius:0 0 3px 3px; cursor:crosshair;';

        root.appendChild(infoBar);
        root.appendChild(canvas);
        el.appendChild(root);

        const ctx = canvas.getContext('2d');

        /* ── tooltip ─────────────────────────────────────────── */
        const tip = document.createElement('div');
        tip.style.cssText =
            'position:fixed; display:none; background:rgba(0,0,0,0.75); color:#fff;' +
            'font-size:11px; font-family:monospace; padding:3px 7px; border-radius:3px;' +
            'pointer-events:none; z-index:9999;';
        document.body.appendChild(tip);

        canvas.addEventListener('mousemove', (e) => {
            const rect  = canvas.getBoundingClientRect();
            const mx    = e.clientX - rect.left;
            const my    = e.clientY - rect.top;
            const cols  = model.get('grid_cols');
            const rows  = model.get('grid_rows');
            const c     = Math.floor(mx / STEP);
            const r     = Math.floor(my / STEP);
            if (c >= 0 && c < cols && r >= 0 && r < rows) {
                const dataBytes = model.get('grid_data');
                const data      = new Float32Array(dataBytes.buffer);
                const frac      = data[r * cols + c];
                const pct       = isNaN(frac) ? 'n/a' : (frac * 100).toFixed(1) + '%';
                tip.textContent = `block (row=${r}, col=${c}) — ${pct} fitted`;
                tip.style.left  = (e.clientX + 12) + 'px';
                tip.style.top   = (e.clientY - 8)  + 'px';
                tip.style.display = 'block';
            } else {
                tip.style.display = 'none';
            }
        });
        canvas.addEventListener('mouseleave', () => { tip.style.display = 'none'; });

        /* ── colour helpers ──────────────────────────────────── */
        function fracToRgb(t) {
            // 0 → grey (220,220,220)   1 → green (76,175,80)
            if (isNaN(t)) return 'rgb(240,240,240)';
            const r = Math.round(220 + (76  - 220) * t) | 0;
            const g = Math.round(220 + (175 - 220) * t) | 0;
            const b = Math.round(220 + (80  - 220) * t) | 0;
            return `rgb(${r},${g},${b})`;
        }

        /* ── main draw ───────────────────────────────────────── */
        function draw() {
            const rows    = model.get('grid_rows');
            const cols    = model.get('grid_cols');
            const curR    = model.get('current_row');
            const curC    = model.get('current_col');
            const fitted  = model.get('fitted_count');
            const total   = model.get('total_count');
            const strat   = model.get('strategy');
            const curStr  = model.get('current_index_str');

            const W = GAP + cols * STEP;
            const H = GAP + rows * STEP;
            canvas.width  = W;
            canvas.height = H;
            canvas.style.width  = W + 'px';
            canvas.style.height = H + 'px';

            ctx.fillStyle = '#e8e8e8';
            ctx.fillRect(0, 0, W, H);

            const dataBytes = model.get('grid_data');
            if (!dataBytes || dataBytes.byteLength === 0) return;
            const data = new Float32Array(dataBytes.buffer);

            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const frac = data[r * cols + c];
                    const x = GAP + c * STEP;
                    const y = GAP + r * STEP;

                    ctx.fillStyle = fracToRgb(frac);
                    ctx.fillRect(x, y, CELL, CELL);

                    if (r === curR && c === curC) {
                        // Amber highlight border (2 px, inset by 0.5)
                        ctx.strokeStyle = '#ffc107';
                        ctx.lineWidth   = 2;
                        ctx.strokeRect(x + 0.5, y + 0.5, CELL - 1, CELL - 1);
                    }
                }
            }

            /* info bar */
            const pct      = total > 0 ? ((fitted / total) * 100).toFixed(1) : '0.0';
            const stratStr = strat || 'custom';
            const curPart  = curStr ? ` \u2022 current\u00a0${curStr}` : '';
            infoBar.textContent = `${fitted}\u00a0/\u00a0${total} fitted (${pct}%) \u2022 ${stratStr}${curPart}`;
        }

        /* ── react to trait changes ──────────────────────────── */
        model.on('change:grid_data',        draw);
        model.on('change:current_row',      draw);
        model.on('change:current_col',      draw);
        model.on('change:fitted_count',     draw);
        model.on('change:grid_rows',        draw);
        model.on('change:grid_cols',        draw);
        model.on('change:strategy',         draw);
        model.on('change:current_index_str', draw);

        /* ── cleanup tooltip when widget removed ─────────────── */
        const observer = new MutationObserver(() => {
            if (!document.body.contains(canvas)) {
                tip.remove();
                observer.disconnect();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });

        draw();
    }

    export default { render };
    """

    # ------------------------------------------------------------------ #
    # Python-side helpers                                                  #
    # ------------------------------------------------------------------ #

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fi = None
        self._on_index_changed_cb = None
        self._on_complete_cb = None

    # ------------------------------------------------------------------

    def _push_state(self):
        """Recompute the binned grid from the attached FitIndices and push
        all synced traitlets in a single batch."""
        fi = self._fi
        if fi is None:
            return
        grid, cur_row, cur_col = fi._compute_grid()
        strat = fi.strategy if isinstance(fi.strategy, str) else "custom"
        cur_str = (
            str(fi.current_index)
            if fi.current_index is not None and fi.current_index != ()
            else ""
        )
        with self.hold_trait_notifications():
            self.grid_data = grid.astype(np.float32).tobytes()
            self.grid_rows = int(grid.shape[0])
            self.grid_cols = int(grid.shape[1])
            self.current_row = int(cur_row)
            self.current_col = int(cur_col)
            self.fitted_count = fi.fitted_count
            self.total_count = fi.total_count
            self.strategy = strat
            self.current_index_str = cur_str

    # ------------------------------------------------------------------

    @classmethod
    def from_fit_indices(cls, fit_indices):
        """Create a widget bound to *fit_indices* and wire up its events.

        Parameters
        ----------
        fit_indices : hyperspy.fit_indices.FitIndices

        Returns
        -------
        FitIndicesWidget
        """
        widget = cls()
        widget._fi = fit_indices

        # Initial render
        widget._push_state()

        # Callbacks
        def _on_index_changed(obj, index):
            widget._push_state()

        def _on_complete(obj):
            widget._push_state()
            # Disconnect so the index_changed callback doesn't linger
            try:
                obj.events.index_changed.disconnect(_on_index_changed)
            except Exception:
                pass

        widget._on_index_changed_cb = _on_index_changed
        widget._on_complete_cb = _on_complete

        fit_indices.events.index_changed.connect(_on_index_changed, ["obj", "index"])
        fit_indices.events.fitting_complete.connect(_on_complete, ["obj"])

        return widget

    # ------------------------------------------------------------------

    def disconnect(self):
        """Manually disconnect event callbacks (called on abort / close)."""
        fi = self._fi
        if fi is None:
            return
        if self._on_index_changed_cb is not None:
            try:
                fi.events.index_changed.disconnect(self._on_index_changed_cb)
            except Exception:
                pass
        if self._on_complete_cb is not None:
            try:
                fi.events.fitting_complete.disconnect(self._on_complete_cb)
            except Exception:
                pass
