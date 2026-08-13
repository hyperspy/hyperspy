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

"""Sphinx-Gallery scraper for HyperSpy's ``anyplotlib`` plotting backend.

Why not use :class:`anyplotlib.sphinx_anywidget.AnywidgetScraper` directly?
Three reasons, all of them consequences of how HyperSpy examples are written:

1. **HyperSpy examples never bind the figure to a name.**  ``s.plot()``
   returns ``None`` and stashes the figure on the private ``s._plot``
   attribute, so the upstream scraper — which scans ``example_globals`` for
   an ``AnyWidget`` — finds nothing.  Instead we track *every* ``AnyWidget``
   instantiated while the example runs, which is exactly what the Pyodide
   bridge (``anywidget_bridge.js`` step 9) does at page-load time.  Using the
   same rule on both sides is what keeps the build-time ``fig_index`` aligned
   with the browser-side widget order.

2. **Sphinx-Gallery calls scrapers once per code block.**  The upstream
   scraper re-emits whichever widget it finds on every subsequent block, so a
   five-block example would show the same figure five times.  We emit each
   widget exactly once, in the block that created it.

3. **One Chromium per thumbnail is too slow.**  Upstream launches (and tears
   down) a browser for every figure.  The gallery has ~60 of them, so we keep
   a single browser alive for the whole build instead.

Everything else — the standalone HTML, the iframe wrapper, the lightning
badge, the ``# Interactive`` sentinel — is reused from upstream so the two
stay in sync.
"""

from __future__ import annotations

import atexit
import json as _json
import re
from html import escape as _html_escape
from pathlib import Path

from anyplotlib.sphinx_anywidget._scraper import (
    _PYODIDE_MOCK_PACKAGES_RE,
    _PYODIDE_PACKAGES_RE,
    MAX_DOC_WIDTH,
    _iframe_html,
)
from sphinx.util import logging

# Sentinel that promotes a figure from a static snapshot to a live one.
# Upstream anchors this to the end of a *line*; we keep the same spelling so
# examples are portable between the two scrapers.
_INTERACTIVE_RE = re.compile(r"#\s*interactive\s*\.?\s*$", re.IGNORECASE | re.MULTILINE)

#: Distributions micropip must not try to download when it resolves the
#: hyperspy wheel's requirements.  ``anywidget_bridge.js`` already puts working
#: stand-ins for all of these in ``sys.modules`` before the install, but that
#: happens too late for dependency *resolution*, which still walks the
#: requirement list and would fail on e.g. ``traits`` (a C extension with no
#: pure-Python wheel).  Registering them as micropip mock packages skips them.
#:
#: Examples can extend this per-file with ``_PYODIDE_MOCK_PACKAGES = [...]``.
#: ``prettytable`` is deliberately absent: it is a small pure-Python wheel that
#: micropip can install, and HyperSpy's model output needs the real class, not
#: upstream's stub.  See _pyodide_rsciio_supplement.py.
DEFAULT_MOCK_PACKAGES = (
    "traits",
    "rosettasciio",
    "natsort",
)

#: Pyodide-distribution packages to ``loadPackage`` before an example runs.
#: Examples can extend this with ``_PYODIDE_PACKAGES = [...]``.
DEFAULT_PACKAGES = ()

#: Requirements micropip should fetch from PyPI, for examples that need a
#: package Pyodide does not ship — HyperSpy extensions such as holospy, say.
#: Examples declare these with ``_PYODIDE_MICROPIP = [...]``; the bridge patch
#: below collects every declaration on a page and installs them together.
DEFAULT_MICROPIP = ()

#: Matches ``_PYODIDE_MICROPIP = [...]`` in an example's source.  The two
#: sibling patterns come from anyplotlib; this one is ours.
_PYODIDE_MICROPIP_RE = re.compile(
    r"^_PYODIDE_MICROPIP\s*=\s*(\[[^\]]*\])", re.MULTILINE
)


# ---------------------------------------------------------------------------
# Widget creation tracking
# ---------------------------------------------------------------------------

#: Every ``AnyWidget`` built since the last :func:`reset_widget_tracking`, in
#: construction order.  Populated by the ``__init__`` patch below.
_CREATED: list = []

#: How many entries of :data:`_CREATED` the scraper has already turned into
#: RST.  A cursor rather than a set of ``id()`` values: ``_CREATED`` is cleared
#: between examples, so the widgets it held become collectable and CPython is
#: free to hand their addresses to the next example's widgets — which would
#: make a fresh figure look like one already emitted and silently drop it.
#: Within an example ``_CREATED`` only ever grows, so an index is exact.
_EMITTED = 0

_patched = False


def _install_tracker():
    """Patch ``AnyWidget.__init__`` to record instances in creation order."""
    global _patched
    if _patched:
        return
    try:
        import anywidget
    except ImportError:  # pragma: no cover - anyplotlib depends on anywidget
        return

    original = anywidget.AnyWidget.__init__

    def tracked(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _CREATED.append(self)

    tracked._hyperspy_wrapped = original  # noqa: SLF001 - for introspection
    anywidget.AnyWidget.__init__ = tracked
    _patched = True


def reset_widget_tracking(gallery_conf=None, fname=None, when=None):
    """Sphinx-Gallery ``reset_modules`` hook — run before each example.

    Two jobs:

    * Clear the creation log so figure indices restart at 0 for every example,
      matching the browser, where each example source is executed into a fresh
      ``_CREATED_WIDS`` list.
    * Put the plotting backend back to matplotlib.  ``Plot.backend`` is global
      and the whole gallery runs in one process, so without this the first
      example to opt into anyplotlib would silently switch every later example
      too — including the handful that drive matplotlib objects directly and
      raise ``BackendCapabilityError`` under any other backend.  Resetting also
      makes each script behave the same way here as it does standalone: you
      get matplotlib unless the script asks for something else.
    """
    global _EMITTED

    _install_tracker()
    _CREATED.clear()
    _EMITTED = 0

    from hyperspy.defaults_parser import preferences

    if preferences.Plot.backend != "matplotlib":
        preferences.Plot.backend = "matplotlib"


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------


class _BrowserPool:
    """Lazily-started, process-wide headless Chromium reused for thumbnails."""

    def __init__(self):
        self._pw = None
        self._browser = None

    def page(self):
        if self._browser is None:
            from playwright.sync_api import sync_playwright

            self._pw = sync_playwright().start()
            try:
                self._browser = self._launch()
            except Exception:
                # CI has the playwright package (doc extra) but not the
                # browser binary, and the reusable docs workflow has no hook
                # for `playwright install`. Fetch Chromium once and retry;
                # the caller already treats any remaining failure as "no
                # thumbnail" rather than a build error.
                import subprocess
                import sys

                subprocess.run(
                    [sys.executable, "-m", "playwright", "install", "chromium"],
                    check=True,
                )
                self._browser = self._launch()
            atexit.register(self.close)
        return self._browser.new_page()

    def _launch(self):
        return self._pw.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"]
        )

    def close(self):
        if self._browser is not None:
            try:
                self._browser.close()
            finally:
                self._browser = None
        if self._pw is not None:
            try:
                self._pw.stop()
            finally:
                self._pw = None


_POOL = _BrowserPool()


def _thumbnail_png(widget, html_path: Path) -> bytes:
    """Screenshot *widget*'s standalone page as the gallery thumbnail.

    Rendered in the *light* colour scheme, unlike upstream's dark default:
    these thumbnails sit in a grid next to matplotlib-rendered ones and above
    the live iframe of the same figure, both of which follow the page theme,
    which is light by default. A dark thumbnail made the same figure look like
    two different plots.
    """
    from anyplotlib.sphinx_anywidget._repr_utils import build_standalone_html

    html = build_standalone_html(widget, resizable=False)
    html = html.replace(
        "renderFn({ model, el });",
        "renderFn({ model, el }); window._aplReady = true;",
    )
    html = html.replace("background: transparent;", "background: #ffffff;")
    shot_path = html_path.with_name(html_path.stem + "__thumb.html")
    shot_path.write_text(html, encoding="utf-8")

    page = _POOL.page()
    try:
        page.emulate_media(color_scheme="light")
        page.goto(shot_path.as_uri())
        page.wait_for_function("() => window._aplReady === true", timeout=30_000)
        page.evaluate(
            "() => new Promise(r =>"
            " requestAnimationFrame(() => requestAnimationFrame(r)))"
        )
        return page.locator("#widget-root").screenshot()
    finally:
        page.close()
        shot_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


def _refresh_panels(widget):
    """Re-serialise each panel so its JSON reflects current widget geometry.

    anyplotlib pushes widget moves as targeted updates that bypass the panel's
    JSON trait. Both the static iframe and the thumbnail are built from that
    trait, so without this a positioned ROI is drawn at the geometry it had
    when it was constructed — for the Live FFT example, a zero-size box.
    """
    try:
        axes = widget.get_axes()
    except Exception:
        return
    for ax in axes:
        plot = getattr(ax, "_plot", None)
        push = getattr(plot, "_push", None)
        if callable(push):
            try:
                push()
            except Exception:  # pragma: no cover - build-time robustness
                pass


class HyperSpyAnywidgetScraper:
    """Emit every ``anyplotlib`` figure an example creates, exactly once."""

    def __init__(self):
        # src_file -> number of figures emitted so far (== next fig_index)
        self._counts: dict[str, int] = {}

    def __repr__(self) -> str:
        return "HyperSpyAnywidgetScraper()"

    def __call__(self, block, block_vars, gallery_conf):
        global _EMITTED

        pending = _CREATED[_EMITTED:]
        if not pending:
            return ""
        _EMITTED = len(_CREATED)

        block_source = block[1] if isinstance(block, (list, tuple)) else ""
        is_interactive = bool(_INTERACTIVE_RE.search(block_source))

        src_file = str(block_vars.get("src_file", ""))
        image_path_iterator = block_vars["image_path_iterator"]

        src_dir = Path(gallery_conf["src_dir"])
        widgets_dir = src_dir / "_static" / "viewer_widgets"
        widgets_dir.mkdir(parents=True, exist_ok=True)

        # Figures produced by the same block are shown side by side. In the
        # ROI examples the whole point is that dragging a region in one figure
        # redraws the other, which you cannot see if they are a screenful
        # apart. `_iframe_html`'s resize script measures its own parent, so
        # giving each figure a flex item of its own makes it scale to the
        # column it lands in, and the row re-wraps to one figure per line on
        # narrow screens.
        #
        # Cells are weighted by each figure's aspect ratio (via `flex-grow`,
        # see custom-styles.css) so every figure in a row renders at the same
        # *height*.  With equal-width cells a two-panel navigator+signal
        # figure — twice as wide as a single-panel one — was scaled to half
        # the height of its neighbours and became unreadably small.
        side_by_side = len(pending) > 1
        aspects = []
        per_figure_widths = []
        if side_by_side:
            from anyplotlib.sphinx_anywidget._repr_utils import _widget_px

            for widget in pending:
                try:
                    w_px, h_px = _widget_px(widget)
                    aspects.append(w_px / h_px if h_px else 1.0)
                except Exception:  # pragma: no cover - build-time fallback
                    aspects.append(1.0)
            total_aspect = sum(aspects) or 1.0
            row_budget = MAX_DOC_WIDTH - 14 * (len(pending) - 1)
            per_figure_widths = [
                max(1, int(row_budget * aspect / total_aspect)) for aspect in aspects
            ]

        rst_parts = []
        for widget_index, widget in enumerate(pending):
            fig_index = self._counts.get(src_file, 0)
            self._counts[src_file] = fig_index + 1

            png_path = Path(next(image_path_iterator))
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig_id = png_path.stem

            html_name = fig_id + ".html"
            html_path = widgets_dir / html_name

            _refresh_panels(widget)

            try:
                from anyplotlib.sphinx_anywidget._repr_utils import (
                    _widget_px,
                    build_standalone_html,
                )

                html_path.write_text(
                    build_standalone_html(widget, resizable=False, fig_id=fig_id),
                    encoding="utf-8",
                )
                width, height = _widget_px(widget)
            except Exception as exc:  # pragma: no cover - build-time diagnostics
                print(f"[hyperspy_anywidget] WARNING: no iframe for {fig_id}: {exc}")
                continue

            try:
                png_path.write_bytes(_thumbnail_png(widget, html_path))
            except Exception as exc:  # pragma: no cover - build-time diagnostics
                print(f"[hyperspy_anywidget] WARNING: no thumbnail for {fig_id}: {exc}")

            depth = len(png_path.parent.parent.relative_to(src_dir).parts)
            src = "../" * depth + f"_static/viewer_widgets/{html_name}"

            figure_html = _iframe_html(
                src,
                width,
                height,
                fig_id=fig_id,
                interactive=is_interactive,
                max_width=per_figure_widths[widget_index] if side_by_side else None,
            )
            if side_by_side:
                figure_html = (
                    f'<div class="hspy-figure-cell" '
                    f'style="flex-grow: {aspects[widget_index]:.4f}">'
                    + figure_html
                    + "</div>"
                )
            rst_parts.append("\n\n.. raw:: html\n\n    " + figure_html + "\n\n")

            if is_interactive:
                script = self._pyodide_block(src_file, fig_id, fig_index)
                if script:
                    rst_parts.append("\n\n.. raw:: html\n\n    " + script + "\n\n")

        if side_by_side:
            rst_parts.insert(
                0, '\n\n.. raw:: html\n\n    <div class="hspy-figure-row">\n\n'
            )
            rst_parts.append("\n\n.. raw:: html\n\n    </div>\n\n")

        return "".join(rst_parts)

    @staticmethod
    def _pyodide_block(src_file: str, fig_id: str, fig_index: int) -> str:
        """Embed the example source so the bridge can re-run it in Pyodide."""
        try:
            python_src = Path(src_file).read_text(encoding="utf-8")
        except OSError:
            return ""
        if not python_src:
            return ""

        def _list_attr(pattern, attr, defaults):
            values = list(defaults)
            match = pattern.search(python_src)
            if match:
                try:
                    import ast

                    values += [
                        v for v in ast.literal_eval(match.group(1)) if v not in values
                    ]
                except (ValueError, SyntaxError):
                    pass
            if not values:
                return ""
            return f' {attr}="{_html_escape(_json.dumps(values), quote=True)}"'

        return (
            f'<script type="text/x-python"'
            f' data-fig-id="{fig_id}"'
            f' data-fig-index="{fig_index}"'
            f' data-src-file="{Path(src_file).stem}"'
            f"{_list_attr(_PYODIDE_PACKAGES_RE, 'data-pyodide-packages', DEFAULT_PACKAGES)}"
            f"{_list_attr(_PYODIDE_MOCK_PACKAGES_RE, 'data-pyodide-mock-packages', DEFAULT_MOCK_PACKAGES)}"
            f"{_list_attr(_PYODIDE_MICROPIP_RE, 'data-pyodide-micropip', DEFAULT_MICROPIP)}"
            f' data-src="{_html_escape(_json.dumps(python_src), quote=True)}"></script>'
        )


# ---------------------------------------------------------------------------
# Temporary fixes to anyplotlib's Pyodide bridge
# ---------------------------------------------------------------------------

#: Patches applied to the copy of ``anywidget_bridge.js`` in the build output.
#:
#: These are bugs in anyplotlib's traits-to-traitlets shim that only HyperSpy
#: happens to trip over.  Each one is a *temporary* local fix: the real fix
#: belongs upstream in ``anyplotlib/sphinx_anywidget/static/``, and the entry
#: should be deleted here as soon as a release carrying it is required in
#: ``pyproject.toml``.  A patch whose ``find`` text is absent is reported as a
#: build warning rather than silently skipped, so an upstream fix (or an
#: upstream refactor that moves the code) cannot go unnoticed.
BRIDGE_PATCHES = (
    {
        "id": "range-trait-name-bounds",
        "why": (
            "traits.Range accepts *trait names* as bounds "
            "(HyperSpy: `index = t.Range('low_index', 'high_index')`) and "
            "defaults to the low bound's value.  The shim instead used the "
            "name itself as the default, so the very first `axis.index = 0` "
            "read as a change from the string 'low_index' and fired "
            "`_index_changed` before `axis.axis` existed."
        ),
        "find": """class _Range(_tr.Any):
    def __init__(self, low=None, high=None, value=None, **kw):
        super().__init__(value if value is not None else low, allow_none=True, **kw)""",
        "replace": """class _Range(_tr.Any):
    def __init__(self, low=None, high=None, value=None, **kw):
        default = value if value is not None else low
        if isinstance(default, str):
            # A trait-name bound; traits resolves it at runtime and starts at
            # the low bound's value, which is 0 for every such trait HyperSpy
            # declares.  Using the name would make the first real assignment
            # look like a change and fire observers during __init__.
            default = 0
        super().__init__(default, allow_none=True, **kw)""",
    },
    {
        "id": "traits-observation-api",
        "why": (
            "The shim stops at `traits.api`, but HyperSpy's AxesManager and "
            "spikes-removal tool import `traits.observation.api` and pass its "
            "expression objects (plus `remove=`) to `observe()`.  Splices in "
            "_pyodide_traits_supplement.py, which adds the module and makes "
            "`HasTraits.observe` traits-compatible."
        ),
        "find": "print('[sphinx_anywidget] traits shim installed')",
        "replace": lambda: (
            _SUPPLEMENT_SOURCE + "\nprint('[sphinx_anywidget] traits shim installed')"
        ),
    },
    {
        "id": "rsciio-utils-file",
        "why": (
            "The shim stubs rsciio.utils.path and rsciio.utils.rgb but not "
            "rsciio.utils.file, which hyperspy/_signals/lazy.py imports at "
            "module level.  Any example whose map() produces lazy output "
            "(Markers/from_signal.py) died with ImportError before drawing.  "
            "Splices in _pyodide_rsciio_supplement.py."
        ),
        "find": "print('[sphinx_anywidget] misc stubs installed')",
        "replace": lambda: (
            _RSCIIO_SUPPLEMENT_SOURCE
            + "\nprint('[sphinx_anywidget] misc stubs installed')"
        ),
    },
    {
        "id": "dask-synchronous-scheduler",
        "why": (
            "WASM has no threads, so dask's configured default scheduler is "
            "not one it can actually build: any HyperSpy call routed through "
            "BaseSignal.map (baseline removal, peak finding, ...) failed with "
            '"Expected one of [distributed, single-threaded, sync, '
            'synchronous]".  Pin the config to the one that works.'
        ),
        "find": "print('[sphinx_anywidget] BaseDataAxis._update_slice patched for traitlets compat')",
        "replace": (
            "try:\n"
            "    import dask as _dask\n"
            "    _dask.config.set(scheduler='synchronous')\n"
            "    print('[hyperspy] dask scheduler pinned to synchronous')\n"
            "except Exception as _exc:\n"
            "    print('[hyperspy] could not pin dask scheduler: ' + str(_exc))\n"
            "print('[sphinx_anywidget] BaseDataAxis._update_slice patched for traitlets compat')"
        ),
    },
    {
        "id": "refresh-panels-before-push",
        "why": (
            "anyplotlib pushes widget moves to the browser as targeted "
            "updates that never touch the owning panel's JSON trait, so that "
            "trait still describes the widget as it was created.  The bridge "
            "wires a figure by pushing exactly those traits, so a ROI that "
            "HyperSpy had since positioned (Live FFT: a 3.84 nm box) arrived "
            "in the browser at its zero-size creation geometry.  Re-serialise "
            "each panel first."
        ),
        "find": "        _AWI_REGISTRY[_fid]  = _w",
        "replace": (
            "        _AWI_REGISTRY[_fid]  = _w\n"
            "        try:\n"
            "            for _ax in _w.get_axes():\n"
            "                _p = getattr(_ax, '_plot', None)\n"
            "                if _p is not None and hasattr(_p, '_push'):\n"
            "                    _p._push()\n"
            "        except Exception:\n"
            "            pass"
        ),
    },
    {
        "id": "expose-pyodide",
        "why": (
            "The bridge keeps its Pyodide instance in a closure, so when an "
            "example fails in the browser the only evidence is a truncated "
            "traceback in the console.  Publishing the handle lets a "
            "Playwright probe run arbitrary Python against the live "
            "interpreter, which is the difference between guessing at shim "
            "gaps and reading them off directly."
        ),
        "find": "    return pyodide;",
        "replace": "    window._hyperspyPyodide = pyodide;\n    return pyodide;",
    },
    {
        "id": "install-anyplotlib",
        "why": (
            "The bridge installs exactly one wheel: the package named by "
            "anywidget_pyodide_package, here HyperSpy.  anyplotlib is an "
            "optional HyperSpy dependency so it is not in that wheel's "
            "requirements, yet the whole gallery plots through it.  Pull it "
            "from PyPI, pinned to the version that rendered the page so the "
            "browser cannot drift from the static snapshot beside it."
        ),
        "find": "await micropip.install(${JSON.stringify(fullWheelUrl)})",
        "replace": lambda: (
            "await micropip.install(${JSON.stringify(fullWheelUrl)})\n"
            f"await micropip.install(['anyplotlib=={_anyplotlib_version()}'])\n"
            "_extra_reqs = ${JSON.stringify(_globalMicropip)}\n"
            "if _extra_reqs:\n"
            # HyperSpy extensions require "hyperspy>=2.x", but the wheel built
            # for this page reports 0.0.0 (the sentinel version the upstream
            # wheel builder uses), so micropip would satisfy the requirement by
            # downloading a *released* HyperSpy over the development one.
            # Deps cannot simply be turned off -- micropip 0.7's deps=False is
            # broken ("attempted to install wheel before downloading it") --
            # so tell micropip's resolver that what is installed is new enough.
            "    try:\n"
            "        import importlib.metadata as _md, pathlib as _pl\n"
            "        _meta = _pl.Path(str(_md.distribution('hyperspy')._path)) / 'METADATA'\n"
            "        _text = _meta.read_text()\n"
            "        if 'Version: 0.0.0' in _text:\n"
            "            _meta.write_text(_text.replace('Version: 0.0.0',\n"
            "                                           'Version: 9999.0.0', 1))\n"
            "    except Exception:\n"
            "        pass\n"
            "    try:\n"
            "        await micropip.install(_extra_reqs)\n"
            "    except Exception as _exc:\n"
            # Never fatal: this runs during boot, so letting it raise would
            # leave every figure on the page dead, not just the one example
            # that wanted the package.
            "        print('[hyperspy] could not install ' + repr(_extra_reqs)\n"
            "              + ': ' + str(_exc))"
        ),
    },
    {
        "id": "collect-micropip-requirements",
        "why": (
            "Adds the DOM scan backing _PYODIDE_MICROPIP.  The bridge can only "
            "pull extra packages through pyodide.loadPackage, which is limited "
            "to what the Pyodide distribution ships -- no good for HyperSpy "
            "extensions on PyPI such as holospy.  Collected here, next to the "
            "existing mock-package scan, and installed by 'install-anyplotlib'."
        ),
        "find": """    const wheelUrl = _DOCS_ROOT + '_static/wheels/';""",
        "replace": """    const _globalMicropip = [];
    for (const s of document.querySelectorAll(
        'script[type="text/x-python"][data-pyodide-micropip]')) {
      try {
        const pkgs = JSON.parse(s.dataset.pyodideMicropip || '[]');
        for (const p of pkgs) if (!_globalMicropip.includes(p)) _globalMicropip.push(p);
      } catch (_) {}
    }

    const wheelUrl = _DOCS_ROOT + '_static/wheels/';""",
    },
)


def _anyplotlib_version() -> str:
    from anyplotlib import __version__

    return __version__


def _load_supplement(name: str) -> str:
    """Return a Pyodide supplement's source, checked for JS-literal safety."""
    path = Path(__file__).with_name(name)
    source = path.read_text(encoding="utf-8")
    for forbidden in ("`", "${", "\\"):
        if forbidden in source:
            raise ValueError(
                f"{path.name} contains {forbidden!r}, which would corrupt the "
                "JavaScript template literal it is embedded in."
            )
    return source


_SUPPLEMENT_SOURCE = _load_supplement("_pyodide_traits_supplement.py")
_RSCIIO_SUPPLEMENT_SOURCE = _load_supplement("_pyodide_rsciio_supplement.py")


def _patch_bridge(app, exception):
    """Rewrite the built ``anywidget_bridge.js`` with :data:`BRIDGE_PATCHES`.

    Deliberately runs even when the build raised: Sphinx-Gallery reports failing
    examples from its own ``build-finished`` handler, and one broken example
    should not leave every *working* figure on the site with an unpatched
    bridge.  The output HTML is already written by this point either way.
    """
    if app.builder.name != "html":
        return

    target = Path(app.outdir) / "_static" / "anywidget_bridge.js"
    if not target.is_file():
        return

    # Always start from the installed anyplotlib copy rather than from whatever
    # is in the output directory.  Sphinx only re-copies static assets when it
    # thinks they changed, so on an incremental build the target may already be
    # patched -- and several of these patches keep their own anchor text, so
    # re-applying them would duplicate the injected code.
    from anyplotlib.sphinx_anywidget import _STATIC_SRC

    pristine = _STATIC_SRC / "anywidget_bridge.js"
    source = pristine.read_text(encoding="utf-8")

    logger = logging.getLogger(__name__)
    patched = source
    for patch in BRIDGE_PATCHES:
        if patch["find"] not in patched:
            logger.warning(
                "hyperspy_anywidget: bridge patch %r no longer applies. Check "
                "whether anyplotlib fixed it upstream (then delete the patch) "
                "or moved the code (then update it).",
                patch["id"],
            )
            continue
        replacement = patch["replace"]
        patched = patched.replace(
            patch["find"], replacement() if callable(replacement) else replacement
        )

    if patched != target.read_text(encoding="utf-8"):
        target.write_text(patched, encoding="utf-8")


def setup(app):
    # Priority below Sphinx-Gallery's default (500) so the patch lands before
    # its summarize_failing_examples handler gets a chance to raise.
    app.connect("build-finished", _patch_bridge, priority=100)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
