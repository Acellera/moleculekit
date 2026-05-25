# How to view a molecule

## Goal

Open a viewer on a {py:class}`~moleculekit.molecule.Molecule` object — overview of all supported backends.

## Minimal example

```python
from moleculekit.molecule import Molecule

mol = Molecule("3PTB")

# Auto-detect the best available viewer
mol.view()
```

## Parameters that matter

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `viewer` | `str` | auto | `"molstar"`, `"vmd"`, or `"pymol"` |
| `name` | `str` or `None` | `None` | Display label for the molecule in the viewer |
| `sel` | `str` | `None` (all) | Atom selection to pass to the viewer |
| `style` | `str` or `None` | `None` | Representation style hint for compatible viewers |

## Common variations

```python
# Browser-based viewer (built in, no external install required)
mol.view(viewer="molstar")
```

```python
# VMD desktop viewer
mol.view(viewer="vmd")
```

```python
# PyMOL desktop viewer
mol.view(viewer="pymol")
```

## Viewer auto-selection order

When `viewer=` is not given, moleculekit picks a backend in this order:

1. `MOLECULEKIT_VIEWER` environment variable (e.g. `export MOLECULEKIT_VIEWER=molstar`).
2. `moleculekit.config["viewer"]` runtime setting.
3. VMD found on `$PATH`.
4. PyMOL found on `$PATH`.
5. Molstar fallback (always available).

## Molstar backend behaviour

- The first call to `mol.view(viewer="molstar")` starts an HTTP server on the first free port from 8765 onward and opens a browser tab. Subsequent calls (same or a different `Molecule`) push to the existing tab.
- Updates propagate automatically: a daemon thread polls registered molecules every ~0.5 s and pushes changes via Server-Sent Events. There is no need to call `.view()` again after mutating a molecule.
- Topology changes (mutation, filtering, bond edits) trigger a full rebuild in the viewer. Coordinate-only changes update in place without resetting the camera.
- Multiple registered molecules appear in the slot sidebar at the bottom-right of the viewer with an eye icon (show/hide) and an × button (remove). Closing a slot removes it from the viewer; the Python `Molecule` object is unaffected.
- Bond orders and per-atom formal charges are always rendered.

## Gotchas

- VMD and PyMOL must be on `$PATH` for their respective backends to work.
- The molstar backend opens a tab in the default web browser; headless environments cannot open a browser tab automatically.
- Restarting the Python process invalidates the existing molstar browser tab — the page will show a "Server restarted — refresh page" banner. Reload the tab to connect to the new server.
- Trajectories are sent to molstar in full. For very large trajectories (>~100 MB of coordinates) consider striding the frames yourself before calling `.view()`.
