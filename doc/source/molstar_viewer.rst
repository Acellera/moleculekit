Browser viewer (``viewer="molstar"``)
=====================================

Goal
----

Visualize a :class:`~moleculekit.molecule.Molecule` in a `Mol* <https://molstar.org/>`_
viewer running in your default browser, driven by an HTTP server inside your
Python process. The viewer auto-updates as you mutate the ``Molecule`` in
place — no need to call ``.view()`` again after every change. Bond orders
and per-atom formal charges are always rendered.

Example
-------

.. code-block:: python

    from moleculekit.molecule import Molecule

    mol = Molecule("1awf")
    mol.view(viewer="molstar", name="receptor")

    # Edit the molecule — the viewer updates within ~0.5 s.
    mol.filter("name CA")

    # Coordinate-only changes don't trigger a rebuild (camera stays put).
    import numpy as np
    mol.coords[:, :, 0] += np.random.default_rng(0).normal(scale=0.05, size=mol.coords[:, :, 0].shape).astype(np.float32)

You can register more than one molecule; each one appears in the slot
sidebar on the top-left of the viewer with an eye icon (show/hide) and an
× button (remove from viewer):

.. code-block:: python

    ligand = Molecule("ligand.sdf")
    ligand.view(viewer="molstar", name="ligand")

Parameters
----------

``viewer="molstar"``
    Selects the browser backend. ``molstar`` is also the automatic fallback
    when neither ``vmd`` nor ``pymol`` is found in ``PATH``.

``name`` (optional)
    Display label for the molecule in the viewer's slot sidebar. Sets
    ``mol.viewname``. If omitted, the existing ``mol.viewname`` (or an
    auto-generated ``mol_N``) is used.

Choosing the viewer
-------------------

When the ``viewer=`` argument is not given, the resolution order is:

1. ``MOLECULEKIT_VIEWER`` environment variable (e.g.
   ``export MOLECULEKIT_VIEWER=molstar``)
2. ``moleculekit.config["viewer"]``
3. Auto-detection of ``vmd`` then ``pymol`` in ``PATH``
4. Fallback to ``molstar``

Behavior
--------

- The first call to ``.view(viewer="molstar")`` starts an HTTP server on
  the first free port from 8765 onward and opens a browser tab. Subsequent
  calls (same or different ``Molecule``) push to the existing tab.
- Updates propagate automatically: a daemon thread polls registered
  molecules every ~0.5 s and pushes changes via Server-Sent Events. There
  is no need to call ``.view()`` again after mutating a molecule.
- Topology changes (mutation, filtering, bond edits) trigger a full
  rebuild in the viewer. Coordinate-only changes update in place without
  resetting the camera.
- Closing a slot via the × button removes it from the viewer; the Python
  ``Molecule`` object is unaffected.

Gotchas
-------

- Requires a default web browser on the machine running Python; headless
  environments cannot open the tab automatically.
- Restarting the Python process invalidates the existing browser tab —
  the page will show a "Server restarted — refresh page" banner. Reload
  the tab to connect to the new server.
- Trajectories are sent in full. For very large trajectories (>~100 MB of
  coordinates) consider striding the frames yourself before calling
  ``.view()``.
