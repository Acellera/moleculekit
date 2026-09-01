# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
"""Registry of viewer backends.

A backend is anything that can show a Molecule: an application embedding
moleculekit, a notebook widget, a browser viewer driven from Pyodide.
Registering one makes its name usable as ``mol.view(viewer=...)``, and lets a
viewer that is already on screen follow the changes made to ``mol.reps``
afterwards, rather than having to rebuild its scene or replace moleculekit's
own methods to notice them.

A backend is duck-typed. Only ``view`` is required:

- ``view(mol, name=None)``: show the molecule. Called by
  :meth:`moleculekit.molecule.Molecule.view`.
- ``representation_added(mol, index, params)``: a representation was appended.
- ``representation_updated(mol, index, params)``: one was changed in place.
- ``representation_removed(mol, index)``: one was removed, or all of them when
  ``index`` is None.

``params`` is the same translated description the Mol* scene is built from
(``type``, ``color``, ``opacity``, ``size_factor``, ``label_fields``, ``sel``
and the rest), or None when the selection matched no atoms. A backend that
leaves a method out simply does not hear about that kind of change.

Examples
--------
>>> from moleculekit.viewer.backends import register_viewer
>>> class Printer:
...     def view(self, mol, name=None):
...         print(f"showing {mol.numAtoms} atoms")
...     def representation_added(self, mol, index, params):
...         print(f"rep {index} is a {params['type']}")
>>> register_viewer("printer", Printer())          # doctest: +SKIP
>>> mol.reps.add("protein", "NewCartoon")          # doctest: +SKIP
rep 0 is a cartoon
>>> mol.view(viewer="printer")                     # doctest: +SKIP
showing 1701 atoms
"""

_backends = {}


def register_viewer(name: str, backend):
    """Register a viewer backend under a name.

    Parameters
    ----------
    name : str
        The name to pass as ``mol.view(viewer=name)``. Case insensitive.
    backend : object
        An object with a ``view`` method, and optionally the representation
        callbacks described in this module's docstring.

    Raises
    ------
    ValueError
        If the backend has no ``view`` method, or the name is one of the
        built-in viewers, which are not replaceable this way.
    """
    key = name.lower()
    if key in ("vmd", "pymol", "ngl", "webgl", "molstar"):
        raise ValueError(f"{name!r} is a built-in viewer and cannot be replaced.")
    if not callable(getattr(backend, "view", None)):
        raise ValueError(
            f"A viewer backend needs a view(mol, name=None) method, "
            f"{type(backend).__name__} has none."
        )
    _backends[key] = backend


def unregister_viewer(name: str):
    """Forget a registered backend.

    Parameters
    ----------
    name : str
        The name it was registered under.
    """
    _backends.pop(name.lower(), None)


def get_viewer(name: str):
    """Look a backend up by name.

    Parameters
    ----------
    name : str
        The name it was registered under. Case insensitive.

    Returns
    -------
    backend : object or None
        The backend, or None if nothing is registered under that name.
    """
    return _backends.get(name.lower())


def default_viewer():
    """The registered backend to use when nothing else chose one.

    A live viewer being registered is a strong signal that it is where a
    molecule should go, so a single registered backend becomes the default.
    Several are ambiguous, and then the usual resolution order applies.

    Returns
    -------
    name : str or None
        The name of the only registered backend, or None.
    """
    return next(iter(_backends)) if len(_backends) == 1 else None


def notify(event: str, mol, index, params):
    """Tell every backend that cares about a change to ``mol.reps``.

    Parameters
    ----------
    event : str
        ``added``, ``updated`` or ``removed``.
    mol : Molecule
        The molecule whose representations changed. A backend showing several
        molecules uses this to tell which one it was, and ignores molecules it
        is not showing.
    index : int or None
        Which representation, or None when all of them were removed.
    params : callable
        Returns the translated representation, called only if a backend is
        listening, since translating resolves the selection against the
        molecule and a scene being built has no need of it.
    """
    if not _backends:
        return
    handlers = [
        handler
        for backend in _backends.values()
        if (handler := getattr(backend, f"representation_{event}", None)) is not None
    ]
    if not handlers:
        return
    if event == "removed":
        for handler in handlers:
            handler(mol, index)
        return
    translated = params()
    for handler in handlers:
        handler(mol, index, translated)
