# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from typing import TYPE_CHECKING
import re
import numpy as np
import logging

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule

logger = logging.getLogger(__name__)


def _element_symbol(element: str, name: str) -> str:
    """Return a title-cased element symbol (``CL`` -> ``Cl``). Falls back to the
    leading alphabetic characters of ``name`` when ``element`` is empty."""
    el = str(element)
    if not el:
        match = re.match(r"[A-Za-z]+", str(name))
        el = match.group(0) if match else "X"
    el = el[:2]
    return el[0].upper() + el[1:].lower()


def canonicalizeAtomNames(
    mol: "Molecule",
    sel: str | np.ndarray = "all",
    _logger: bool = True,
) -> "Molecule":
    """Give atoms unique ``<Element><index>`` names (``C1``, ``C2``, ``N1``, ...) within each residue.

    Only residues whose atom names are not already unique (or contain an empty
    name) are renamed; residues that already have unique names are left
    untouched. Canonical protein/nucleic residues therefore pass through
    unchanged, since their atom names (``N``, ``CA``, ``CB``, ...) are already
    unique and are force-field-meaningful, which makes this safe to run on a
    whole system. The naming counter restarts per residue, so identical residues
    (e.g. multiple copies of the same ligand) receive the same names.

    This is typically needed after loading a molecule whose atoms are named only
    by element symbol (as an SDF produces), where every carbon is called ``C``,
    every nitrogen ``N`` and so on, since non-unique names break topology
    generation.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The molecule whose atom names to canonicalize. Not modified in place.
    sel : str or np.ndarray
        An atom selection string, a boolean mask, or an integer index array (see
        :meth:`Molecule.atomselect <moleculekit.molecule.Molecule.atomselect>`).
        Only residues with at least one selected atom are considered; the rest
        are left untouched. Default is "all".

    Returns
    -------
    newmol : :class:`Molecule <moleculekit.molecule.Molecule>`
        A copy of the input molecule with canonicalized atom names.

    Examples
    --------
    >>> mol = Molecule("ligand.sdf")
    >>> mol = canonicalizeAtomNames(mol)
    """
    from moleculekit.util import sequenceID

    mol = mol.copy()

    selmask = mol.atomselect(sel)
    residues = sequenceID((mol.resid, mol.insertion, mol.chain, mol.segid))

    for r in np.unique(residues):
        idx = np.where(residues == r)[0]
        if not selmask[idx].any():
            continue
        names = mol.name[idx]
        # Already unique and fully named: nothing to fix, and the existing names
        # may be force-field-meaningful, so leave them alone.
        if len(np.unique(names)) == len(names) and np.all(names != ""):
            continue

        counts = {}
        for i in idx:
            el = _element_symbol(str(mol.element[i]), str(mol.name[i]))
            counts[el] = counts.get(el, 0) + 1
            newname = f"{el}{counts[el]}"
            if _logger:
                logger.debug(f"Rename atom {i}: {mol.name[i]:>4s} --> {newname}")
            mol.name[i] = newname

    return mol
