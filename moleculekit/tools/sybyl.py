# (c) 2015-2026 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
"""SYBYL atom typing from perceived chemistry and stored bond orders.

PROPKA decides which groups of a non-canonical residue can titrate entirely
from SYBYL atom types, and it has no bond-order field to consult: an
:class:`propka.atom.Atom` carries ``bonded_atoms``, ``num_pi_elec_2_3_bonds``
and ``steric_number``, and nothing else about how its bonds are ordered. So
``propka.ligand.assign_sybyl_type`` reconstructs the chemistry from coordinates
- ring membership, planarity and aromaticity by geometry, double bonds by a
distance threshold - and most of its helpers exist only for that.

Reconstruction gets wrong what a stored bond order states outright. PROPKA's
double-bond test compares against ``MAX_C_DOUBLE_BOND`` of 1.3 A, calibrated
for carbon, so a P=O at ~1.5 A never registers and every terminal phosphate
oxygen is typed ``O.3``. Each then becomes an independent titratable site, and
since a phosphate has one ionizable proton fewer than it has terminal oxygens,
the group is over-charged by one.

This module reads the bond orders the molecule already carries and lets RDKit
perceive the rest. Ring membership, aromaticity, conjugation and hybridisation
are RDKit's answers, not re-derived here: aromaticity in particular cannot be
read off bond types, since mmCIF commonly stores aromatic rings in Kekule form
with no aromatic bond recorded at all.

Reproducing PROPKA's conventions
--------------------------------
PROPKA reads certain types as markers rather than as neutral chemistry, so this
module matches its spellings rather than any external SYBYL implementation:

- a carboxylate is emitted as an ``O.co2-`` / ``O.co2`` *pair*, one per oxygen;
- ``N.pl3`` doubles as the guanidinium and amidinium detector - PROPKA's ``C.2``
  branch looks for a carbon bearing two ``N.pl3`` nitrogens that each have a
  single heavy neighbour;
- ``P.3`` is emitted for phosphorus so that, when these types are handed to
  PROPKA, its phosphorus branch - which ends by resetting every bonded oxygen to
  ``O.3`` - returns early instead of overwriting them.

What this does and does not settle
----------------------------------
Atom types settle *group identity* - which groups PROPKA will find - and
nothing else. They cannot settle protonation, because SYBYL has no vocabulary
for it: a neutral, mono-anionic and di-anionic phosphate all type identically,
as ``O.2`` plus terminal ``O.3``. So bond orders are respected here, and the
hydrogens and formal charges an input states are not, with one exception -
``N.4`` needs the charge, and ``O.co2-`` is given to the oxygen the input
charged, since both types carry charge information themselves.

A templated residue's protonation is therefore respected one layer up rather
than here: ``preparation_propka._apply_templated_formal_charges`` pins each group's
charge from ``mol.formalcharge`` and marks it non-titratable, so PROPKA never
re-decides what the template already settled. Anything calling this module and
handing the result to PROPKA without that step will get protonation re-invented.

Only the types PROPKA consumes are produced, in group detection (``C.2``,
``Cl``, ``F``, ``N.1``, ``N.3``, ``N.4``, ``N.am``, ``N.ar``, ``N.pl3``,
``O.2``, ``O.3``, ``O.co2``, ``S.3``) and in the pi-electron tables (``C.1``,
``C.ar``, ``O.co2``). Anything else falls back to the capitalised element
symbol, as PROPKA itself does.
"""

import numpy as np

from moleculekit.util import sequenceID

_DOUBLE = "2"
_TRIPLE = "3"
_AROMATIC = "ar"
_AMIDE = "am"
# Bond types that state an order. Anything else carries no information.
_INFORMATIVE = frozenset((_DOUBLE, _TRIPLE, _AROMATIC, _AMIDE, "1"))

_HALOGENS = frozenset(("F", "Cl", "Br", "I"))


def _ordered_residues(mol):
    """Per-atom mask of the residues that state bond orders of their own.

    Most of the typing depends on them, and guessing without them is worse than
    the geometric perception it would replace, not merely less informed: with no
    double bond to find every carbonyl oxygen becomes ``O.3`` where a distance
    test correctly finds ``O.2``, an aromatic ring reads as sp3, and RDKit
    perceives no aromaticity either. So a residue with no orders gets only the
    types that do not need them.

    The question is per residue, not per molecule, because a caller templates
    the residues it has SMILES for and leaves the rest: orders recorded on one
    residue say nothing about its neighbour. Only bonds internal to a residue
    count, so a peptide bond lends its order to neither side.
    """
    ordered = np.zeros(mol.numAtoms, dtype=bool)
    if mol.bonds is None or not len(mol.bonds):
        return ordered
    if len(mol.bondtype) != len(mol.bonds):
        return ordered

    residues = sequenceID((mol.resid, mol.insertion, mol.chain, mol.segid))
    informative = np.isin(mol.bondtype.astype(str), list(_INFORMATIVE))
    internal = residues[mol.bonds[:, 0]] == residues[mol.bonds[:, 1]]
    return np.isin(residues, residues[mol.bonds[informative & internal, 0]])


def _perceive(mol):
    """RDKit molecule with rings, aromaticity, conjugation and hybridisation.

    Metal-coordination bonds are dropped first. A dative contact does not change
    the donor's hybridisation, and counting one makes a coordinating carboxylate
    or thiolate look like an atom with one bond too many - a calcium on a
    gamma-carboxyglutamate oxygen is enough to hide the carboxylate from
    detection entirely.

    Deliberately a *partial* sanitization. The full pass also validates
    valences, which real structures fail routinely - a 4-connected neutral
    nitrogen is enough - and none of that validation is needed to perceive
    rings. Atom indices are preserved, so index ``i`` here is index ``i`` in
    ``mol``.
    """
    from rdkit import Chem

    if len(mol.bondtype) == len(mol.bonds) and (mol.bondtype == "mc").any():
        mol = mol.copy()
        keep = np.where(mol.bondtype != "mc")[0]
        mol.bonds = mol.bonds[keep]
        mol.bondtype = mol.bondtype[keep]

    rmol = mol.toRDKitMol(sanitize=False, assignStereo=False, _logger=False)
    Chem.SanitizeMol(
        rmol,
        sanitizeOps=(
            Chem.SANITIZE_SYMMRINGS
            | Chem.SANITIZE_SETAROMATICITY
            | Chem.SANITIZE_SETCONJUGATION
            | Chem.SANITIZE_SETHYBRIDIZATION
        ),
    )
    return rmol


def _heavy_neighbours(atom):
    return [n for n in atom.GetNeighbors() if n.GetSymbol() != "H"]


def _double_bonded(atom, symbol=None):
    """Neighbours reached by a double bond, optionally filtered by element."""
    from rdkit import Chem

    out = []
    for bond in atom.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        other = bond.GetOtherAtom(atom)
        if symbol is None or other.GetSymbol() == symbol:
            out.append(other)
    return out


# A carboxylic acid or carboxylate: a three-connected carbon bearing a
# terminal double-bonded oxygen and a second oxygen that is either terminal
# (deprotonated, or protonated with hydrogens stripped) or hydroxyl. Matching
# rather than walking the connectivity keeps RDKit responsible for the
# chemistry; the double bond can be required because a carboxylate cannot be
# drawn validly without one - two single bonds would leave the carbon
# four-connected.
_CARBOXYLATE = "[CX3](=[OX1])[OX1,OX2H1]"


def _carboxylate_pairs(rmol, mol):
    """``{carbon: (anionic_oxygen, neutral_oxygen)}`` for every carboxylate.

    ``O.co2-`` carries a charge marker - PROPKA's own ``sybyl_charges`` table
    reads it as charged - so it is given to the oxygen the input actually
    charged, rather than to whichever the match happened to list first. Where the
    input charges neither (a carboxylic acid, or a residue nothing templated),
    the lower index takes it, so the result stays deterministic. The distinction
    is arbitrary in PROPKA itself, which labels whichever oxygen it found first.
    """
    from rdkit import Chem

    query = Chem.MolFromSmarts(_CARBOXYLATE)
    pairs = {}
    for match in rmol.GetSubstructMatches(query, uniquify=True):
        carbon, first, second = match
        anionic = [o for o in (first, second) if int(mol.formalcharge[o]) < 0]
        if len(anionic) == 1:
            other = second if anionic[0] == first else first
            pairs[carbon] = (anionic[0], other)
        else:
            pairs[carbon] = tuple(sorted((first, second)))
    return pairs


def _phosphoryl_oxygen(atom):
    """Which terminal oxygen of phosphorus ``atom`` to type ``O.2``.

    A P(V) oxo species has exactly one P=O, so its ionizable protons number one
    fewer than its terminal oxygens. That holds whether or not an order was
    recorded, which matters because geometry cannot supply it: a deposited
    phosphate is a delocalised anion whose P-O distances are equal to within
    hundredths of an Angstrom. A recorded double bond picks the oxygen,
    otherwise the lowest index does, so the answer does not depend on iteration
    order. Which one is picked is immaterial - they are equivalent.
    """
    terminal = [
        n
        for n in _heavy_neighbours(atom)
        if n.GetSymbol() == "O" and len(_heavy_neighbours(n)) == 1
    ]
    if len(terminal) < 2:
        return None
    recorded = [o.GetIdx() for o in _double_bonded(atom, "O")]
    candidates = [i for i in recorded if i in {o.GetIdx() for o in terminal}]
    return min(candidates or [o.GetIdx() for o in terminal])


# An amide nitrogen: one bonded to a carbonyl carbon. This is the shape PROPKA
# looks for too, which additionally requires the nitrogen to be non-aromatic -
# handled here by testing aromaticity first, so an aromatic nitrogen never
# reaches this.
_AMIDE_NITROGEN = "[NX3][CX3]=[OX1]"


def _amide_nitrogens(rmol):
    """Indices of nitrogens bonded to a carbonyl carbon."""
    from rdkit import Chem

    query = Chem.MolFromSmarts(_AMIDE_NITROGEN)
    return {match[0] for match in rmol.GetSubstructMatches(query, uniquify=True)}


def sybylTypes(mol, sel=None):
    """Assign SYBYL atom types from perceived chemistry and stored bond orders.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        A molecule carrying ``bonds``, ``bondtype`` and ``formalcharge``.
    sel : str or np.ndarray or None
        Atom selection, boolean mask or index array restricting which atoms are
        typed. Perception always runs on the whole molecule, so a residue is
        typed in the context of its neighbours.

    Returns
    -------
    types : dict
        ``{atom_index: sybyl_type}`` for the selected heavy atoms. Hydrogens are
        omitted: PROPKA types them ``H`` and never branches on it.

    Notes
    -----
    For a residue that records no bond orders only the phosphorus types are
    returned - the one judgement here that does not need them - and everything
    else is left to whatever the caller had. See :func:`_ordered_residues`.

    Examples
    --------
    >>> from moleculekit.molecule import Molecule
    >>> from moleculekit.tools.sybyl import sybylTypes
    >>> mol = Molecule("3ptb")
    >>> mol.templateResidueFromSmiles(
    ...     "resname BEN", "NC(=[NH2+])c1ccccc1", addHs=True
    ... )
    >>> types = sybylTypes(mol, "resname BEN")
    >>> sorted(set(types.values()))
    ['C.2', 'C.ar', 'N.pl3']
    """
    from rdkit import Chem

    ordered = _ordered_residues(mol)
    rmol = _perceive(mol)

    selected = (
        np.ones(mol.numAtoms, dtype=bool) if sel is None else mol.atomselect(sel)
    )
    indices = [int(i) for i in np.where(selected)[0]]
    types = {}

    # Group-level first: a carboxylate is a property of the pair, and one
    # terminal oxygen per phosphorus is the P=O rather than an ionizable site.
    # Phosphorus needs no bond orders: exactly one terminal oxygen is the P=O
    # however the input drew it, so this runs even for an untyped structure.
    for idx in indices:
        atom = rmol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "P":
            continue
        types[idx] = "P.3"
        oxide = _phosphoryl_oxygen(atom)
        if oxide is not None:
            types[oxide] = "O.2"

    # Everything from here reads the bond orders, so drop the residues that do
    # not state any and leave those to PROPKA's own geometric perception.
    indices = [idx for idx in indices if ordered[idx]]

    carboxylates = _carboxylate_pairs(rmol, mol)
    amide_nitrogens = _amide_nitrogens(rmol)
    for idx in indices:
        if idx in carboxylates:
            first, second = carboxylates[idx]
            types[idx] = "C.2"
            types[first] = "O.co2-"
            types[second] = "O.co2"

    for idx in indices:
        atom = rmol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol == "H" or idx in types:
            continue

        if symbol in _HALOGENS:
            types[idx] = symbol
            continue

        aromatic = atom.GetIsAromatic()
        hybridisation = atom.GetHybridization()
        sp = hybridisation == Chem.HybridizationType.SP
        sp2 = hybridisation == Chem.HybridizationType.SP2

        if symbol == "C":
            if aromatic:
                types[idx] = "C.ar"
            elif sp:
                types[idx] = "C.1"
            elif sp2:
                types[idx] = "C.2"
            else:
                types[idx] = "C.3"

        elif symbol == "N":
            if aromatic:
                types[idx] = "N.ar"
            elif sp:
                types[idx] = "N.1"
            elif int(mol.formalcharge[idx]) > 0 and atom.GetDegree() == 4:
                types[idx] = "N.4"
            elif idx in amide_nitrogens:
                types[idx] = "N.am"
            elif sp2:
                # conjugated: PROPKA's guanidinium / amidinium marker
                types[idx] = "N.pl3"
            else:
                types[idx] = "N.3"

        elif symbol == "O":
            types[idx] = "O.2" if _double_bonded(atom) else "O.3"

        elif symbol == "S":
            oxo = len(_double_bonded(atom, "O"))
            types[idx] = "S.o2" if oxo >= 2 else "S.o" if oxo == 1 else "S.3"

        else:
            types[idx] = symbol

    return {i: t for i, t in types.items() if selected[i]}
