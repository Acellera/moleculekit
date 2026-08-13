"""Prepare non-standard residues for pKa titration and fold the result back
into per-residue build templates.

The public entry points are :func:`capNonstandardResiduesForTitration` and
:func:`templatesFromTitration`, both dict-in / dict-out (``{key: smiles}``),
with optional file endpoints for tools that communicate over files:
``capNonstandardResiduesForTitration(outfile=...)`` writes the titration input
as a ``key,SMILES,base`` CSV, and ``templatesFromTitration`` accepts the pKa
tool's output CSV path in place of the dict. The only assumption made about
the tool (e.g. AcePka) is that it rewrites the ``SMILES`` column and echoes
every other column through, which makes the round-trip self-contained: the
echoed ``base`` column guarantees cap-stripping re-derives each anchor from
exactly the base SMILES that was titrated, with no RCSB re-fetch in between.

Each residue is built from its full base SMILES (an RCSB ligand descriptor or
a caller override), so it carries the complete chemistry even when the
deposited structure has a trimmed sidechain. Every *outgoing* inter-residue
bond is then capped with an inert stand-in for its real partner, so the pKa
tool sees a chemically sane molecule that titrates the residue's own groups in
a faithful environment. Most caps are neutral (an amide, a methyl); a
phosphodiester partner is kept as a phosphate, so an internal nucleotide
carries the real, charged backbone environment a terminal one would not:

- a backbone peptide bond gets an amide cap - acetyl on the ``N`` side,
  N-methyl on the carbonyl ``C`` side (the ACE / NME that keep a mid-chain
  backbone neutral);
- a sidechain / scaffold crosslink gets an inert cap chosen to keep the
  junction atom non-titratable: acetyl on a severed nitrogen, N-methyl on an
  amide carbonyl carbon, and for any other junction a cap reflecting the real
  partner element (a nitrogen partner gives an amide, an oxygen partner a
  hydroxyl, a phosphorus partner a phosphate, everything else a methyl). When a
  condensation crosslink (glycosidic, phosphoester, ...) left the junction atom
  fully valent in the free-form SMILES, the displaced leaving group (the
  base-SMILES atom absent from the deposited residue) is stripped first;
- a genuine free terminus or free-ligand end is left uncapped, so the pKa tool
  assigns its charge (a real C-terminus deprotonates, an N-terminus protonates).

The backbone atoms are located by their structure names (``N`` / ``C``) mapped
onto the SMILES; crosslink atoms are found by walking ``mol.bonds`` and mapped
the same way. Coordinates never affect the returned SMILES - only connectivity
does. After titration, :func:`templatesFromTitration` strips the caps back off
(the residue is a subgraph of its capped molecule) to yield one template SMILES
per residue for the builder.
"""

import csv
import json
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from moleculekit.molecule import Molecule
from moleculekit.rcsb import rcsbFetchLigandSmiles
from moleculekit.tools.nonstandard_residues import (
    ChainResidueSpec,
    CovalentLigandSpec,
    GlycanSpec,
    LigandSpec,
    PerResidueSpec,
    ScaffoldSpec,
    requiresTemplate,
    getResidueMask,
)

if TYPE_CHECKING:
    from rdkit.Chem import Mol, RWMol

logger = logging.getLogger(__name__)


def _inter_residue_crosslinks(
    mol: Molecule, spec: PerResidueSpec
) -> list[tuple[int, int]]:
    """Non-peptide inter-residue bonds touching ``spec``'s residue.

    Walks ``mol.bonds`` for bonds with exactly one endpoint inside the spec's
    residue, excluding backbone peptide ``N``-``C`` bonds (handled by the
    backbone cap path), bonds to water, and metal-coordination bonds (the
    stored ``"mc"`` bond type; a metal cannot be inert-capped, and the
    metal-coordinating donor is deprotonated when its template is built, not
    covalently occupied). Each is returned as
    ``(local_global_idx, partner_global_idx)`` with the in-residue atom first.

    Parameters
    ----------
    mol : Molecule
        The molecule the spec was detected in.
    spec : ChainResidueSpec or ScaffoldSpec or CovalentLigandSpec or LigandSpec
        The residue whose crosslinks to find.

    Returns
    -------
    crosslinks : list of tuple of int
        ``(local_atom, partner_atom)`` global indices, one per crosslink.
    """
    from moleculekit.residues import WATER_RESIDUE_NAMES

    in_res = getResidueMask(mol, spec)
    # Metal-coordination bonds are stored explicitly as the "mc" bond type
    # (readers set it from LINK / struct_conn records, keyed on either endpoint
    # being a metal). Only trust it when the parsed bond types line up with the
    # bonds; untyped inputs simply carry no coordination markers to exclude.
    bondtypes = mol.bondtype if len(mol.bondtype) == len(mol.bonds) else None
    out = []
    for i, (a, b) in enumerate(mol.bonds):
        a, b = int(a), int(b)
        ina, inb = bool(in_res[a]), bool(in_res[b])
        if ina == inb:  # both in or both out of the residue
            continue
        if bondtypes is not None and str(bondtypes[i]) == "mc":
            # Metal coordination, not a covalent crosslink. An inert cap cannot
            # stand in for a metal, and a donor atom coordinating a metal is
            # deprotonated by it, not covalently occupied, so capping would
            # neutralise a group that must stay ionizable. The metal-induced
            # deprotonation is applied later when the template is built (see
            # rdkittools boundary_metal_order). The "mc" type already covers a
            # metal inside a cofactor (e.g. Fe in a HEM coordinating a Tyr-OH).
            continue
        local, partner = (a, b) if ina else (b, a)
        if str(mol.resname[partner]) in WATER_RESIDUE_NAMES:
            continue
        if {str(mol.name[local]), str(mol.name[partner])} == {"N", "C"}:
            continue  # peptide backbone bond: handled by the backbone path
        out.append((local, partner))
    return out


def _attach_ace_cap(rw, n_idx: int) -> list[int]:
    """Amide-bond an acetyl cap (CH3-C(=O)-) onto RDKit atom ``n_idx``.

    Parameters
    ----------
    rw : rdkit.Chem.RWMol
        Editable RDKit molecule to add the cap atoms to, mutated in place.
    n_idx : int
        Index of the backbone nitrogen (the residue's ``N`` atom) to cap.

    Returns
    -------
    cap_atoms : list of int
        Indices of the three atoms this cap added, so a caller reconstructing
        the bare residue (see :func:`_uncapped_residue_smiles`) can strip them.
    """
    from rdkit import Chem

    c = rw.AddAtom(Chem.Atom(6))
    o = rw.AddAtom(Chem.Atom(8))
    ch3 = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(n_idx, c, Chem.BondType.SINGLE)
    rw.AddBond(c, o, Chem.BondType.DOUBLE)
    rw.AddBond(c, ch3, Chem.BondType.SINGLE)
    return [c, o, ch3]


def _attach_nme_cap(rw, c_idx: int) -> list[int]:
    """Turn the backbone carboxyl on RDKit atom ``c_idx`` into an N-methyl
    amide (-C(=O)-NH-CH3) - the peptide-bonded C side of a mid-chain or
    N-terminal residue.

    The residue is built from its full free-amino-acid SMILES (see
    :func:`_isolated_residue_rdkit`), so ``c_idx`` carries a complete
    carboxyl: a doubly-bonded oxygen and a singly-bonded hydroxyl oxygen. The
    hydroxyl oxygen is retyped to the amide nitrogen and a methyl is attached
    to it, so the carboxyl becomes the peptide-context amide without leaving a
    spurious extra oxygen behind. If ``c_idx`` has no singly-bonded oxygen (it
    is already a bare carbonyl), a fresh amide nitrogen is added instead.

    Parameters
    ----------
    rw : rdkit.Chem.RWMol
        Editable RDKit molecule, mutated in place.
    c_idx : int
        Index of the backbone carbonyl carbon (the residue's ``C`` atom).

    Returns
    -------
    cap_atoms : list of int
        Indices of the amide nitrogen and its methyl, so a caller
        reconstructing the bare residue can strip them back to a one-oxygen
        carbonyl.
    """
    from rdkit import Chem

    catom = rw.GetAtomWithIdx(c_idx)
    hydroxyl = None
    for bond in catom.GetBonds():
        other = bond.GetOtherAtom(catom)
        if other.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.SINGLE:
            hydroxyl = other.GetIdx()
            break
    if hydroxyl is not None:
        n = hydroxyl
        namt = rw.GetAtomWithIdx(n)
        namt.SetAtomicNum(7)
        namt.SetFormalCharge(0)
        namt.SetNumExplicitHs(0)
        namt.SetNoImplicit(False)
    else:
        n = rw.AddAtom(Chem.Atom(7))
        rw.AddBond(c_idx, n, Chem.BondType.SINGLE)
    ch3 = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(n, ch3, Chem.BondType.SINGLE)
    return [n, ch3]


def _attach_methyl(rw, atom_idx: int) -> list[int]:
    """Bond a plain methyl (-CH3) onto RDKit atom ``atom_idx``.

    The inert cap for a severed crosslink whose local atom is not an amide
    nitrogen or carbonyl carbon: a methyl adds no ionizable proton, so the
    junction atom stays non-titratable.

    Parameters
    ----------
    rw : rdkit.Chem.RWMol
        Editable molecule, mutated in place.
    atom_idx : int
        Index of the atom to cap.

    Returns
    -------
    cap_atoms : list of int
        The single added methyl carbon index.
    """
    from rdkit import Chem

    ch3 = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(atom_idx, ch3, Chem.BondType.SINGLE)
    return [ch3]


def _attach_amine_amide(rw, atom_idx: int) -> list[int]:
    """Bond a nitrogen onto RDKit atom ``atom_idx`` and acetylate it.

    The inert cap for a severed crosslink whose partner is a nitrogen (an
    N-glycosidic bond, an isopeptide seen from the carbon side): the junction
    becomes an amide (``-NH-C(=O)CH3``) rather than a free amine, so a
    downstream pKa tool sees a non-titratable nitrogen, not an ionizable one.

    Parameters
    ----------
    rw : rdkit.Chem.RWMol
        Editable molecule, mutated in place.
    atom_idx : int
        Index of the atom to cap.

    Returns
    -------
    cap_atoms : list of int
        The added nitrogen and the three acetyl atoms.
    """
    from rdkit import Chem

    n = rw.AddAtom(Chem.Atom(7))
    rw.AddBond(atom_idx, n, Chem.BondType.SINGLE)
    return [n] + _attach_ace_cap(rw, n)


def _attach_hydroxyl(rw, atom_idx: int) -> list[int]:
    """Bond a hydroxyl oxygen (-OH) onto RDKit atom ``atom_idx``.

    The inert cap for a severed crosslink whose partner is an oxygen: a
    phosphate becomes a terminal ``-OH`` (leaving the phosphate itself
    correctly ionizable), a carbon becomes a non-titratable alcohol.

    Parameters
    ----------
    rw : rdkit.Chem.RWMol
        Editable molecule, mutated in place.
    atom_idx : int
        Index of the atom to cap.

    Returns
    -------
    cap_atoms : list of int
        The single added oxygen index.
    """
    from rdkit import Chem

    o = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(atom_idx, o, Chem.BondType.SINGLE)
    return [o]


def _attach_phosphate(rw, atom_idx: int) -> list[int]:
    """Bond a phosphate (``-P(=O)(OH)OH``) onto RDKit atom ``atom_idx``.

    The inert cap for a severed crosslink whose partner is a phosphorus (a
    phosphodiester / phosphoester backbone): the junction becomes a phosphate
    reflecting the real partner element, rather than a methyl that would
    mislabel a phosphorus bond as a carbon one.

    Parameters
    ----------
    rw : rdkit.Chem.RWMol
        Editable molecule, mutated in place.
    atom_idx : int
        Index of the atom to cap.

    Returns
    -------
    cap_atoms : list of int
        The added phosphorus and its three oxygens.
    """
    from rdkit import Chem

    p = rw.AddAtom(Chem.Atom(15))
    od = rw.AddAtom(Chem.Atom(8))
    o1 = rw.AddAtom(Chem.Atom(8))
    o2 = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(atom_idx, p, Chem.BondType.SINGLE)
    rw.AddBond(p, od, Chem.BondType.DOUBLE)
    rw.AddBond(p, o1, Chem.BondType.SINGLE)
    rw.AddBond(p, o2, Chem.BondType.SINGLE)
    return [p, od, o1, o2]


def _saturated_leaving_atom(rw, smi_local: int, mapped: set) -> "int | None":
    """Index of the leaving-group atom to strip from ``smi_local``, or None.

    A condensation crosslink (glycosidic, phosphoester, ...) forms by displacing
    a leaving group from the crosslink atom, but the residue is built from its
    free-form base SMILES, which still carries that group - often leaving the
    atom fully valent, so attaching an inert cap would exceed its valence. When
    one more bond would over-valence the atom, the leaving group is identified
    from the structure: it is the terminal, single-bonded neighbour that is
    absent from the deposited residue (its SMILES atom index is not in
    ``mapped``, the structure-to-SMILES atom map). Its index is returned so the
    caller can remove it before capping.

    Returns None when the atom still has a free valence (the crosslink displaced
    only a hydrogen, e.g. a serine ``OG``) or when no such leaving atom is found
    (the over-valence is then surfaced upstream as the uncappable fallback).

    Parameters
    ----------
    rw : rdkit.Chem.RWMol
        The residue built from its base SMILES.
    smi_local : int
        Index in ``rw`` of the in-residue crosslink atom.
    mapped : set of int
        The ``rw`` atom indices that a structure residue atom maps onto (the
        values of ``_isolated_residue_rdkit``'s ``res_to_smi``).

    Returns
    -------
    leaving : int or None
        Index in ``rw`` of the leaving-group atom to strip, or None.
    """
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    atom = rw.GetAtomWithIdx(smi_local)
    explicit_valence = (
        sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()) + atom.GetNumExplicitHs()
    )
    if explicit_valence + 1 <= max(pt.GetValenceList(atom.GetAtomicNum())):
        return None
    for nb in atom.GetNeighbors():
        bond = rw.GetBondBetweenAtoms(smi_local, nb.GetIdx())
        if (
            nb.GetIdx() not in mapped
            and nb.GetDegree() == 1
            and bond.GetBondType() == Chem.BondType.SINGLE
        ):
            return nb.GetIdx()
    return None


def _is_carbonyl_carbon(rw, atom_idx: int) -> bool:
    """Whether RDKit atom ``atom_idx`` in ``rw`` is a carbon double-bonded to an
    oxygen (a carbonyl carbon). Read off the base-SMILES-derived molecule, whose
    bond orders are definitive (structure inputs may lack them)."""
    from rdkit import Chem

    atom = rw.GetAtomWithIdx(atom_idx)
    return atom.GetSymbol() == "C" and any(
        b.GetBondType() == Chem.BondType.DOUBLE
        and b.GetOtherAtom(atom).GetSymbol() == "O"
        for b in atom.GetBonds()
    )


def _classify_junction(
    mol: Molecule, local_idx: int, partner_idx: int, rw, smi_local_idx: int
) -> str:
    """Classify an inter-residue bond so it can be inert-capped.

    The cap must leave the local (in-residue) atom non-titratable. A severed
    nitrogen must become an amide (acetyl), never a free amine; an amide
    carbonyl carbon whose partner is a nitrogen is kept an amide (N-methyl);
    anything else takes a methyl (thioether, ester, ketone, C-C - all
    non-titratable).

    Parameters
    ----------
    mol : Molecule
        The structure the bond lives in (read for element symbols).
    local_idx : int
        Global index in ``mol`` of the bond's in-residue atom.
    partner_idx : int
        Global index in ``mol`` of the bond's other-residue atom.
    rw : rdkit.Chem.RWMol
        The residue built from its base SMILES (read for the local atom's
        carbonyl double bond).
    smi_local_idx : int
        Index of the local atom in ``rw`` (its base-SMILES counterpart).

    Returns
    -------
    kind : str
        ``"amide_n"``, ``"amide_c"`` or ``"other"``.
    """
    if str(mol.element[local_idx]) == "N":
        return "amide_n"
    if _is_carbonyl_carbon(rw, smi_local_idx) and str(mol.element[partner_idx]) == "N":
        return "amide_c"
    return "other"


def _capped_residue_rdkit(
    mol: Molecule, spec: PerResidueSpec, base_smiles: str
) -> "tuple[RWMol, list[int]]":
    """Build ``spec``'s residue from ``base_smiles`` and cap it for its chain
    context and every non-peptide crosslink, returning the editable molecule
    and the cap-atom indices.

    Two independent things get capped, both inertly (no titratable group left
    behind on the severed side):

    - Backbone (:class:`ChainResidueSpec` only): acetyl / N-methyl amide caps
      go on the peptide-bonded backbone sides (the ``N`` / ``C`` atoms); a
      genuine free terminus is left as the free amine or free carboxylic acid
      ``base_smiles`` already provides - no oxygen is added or removed to
      synthesize it.
    - Crosslinks (every spec type): each non-peptide inter-residue bond found
      by :func:`_inter_residue_crosslinks` (a disulfide, thioether, glycan
      link, isopeptide, ...) is inert-capped too, so a residue whose sidechain
      is covalently tied up (e.g. a Cys-like ``SG`` in a thioether) is never
      offered to a downstream pKa tool as if that group were still free (a
      free thiol, a free amine, ...). The cap reflects the real partner element
      (see :func:`_attach_amine_amide` / :func:`_attach_hydroxyl` /
      :func:`_attach_phosphate`), and a condensation crosslink's displaced
      leaving group is stripped first (see :func:`_saturated_leaving_atom`).

    Shared first stage behind :func:`_cap_residue_smiles` (which keeps the
    caps) and :func:`_uncapped_residue_smiles` (which strips them to recover
    the bare residue skeleton, guaranteed a subgraph of the capped molecule).

    Parameters
    ----------
    mol : Molecule
        The molecule the spec was detected in.
    spec : ChainResidueSpec or ScaffoldSpec or CovalentLigandSpec or LigandSpec
        The residue to cap. For a :class:`ChainResidueSpec`, ``spec.is_n_term``
        / ``spec.is_c_term`` decide which backbone side(s) get a cap versus a
        free terminus; the other spec types have no backbone to cap, only
        crosslinks.
    base_smiles : str
        SMILES describing the residue's full chemistry (e.g. an RCSB ligand
        SMILES).

    Returns
    -------
    rw : rdkit.Chem.RWMol
        Editable, unsanitized capped molecule.
    cap_atoms : list of int
        Indices of every atom belonging to a cap, for
        :func:`_uncapped_residue_smiles` to strip.

    Raises
    ------
    ValueError
        If ``base_smiles`` does not parse, if a :class:`ChainResidueSpec`'s
        backbone ``N`` / ``C`` atom is missing or cannot be mapped onto
        ``base_smiles``, or if a crosslink atom cannot be mapped onto
        ``base_smiles``. This is the signal
        :func:`capNonstandardResiduesForTitration` catches to fall back to
        titrating the residue uncapped.
    """
    rw, res_to_smi = _isolated_residue_rdkit(mol, spec, base_smiles)
    rw.UpdatePropertyCache(strict=False)
    mapped = set(res_to_smi.values())

    def _tag(cap_idxs: list[int]) -> None:
        # Cap atoms are marked with a property rather than tracked by index, so
        # they survive the leaving-group removals below (which shift indices).
        for i in cap_idxs:
            rw.GetAtomWithIdx(i).SetBoolProp("_cap", True)

    # Plan every crosslink on the pristine molecule (kind, partner element, and
    # any leaving group to strip) before mutating it, so those lookups read
    # unshifted indices and valences.
    plan = []
    for local_g, partner_g in _inter_residue_crosslinks(mol, spec):
        if local_g not in res_to_smi:
            raise ValueError(
                f"Residue {spec.resname} ({spec.residue}) crosslink atom "
                f"{str(mol.name[local_g])!r} could not be mapped onto its "
                "SMILES; cannot cap it."
            )
        smi_local = res_to_smi[local_g]
        kind = _classify_junction(mol, local_g, partner_g, rw, smi_local)
        leaving = (
            _saturated_leaving_atom(rw, smi_local, mapped) if kind == "other" else None
        )
        plan.append((smi_local, kind, str(mol.element[partner_g]), leaving))

    # Backbone (chain residues only), located by atom name as before.
    if isinstance(spec, ChainResidueSpec):
        mask = getResidueMask(mol, spec)

        def _smi_of_named(name: str) -> int:
            g = np.where(mask & (mol.name == name))[0]
            if len(g) == 0 or int(g[0]) not in res_to_smi:
                raise ValueError(
                    f"Residue {spec.resname} ({spec.residue}) backbone atom "
                    f"{name!r} not found / not mapped; cannot cap it."
                )
            return res_to_smi[int(g[0])]

        if not spec.is_n_term:
            _tag(_attach_ace_cap(rw, _smi_of_named("N")))
        if not spec.is_c_term:
            _tag(_attach_nme_cap(rw, _smi_of_named("C")))

    # Non-peptide crosslinks (every spec type): inert-cap each one so no severed
    # sidechain looks like a free titratable group. A severed nitrogen becomes
    # an amide and an amide carbonyl carbon an N-methyl amide (as before); any
    # other junction is capped to reflect the real partner element - a
    # nitrogen-linked carbon (glycosidic) to an amide, an oxygen partner to a
    # hydroxyl, a phosphorus partner to a phosphate, everything else to a
    # methyl. When a condensation crosslink left the atom fully valent in the
    # free-form SMILES, the displaced leaving group is stripped first.
    leaving_atoms = []
    for smi_local, kind, partner_el, leaving in plan:
        if kind == "amide_n":
            _tag(_attach_ace_cap(rw, smi_local))
        elif kind == "amide_c":
            _tag(_attach_nme_cap(rw, smi_local))
        else:
            if (
                partner_el == "O"
                and leaving is not None
                and not _is_carbonyl_carbon(rw, smi_local)
            ):
                # The crosslink atom is already a complete group in the base
                # SMILES (e.g. a phosphate whose phosphorus is at valence 5)
                # carrying an -OH at the linkage position. That -OH already
                # represents the outgoing O-bond, so leave it untouched:
                # stripping and recapping it would drop an oxygen the template
                # strip cannot restore, leaving RDKit to fill the open valence
                # with a spurious hydrogen (an H-phosphonate instead of a
                # phosphate).
                continue
            if leaving is not None:
                leaving_atoms.append(leaving)
            if partner_el == "N":
                _tag(_attach_amine_amide(rw, smi_local))
            elif partner_el == "O" and not _is_carbonyl_carbon(rw, smi_local):
                # A hydroxyl keeps a phosphate ionizable and an sp3 carbon a
                # neutral alcohol, but on a carbonyl carbon it would be a
                # titratable free acid; that case falls through to a methyl.
                _tag(_attach_hydroxyl(rw, smi_local))
            elif partner_el == "P":
                _tag(_attach_phosphate(rw, smi_local))
            else:
                _tag(_attach_methyl(rw, smi_local))

    for idx in sorted(set(leaving_atoms), reverse=True):
        rw.RemoveAtom(idx)
    cap_atoms = [a.GetIdx() for a in rw.GetAtoms() if a.HasProp("_cap")]
    return rw, cap_atoms


def _isolated_residue_rdkit(
    mol: Molecule, spec: PerResidueSpec, base_smiles: str
) -> "tuple[RWMol, dict]":
    """Build ``spec``'s residue from its full ``base_smiles`` and map the
    structure residue's heavy atoms onto that SMILES-derived molecule.

    The molecule is built from ``base_smiles`` (e.g. the complete RCSB free-
    amino-acid SMILES), NOT from the structure residue's atoms, so it always
    carries the residue's full chemistry - even when the deposited structure
    has a trimmed sidechain, which would otherwise yield an incomplete
    titration molecule (worst case, missing the ionizable group of interest).
    The structure residue is used only to locate that chemistry within the
    SMILES: its heavy atoms are mapped onto ``base_smiles`` through a
    maximum-common-substructure match (by connectivity, not by atom name), so
    callers can look up any structure atom - the backbone ``N`` / ``C``, a
    crosslinked sidechain atom - by its global index into ``mol``.

    Parameters
    ----------
    mol : Molecule
        The molecule the spec was detected in.
    spec : ChainResidueSpec or ScaffoldSpec or CovalentLigandSpec or LigandSpec
        The residue to isolate.
    base_smiles : str
        SMILES describing the residue's full chemistry (e.g. an RCSB ligand
        SMILES), whose atoms include the structure residue's own chemistry.

    Returns
    -------
    rw : rdkit.Chem.RWMol
        Editable, unsanitized RDKit molecule built from ``base_smiles``.
    res_to_smi : dict
        Maps each structure-residue heavy atom's **global** index (into
        ``mol``) to its atom index in ``rw``. Atoms of the structure residue
        that could not be mapped onto ``base_smiles`` (e.g. because the MCS
        match did not cover them) are simply absent from the dict.

    Raises
    ------
    ValueError
        If ``base_smiles`` does not parse. This is one of the signals
        :func:`capNonstandardResiduesForTitration` catches to fall back to
        titrating the residue uncapped (the other, an uncappable or
        unmappable backbone/crosslink atom, is raised by
        :func:`_capped_residue_rdkit` once it looks up the specific atom it
        needs).
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    from moleculekit.rdkittools import molecule_to_rdkitmol

    smi_rd = Chem.MolFromSmiles(base_smiles)
    if smi_rd is None:
        raise ValueError(
            f"Could not parse template SMILES {base_smiles!r} for residue "
            f"{spec.resname} ({spec.residue})."
        )

    res = mol.copy(sel=getResidueMask(mol, spec))
    res.remove("element H", _logger=False)

    # Map the structure residue's heavy atoms onto the complete SMILES by
    # their graph position (element + connectivity), so N-methyl backbones,
    # proline, and sidechain amine/carboxyl groups don't confuse which atoms
    # are which.
    res.guessBonds()
    res_rd = molecule_to_rdkitmol(res, sanitize=False, _logger=False)
    mcs = rdFMCS.FindMCS(
        [res_rd, smi_rd],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        timeout=10,
    )
    patt = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
    res_match = res_rd.GetSubstructMatch(patt) if patt is not None else ()
    smi_match = smi_rd.GetSubstructMatch(patt) if patt is not None else ()

    # res was filtered from mol then had its hydrogens removed, so its atoms
    # are exactly mol's masked heavy atoms in ascending global order; that
    # order is preserved by molecule_to_rdkitmol, so res_rd's i-th atom is
    # the i-th heavy masked global atom.
    res_sel_global = np.where(getResidueMask(mol, spec))[0]
    heavy_global = [g for g in res_sel_global if str(mol.element[g]) != "H"]
    res_local_to_global = {i: int(g) for i, g in enumerate(heavy_global)}
    res_to_smi = {
        res_local_to_global[r]: int(s)
        for r, s in zip(res_match, smi_match)
        if r in res_local_to_global
    }

    rw = Chem.RWMol(smi_rd)
    return rw, res_to_smi


def _cap_residue_smiles(mol: Molecule, spec: PerResidueSpec, base_smiles: str) -> str:
    """Return a SMILES for ``spec``'s residue capped for its chain context.

    Acetyl / N-methyl amide caps are added to the peptide-bonded backbone
    sides (the ``N`` / ``C`` atoms); a genuine free terminus is left open so
    a downstream pKa predictor titrates it. Coordinates are irrelevant to the
    returned SMILES: ``base_smiles`` only needs to describe the residue's
    connectivity, and RDKit valence assigns the backbone hydrogens once the
    capped molecule is sanitized.

    Parameters
    ----------
    mol : Molecule
        The molecule the spec was detected in.
    spec : ChainResidueSpec or ScaffoldSpec or CovalentLigandSpec or LigandSpec
        The residue to cap. For a :class:`ChainResidueSpec`, ``spec.is_n_term``
        / ``spec.is_c_term`` decide which side(s) get a cap versus a free
        terminus; the other spec types only get their crosslinks capped.
    base_smiles : str
        SMILES describing the residue's full chemistry (e.g. an RCSB ligand
        SMILES); the capped molecule is built from it, with the backbone
        located via :func:`_isolated_residue_rdkit`.

    Returns
    -------
    smiles : str
        Capped SMILES, backbone protonation resolved by RDKit valence.
    """
    from rdkit import Chem

    rw, _cap_atoms = _capped_residue_rdkit(mol, spec, base_smiles)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return Chem.MolToSmiles(out)


def _uncapped_residue_smiles(
    mol: Molecule, spec: PerResidueSpec, base_smiles: str
) -> str:
    """Return a SMILES anchor for ``spec``'s residue heavy-atom skeleton.

    Used later as a relaxed substructure query that locates the residue's
    atoms inside the capped molecule built by :func:`_cap_residue_smiles`
    (see :func:`capNonstandardResiduesForTitration`), so this anchor must
    itself be a subgraph of that capped SMILES. It is derived to be exactly
    that - the same capped molecule with its cap atoms deleted - so the
    subgraph relationship holds by construction for every chain context, with
    no per-context reasoning needed here: a peptide-bonded ``C`` side (whose
    cap is an amide) strips back to a one-oxygen carbonyl, while a genuine
    C-terminus (left a free carboxylic acid by the capping step) keeps both
    its oxygens.

    Parameters
    ----------
    mol : Molecule
        The molecule the spec was detected in.
    spec : ChainResidueSpec or ScaffoldSpec or CovalentLigandSpec or LigandSpec
        The residue whose skeleton to return.
    base_smiles : str
        SMILES describing the residue's full chemistry (e.g. an RCSB ligand
        SMILES).

    Returns
    -------
    smiles : str
        Uncapped SMILES of the residue's heavy-atom skeleton, context-shaped
        on the ``C`` side to substructure-match :func:`_cap_residue_smiles`'s
        output for the same ``spec``.
    """
    from rdkit import Chem

    rw, cap_atoms = _capped_residue_rdkit(mol, spec, base_smiles)
    for idx in sorted(cap_atoms, reverse=True):
        rw.RemoveAtom(idx)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return Chem.MolToSmiles(out)


def _titration_specs(specs: list, smiles: dict | None, bases: dict | None = None):
    """Yield ``(key, spec, base_smiles)`` for each unique template-requiring
    residue in ``specs``, deduplicated by key.

    Shared spec walk behind :func:`capNonstandardResiduesForTitration` and
    :func:`templatesFromTitration`, so both agree on which residues are
    processed, their keys (``new_resname or resname`` for a chain residue,
    ``resname`` for a ligand, scaffold, or covalent ligand), and their base
    SMILES. Specs that do not need a template
    (:func:`moleculekit.tools.nonstandard_residues.requiresTemplate` is False
    - e.g. a disulfide ``CYS`` renamed to ``CYX``) are skipped, and so is
    every :class:`~moleculekit.tools.nonstandard_residues.GlycanSpec`: a
    sugar's pKa is not titrated (GLYCAM has no ionizable sugar unit) and it
    gets its topology from a shipped residue CIF rather than an RCSB /
    override SMILES base.

    Parameters
    ----------
    specs : list
        The specs from
        :func:`moleculekit.tools.nonstandard_residues.detectNonStandardResidues`.
    smiles : dict or None
        ``{resname: smiles}`` overrides; resnames absent here are fetched from
        RCSB by their CCD code.
    bases : dict or None
        ``{key: base_smiles}`` known base SMILES per *key* (not resname), e.g.
        read back from a titration CSV's echoed ``base`` column. A key present
        here uses its entry verbatim - no override lookup, no RCSB fetch.

    Yields
    ------
    key : str
        ``new_resname or resname`` (chain residue) or ``resname`` (ligand,
        scaffold, or covalent ligand).
    spec : ChainResidueSpec or ScaffoldSpec or CovalentLigandSpec or LigandSpec
        The spec for this key.
    base_smiles : str
        The residue's full base SMILES.

    Raises
    ------
    RuntimeError
        If a residue needing a template has neither an override in ``smiles``
        nor a fetchable RCSB SMILES.
    """
    overrides = dict(smiles or {})
    cache: dict[str, str] = {}

    def base_for(resname: str) -> str:
        if resname in overrides:
            return overrides[resname]
        if resname not in cache:
            cache[resname] = rcsbFetchLigandSmiles(resname)
        return cache[resname]

    seen: set[str] = set()
    for spec in specs:
        # GlycanSpec is explicitly excluded here (on top of the
        # requiresTemplate() gate below) so a sugar is never titrated even if
        # that gate's definition changes later: GLYCAM sugars carry no
        # ionizable group and never need a titration-input SMILES.
        if isinstance(spec, GlycanSpec):
            continue
        if not requiresTemplate(spec):
            continue
        if isinstance(spec, ChainResidueSpec):
            key = spec.new_resname or spec.resname
        elif isinstance(spec, (LigandSpec, ScaffoldSpec, CovalentLigandSpec)):
            key = spec.resname
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        if bases is not None and key in bases:
            yield key, spec, bases[key]
        else:
            yield key, spec, base_for(spec.resname)


def capNonstandardResiduesForTitration(
    mol: Molecule,
    specs: list,
    smiles: dict | None = None,
    outfile: str | None = None,
    _logger: bool = True,
) -> dict[str, str]:
    """Build per-context pKa-titration input SMILES for the non-standard
    residues, keyed by ``new_resname or resname``.

    Each chain-resident NCAA is capped for its chain context, and every
    scaffold / covalent-ligand residue has its non-peptide crosslink(s) inert-
    capped the same way (see :func:`_cap_residue_smiles`); a genuinely free
    ligand is passed through uncapped, as is any residue whose backbone or
    crosslink cannot be capped (a warning is logged and it is titrated whole).
    Entries are deduplicated by key so an NCAA appearing in several places is
    titrated once per unique (chemistry, context).

    The result is a plain ``{key: smiles}`` dict. The caller runs its own pKa
    tool over the values (e.g. AcePka) and passes the protonated result back to
    :func:`templatesFromTitration` - as a dict of the same keys, or, for a
    file-based tool, as the path to its output CSV (see ``outfile``).

    Parameters
    ----------
    mol : Molecule
        The molecule the specs were detected in.
    specs : list
        The specs returned by
        :func:`moleculekit.tools.nonstandard_residues.detectNonStandardResidues`.
    smiles : dict or None
        Optional ``{resname: smiles}`` overrides; resnames absent here are
        fetched from RCSB by their CCD code.
    outfile : str or None
        When given, also write the result as a CSV with a ``key``, ``SMILES``
        and ``base`` column per entry - the input for a file-based pKa tool
        that rewrites ``SMILES`` and echoes the other columns through (e.g.
        AcePka). The echoed ``base`` column is what lets
        :func:`templatesFromTitration` re-derive each anchor from exactly the
        base SMILES that was titrated, with no state carried in between.

    Returns
    -------
    titration : dict
        ``{key: smiles}`` to titrate, one entry per unique template-requiring
        residue.

    Raises
    ------
    RuntimeError
        If a residue needing a template has neither an override in ``smiles``
        nor a fetchable RCSB SMILES.
    """
    out: dict[str, str] = {}
    bases: dict[str, str] = {}
    for key, spec, base in _titration_specs(specs, smiles):
        bases[key] = base
        if isinstance(spec, LigandSpec):
            out[key] = base
            continue
        try:
            out[key] = _cap_residue_smiles(mol, spec, base)
        except ValueError as e:  # non-alpha / uncappable backbone: titrate whole
            if _logger:
                logger.warning(
                    f"Could not cap residue {spec.resname} ({spec.residue}); "
                    f"titrating it uncapped ({e})."
                )
            out[key] = base
    if outfile is not None:
        with open(outfile, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["key", "SMILES", "base"])
            for key, smi in out.items():
                writer.writerow([key, smi, bases[key]])
    return out


def _relaxed_query(smiles: str) -> "Mol":
    """Build a relaxed substructure query molecule from ``smiles``.

    The query matches on element and connectivity only: bond orders are made
    generic (single/double/aromatic all match each other), so it still
    matches a residue skeleton whose titrated group AcePka has resonated or
    changed the bond order of (e.g. a carboxylic acid ``C(=O)-OH`` versus its
    deprotonated carboxylate ``C(=O)-O^-``, which RDKit can represent with a
    different bond order on the two C-O bonds).

    Hydrogen counts are made generic too. Stripping a crosslink atom's leaving
    group and cap (see :func:`_uncapped_residue_smiles`) leaves that atom with a
    different hydrogen count in the anchor than it has in the capped molecule
    (its cap bond is gone), and a stereocentre's explicit ``[C@H]`` hydrogen is
    frozen rather than re-derived from the reduced valence; matching on hydrogen
    count would then fail to relocate the atom.

    Parameters
    ----------
    smiles : str
        SMILES of the residue anchor to turn into a query, e.g. one entry's
        ``residue_smiles`` from :func:`capNonstandardResiduesForTitration`.

    Returns
    -------
    query : rdkit.Chem.Mol
        Query molecule usable with ``GetSubstructMatch``.
    """
    from rdkit import Chem
    from rdkit.Chem import rdmolops

    q = Chem.MolFromSmiles(smiles)
    # Drop baked-in hydrogen counts (from valence or from stereo ``[C@H]``) and
    # the radical electrons an under-valent stripped atom parses with, so
    # neither constrains the match; connectivity and element still do.
    for atom in q.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetNumRadicalElectrons(0)
    Chem.SanitizeMol(q)
    params = rdmolops.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    return rdmolops.AdjustQueryProperties(q, params)


def _submol_from_atoms(mol: "Mol", atom_idxs: list[int] | tuple[int, ...]) -> "Mol":
    """Extract the sub-molecule spanning ``atom_idxs``.

    Only bonds with both endpoints in ``atom_idxs`` are kept, so the
    returned sub-molecule excludes anything outside the matched atom set
    (e.g. cap atoms attached to a residue's backbone). Each atom keeps the
    formal charge and hydrogen count it had in ``mol``, so AcePka's
    protonation decisions on the residue survive the extraction.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule to extract the sub-molecule from.
    atom_idxs : list[int] or tuple[int, ...]
        Indices (into ``mol``) of the atoms to keep.

    Returns
    -------
    submol : rdkit.Chem.Mol
        Sub-molecule spanning ``atom_idxs`` and the bonds among them.
    """
    from rdkit import Chem

    idxset = set(int(i) for i in atom_idxs)
    bond_idxs = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in idxset and b.GetEndAtomIdx() in idxset
    ]
    return Chem.PathToSubmol(mol, bond_idxs)


def templatesFromTitration(
    mol: Molecule,
    specs: list,
    protonated: "dict | str | os.PathLike",
    smiles: dict | None = None,
    outfile: str | None = None,
) -> dict[str, str]:
    """Strip the caps off pKa-titrated SMILES back into per-residue templates.

    Inverse of :func:`capNonstandardResiduesForTitration`, driven by the same
    ``mol`` and ``specs`` so no intermediate state has to be carried between
    the two calls. For each template-requiring chain, scaffold, or covalent-
    ligand residue, its uncapped anchor is re-derived
    (:func:`_uncapped_residue_smiles`) and used as a relaxed substructure query
    (:func:`_relaxed_query`) to locate that residue's own atoms inside its
    protonated, capped SMILES; the matched atoms are extracted
    (:func:`_submol_from_atoms`), dropping the ACE / NME / crosslink caps that
    fall outside the match. A pKa tool only ever changes protonation state,
    never the heavy-atom skeleton, so the anchor is guaranteed to be a
    subgraph of the protonated molecule. Residues titrated whole (genuinely
    free ligands, or residues whose backbone or crosslink could not be capped)
    are returned unchanged.

    Parameters
    ----------
    mol : Molecule
        The molecule the specs were detected in (the same one passed to
        :func:`capNonstandardResiduesForTitration`).
    specs : list
        The specs from
        :func:`moleculekit.tools.nonstandard_residues.detectNonStandardResidues`.
    protonated : dict or str or os.PathLike
        ``{key: smiles}`` of the pKa-tool-protonated titration output, keyed
        exactly as :func:`capNonstandardResiduesForTitration` returned - or the
        path to the tool's output CSV (e.g. AcePka's ``protonated.csv``). The
        CSV must carry ``key`` and ``SMILES`` columns; when the ``base`` column
        written by ``capNonstandardResiduesForTitration(outfile=...)`` was
        echoed through by the tool, each anchor is re-derived from that echoed
        base - provably the same SMILES that was titrated, with no RCSB
        re-fetch and no ``smiles=`` overrides to carry between sessions.
    smiles : dict or None
        The same ``{resname: smiles}`` overrides passed to
        :func:`capNonstandardResiduesForTitration`, so anchors are re-derived
        from identical base SMILES. Not needed when ``protonated`` is a CSV
        path with a ``base`` column.
    outfile : str or None
        When given, also write the templates as JSON mapping each key to
        ``{"smiles": template}`` - the shape residue-template consumers take.

    Returns
    -------
    templates : dict
        ``{key: smiles}``, one entry per key, with caps removed and the pKa
        tool's protonation applied.

    Raises
    ------
    RuntimeError
        If ``protonated`` is missing a key, a protonated SMILES cannot be
        parsed, or a residue's anchor cannot be located inside its protonated
        SMILES.
    """
    from rdkit import Chem

    bases = None
    if isinstance(protonated, (str, os.PathLike)):
        with open(protonated, newline="") as fh:
            rows = list(csv.DictReader(fh))
        missing = {"key", "SMILES"} - set(rows[0] if rows else ())
        if missing:
            raise RuntimeError(
                f"{protonated} is missing the {sorted(missing)} column(s); "
                "expected the CSV written by capNonstandardResiduesForTitration "
                "with its SMILES column rewritten by the pKa tool."
            )
        bases = {r["key"]: r["base"] for r in rows if r.get("base")}
        protonated = {r["key"]: r["SMILES"] for r in rows}

    out: dict[str, str] = {}
    for key, spec, base in _titration_specs(specs, smiles, bases=bases):
        if key not in protonated:
            raise RuntimeError(f"protonated has no entry for key {key!r}.")
        prot_smi = protonated[key]
        if isinstance(spec, LigandSpec):
            out[key] = prot_smi
            continue
        try:
            anchor = _uncapped_residue_smiles(mol, spec, base)
        except ValueError:
            # Titrated whole (uncappable backbone); nothing to strip.
            out[key] = prot_smi
            continue
        capped_prot = Chem.MolFromSmiles(prot_smi)
        if capped_prot is None:
            raise RuntimeError(
                f"Could not parse protonated SMILES {prot_smi!r} for key {key!r}."
            )
        query = _relaxed_query(anchor)
        match = capped_prot.GetSubstructMatch(query)
        if not match:
            raise RuntimeError(
                f"Could not locate residue skeleton {anchor!r} for key {key!r} "
                f"inside protonated SMILES {prot_smi!r}."
            )
        out[key] = Chem.MolToSmiles(_submol_from_atoms(capped_prot, match))
    if outfile is not None:
        with open(outfile, "w") as fh:
            json.dump({k: {"smiles": v} for k, v in out.items()}, fh, indent=2)
    return out
