"""Prepare non-standard residues for pKa titration and fold the result back
into per-residue build templates.

The public entry points are :func:`capNonstandardResiduesForTitration` and
:func:`templatesFromTitration`, both dict-in / dict-out (``{key: smiles}``).
The caller runs its own pKa tool (e.g. AcePka) over the titration SMILES and
hands the protonated result back; moleculekit never reads the tool's file
formats.

Each residue is built from its full base SMILES (an RCSB ligand descriptor or
a caller override), so it carries the complete chemistry even when the
deposited structure has a trimmed sidechain. Every *outgoing* inter-residue
bond is then neutral-capped, so the pKa tool sees a chemically sane molecule
and titrates only the groups that are genuinely free:

- a backbone peptide bond gets an amide cap - acetyl on the ``N`` side,
  N-methyl on the carbonyl ``C`` side (the ACE / NME that keep a mid-chain
  backbone neutral);
- a sidechain / scaffold crosslink gets an inert cap chosen to keep the
  junction atom non-titratable (acetyl on a nitrogen, N-methyl on an amide
  carbonyl carbon, methyl otherwise);
- a genuine free terminus or free-ligand end is left uncapped, so the pKa tool
  assigns its charge (a real C-terminus deprotonates, an N-terminus protonates).

The backbone atoms are located by their structure names (``N`` / ``C``) mapped
onto the SMILES; crosslink atoms are found by walking ``mol.bonds`` and mapped
the same way. Coordinates never affect the returned SMILES - only connectivity
does. After titration, :func:`templatesFromTitration` strips the caps back off
(the residue is a subgraph of its capped molecule) to yield one template SMILES
per residue for the builder.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
from moleculekit.molecule import Molecule
from moleculekit.rcsb import rcsbFetchLigandSmiles
from moleculekit.tools.nonstandard_residues import (
    ChainResidueSpec,
    CovalentLigandSpec,
    LigandSpec,
    PerResidueSpec,
    ScaffoldSpec,
    requiresTemplate,
)

if TYPE_CHECKING:
    from rdkit.Chem import Mol, RWMol

logger = logging.getLogger(__name__)


def _spec_residue_mask(mol: Molecule, spec: PerResidueSpec) -> np.ndarray:
    """Boolean mask selecting ``spec``'s residue in ``mol``.

    Parameters
    ----------
    mol : Molecule
        The molecule the spec was detected in.
    spec : ChainResidueSpec or ScaffoldSpec or CovalentLigandSpec or LigandSpec
        A residue spec from
        :func:`moleculekit.tools.nonstandard_residues.detectNonStandardResidues`.

    Returns
    -------
    mask : np.ndarray
        Boolean mask, True on the atoms of ``spec``'s residue.
    """
    rid = spec.residue
    return (
        (mol.resname == spec.resname)
        & (mol.segid == str(rid.segid))
        & (mol.chain == str(rid.chain))
        & (mol.resid == int(rid.resid))
        & (mol.insertion == str(rid.insertion))
    )


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

    in_res = _spec_residue_mask(mol, spec)
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
    from rdkit import Chem

    if str(mol.element[local_idx]) == "N":
        return "amide_n"
    atom = rw.GetAtomWithIdx(smi_local_idx)
    is_carbonyl = atom.GetSymbol() == "C" and any(
        b.GetBondType() == Chem.BondType.DOUBLE
        and b.GetOtherAtom(atom).GetSymbol() == "O"
        for b in atom.GetBonds()
    )
    if is_carbonyl and str(mol.element[partner_idx]) == "N":
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
      free thiol, a free amine, ...).

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
    cap_atoms: list[int] = []

    # Backbone (chain residues only), located by atom name as before.
    if isinstance(spec, ChainResidueSpec):
        mask = _spec_residue_mask(mol, spec)

        def _smi_of_named(name: str) -> int:
            g = np.where(mask & (mol.name == name))[0]
            if len(g) == 0 or int(g[0]) not in res_to_smi:
                raise ValueError(
                    f"Residue {spec.resname} ({spec.residue}) backbone atom "
                    f"{name!r} not found / not mapped; cannot cap it."
                )
            return res_to_smi[int(g[0])]

        if not spec.is_n_term:
            cap_atoms += _attach_ace_cap(rw, _smi_of_named("N"))
        if not spec.is_c_term:
            cap_atoms += _attach_nme_cap(rw, _smi_of_named("C"))

    # Non-peptide crosslinks (every spec type): inert-cap each one so no
    # severed sidechain looks like a free titratable group.
    for local_g, partner_g in _inter_residue_crosslinks(mol, spec):
        if local_g not in res_to_smi:
            raise ValueError(
                f"Residue {spec.resname} ({spec.residue}) crosslink atom "
                f"{str(mol.name[local_g])!r} could not be mapped onto its "
                "SMILES; cannot cap it."
            )
        smi_local = res_to_smi[local_g]
        kind = _classify_junction(mol, local_g, partner_g, rw, smi_local)
        if kind == "amide_n":
            cap_atoms += _attach_ace_cap(rw, smi_local)
        elif kind == "amide_c":
            cap_atoms += _attach_nme_cap(rw, smi_local)
        else:
            cap_atoms += _attach_methyl(rw, smi_local)

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

    res = mol.copy(sel=_spec_residue_mask(mol, spec))
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
    res_sel_global = np.where(_spec_residue_mask(mol, spec))[0]
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


def _titration_specs(specs: list, smiles: dict | None):
    """Yield ``(key, spec, base_smiles)`` for each unique template-requiring
    residue in ``specs``, deduplicated by key.

    Shared spec walk behind :func:`capNonstandardResiduesForTitration` and
    :func:`templatesFromTitration`, so both agree on which residues are
    processed, their keys (``new_resname or resname`` for a chain residue,
    ``resname`` for a ligand, scaffold, or covalent ligand), and their base
    SMILES. Specs that do not need a template
    (:func:`moleculekit.tools.nonstandard_residues.requiresTemplate` is False
    - e.g. a disulfide ``CYS`` renamed to ``CYX``, a glycosylated ``ASN``) are
    skipped.

    Parameters
    ----------
    specs : list
        The specs from
        :func:`moleculekit.tools.nonstandard_residues.detectNonStandardResidues`.
    smiles : dict or None
        ``{resname: smiles}`` overrides; resnames absent here are fetched from
        RCSB by their CCD code.

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
        yield key, spec, base_for(spec.resname)


def capNonstandardResiduesForTitration(
    mol: Molecule,
    specs: list,
    smiles: dict | None = None,
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
    :func:`templatesFromTitration` as a dict of the same keys; moleculekit does
    not read or write any pKa tool's file format.

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
    _logger : bool
        Emit a warning when a residue cannot be capped and is titrated whole.

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
    for key, spec, base in _titration_specs(specs, smiles):
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
    return out


def _relaxed_query(smiles: str) -> "Mol":
    """Build a relaxed substructure query molecule from ``smiles``.

    The query matches on element and connectivity only: bond orders are made
    generic (single/double/aromatic all match each other), so it still
    matches a residue skeleton whose titrated group AcePka has resonated or
    changed the bond order of (e.g. a carboxylic acid ``C(=O)-OH`` versus its
    deprotonated carboxylate ``C(=O)-O^-``, which RDKit can represent with a
    different bond order on the two C-O bonds).

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
    mol: Molecule, specs: list, protonated: dict, smiles: dict | None = None
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
    protonated : dict
        ``{key: smiles}`` of the pKa-tool-protonated titration output, keyed
        exactly as :func:`capNonstandardResiduesForTitration` returned.
    smiles : dict or None
        The same ``{resname: smiles}`` overrides passed to
        :func:`capNonstandardResiduesForTitration`, so anchors are re-derived
        from identical base SMILES.

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

    out: dict[str, str] = {}
    for key, spec, base in _titration_specs(specs, smiles):
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
    return out
