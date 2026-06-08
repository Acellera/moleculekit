from moleculekit.molecule import Molecule
from rdkit import Chem
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Boundary heavy atoms in :func:`template_residue_from_smiles` whose
# external (cross-residue) bond goes to a metal element have their
# leaving-group hydrogens stripped (as for any cross-residue bond) AND
# their formal charge reduced by one per metal-bond order, matching the
# deprotonation chemistry of biological metal-thiolate / metal-alkoxide
# / metal-carboxylate coordination (e.g. -SH + Zn -> -S-...Zn). For
# non-metal cross-residue bonds (peptide N-C, disulfide S-S,
# glycosidic O-C, ...) the H is stripped without a formal charge
# change. Distinct from
# ``moleculekit.tools.nonstandard_residues._ION_RESNAMES``, which is a
# residue-name set for spec detection and also covers non-metal anions
# (Cl-, I-, oxyanions) that should not trigger deprotonation here.
from moleculekit.periodictable import METAL_ELEMENTS as _METAL_ELEMENTS


def rdkitmol_to_molecule(rmol):
    """Converts an RDKit molecule to a Molecule object

    Parameters
    ----------
    rmol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to convert

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> rmol = Chem.MolFromSmiles("C1CCCCC1")
    >>> rmol = Chem.AddHs(rmol)
    >>> AllChem.EmbedMolecule(rmol)
    >>> AllChem.MMFFOptimizeMolecule(rmol)
    >>> mol = Molecule.fromRDKitMol(rmol)
    """
    from rdkit.Chem.rdchem import BondType
    from collections import defaultdict

    bondtype_map = defaultdict(lambda: "un")
    bondtype_map.update(
        {
            BondType.SINGLE: "1",
            BondType.DOUBLE: "2",
            BondType.TRIPLE: "3",
            BondType.QUADRUPLE: "4",
            BondType.QUINTUPLE: "5",
            BondType.HEXTUPLE: "6",
            BondType.AROMATIC: "ar",
            BondType.DATIVE: "mc",
        }
    )

    mol = Molecule()
    mol.empty(rmol.GetNumAtoms(), numFrames=rmol.GetNumConformers())
    mol.record[:] = "HETATM"
    mol.resname[:] = rmol.GetProp("_Name") if rmol.HasProp("_Name") else "UNK"
    mol.resid[:] = 1
    mol.viewname = mol.resname[0]

    if rmol.GetNumConformers() != 0:
        mol.coords = np.array(
            np.stack([cc.GetPositions() for cc in rmol.GetConformers()], axis=2),
            dtype=Molecule._dtypes["coords"],
        )

    def get_atomname(mol, element):
        i = 1
        while True:
            name = f"{element}{i}"
            if name not in mol.name:
                return name
            i += 1

    for i, atm in enumerate(rmol.GetAtoms()):
        mol.name[i] = f"{atm.GetSymbol()}_X"  # Rename later
        if atm.HasProp("_Name") and atm.GetProp("_Name") != "":
            mol.name[i] = atm.GetProp("_Name")
        mol.element[i] = atm.GetSymbol()
        mol.formalcharge[i] = atm.GetFormalCharge()
        if atm.HasProp("_TriposPartialCharge"):
            mol.charge[i] = atm.GetPropsAsDict()["_TriposPartialCharge"]
        if atm.HasProp("_TriposAtomType"):
            mol.atomtype[i] = atm.GetPropsAsDict()["_TriposAtomType"]

    for i in range(mol.numAtoms):
        if mol.name[i].endswith("_X"):
            mol.name[i] = get_atomname(mol, mol.element[i])

    for bo in rmol.GetBonds():
        mol.addBond(
            bo.GetBeginAtomIdx(), bo.GetEndAtomIdx(), bondtype_map[bo.GetBondType()]
        )

    return mol


def molecule_to_rdkitmol(
    mol: Molecule,
    sanitize: bool = False,
    kekulize: bool = False,
    assignStereo: bool = True,
    guessBonds: bool = False,
    _logger=True,
):
    """Converts the Molecule to an RDKit molecule

    Parameters
    ----------
    mol : Molecule
        The molecule to convert to an RDKit molecule
    sanitize : bool
        If True the molecule will be sanitized
    kekulize : bool
        If True the molecule will be kekulized
    assignStereo : bool
        If True the molecule will have stereochemistry assigned from its 3D coordinates
    guessBonds : bool
        If True the molecule will have bonds guessed
    """
    from moleculekit.molecule import calculateUniqueBonds
    from rdkit.Chem.rdchem import BondType
    from rdkit.Chem.rdchem import Conformer
    from rdkit import Chem

    bondtype_map = {
        "1": BondType.SINGLE,
        "2": BondType.DOUBLE,
        "3": BondType.TRIPLE,
        "4": BondType.QUADRUPLE,
        "5": BondType.QUINTUPLE,
        "6": BondType.HEXTUPLE,
        "ar": BondType.AROMATIC,
        "mc": BondType.DATIVE,
    }

    _rdmol = Chem.rdchem.RWMol()
    for i in range(mol.numAtoms):
        atm = Chem.rdchem.Atom(mol.element[i])
        atm.SetNoImplicit(
            False
        )  # Set to True to stop rdkit from adding implicit hydrogens
        atm.SetProp("_Name", mol.name[i])
        atm.SetFormalCharge(int(mol.formalcharge[i]))
        atm.SetProp("_TriposAtomType", mol.atomtype[i])
        atm.SetDoubleProp("_TriposPartialCharge", float(mol.charge[i]))
        _rdmol.AddAtom(atm)

    bonds = mol.bonds.copy()
    bondtype = mol.bondtype.copy()
    if guessBonds:
        bonds, bondtype = mol._guessBonds()
    bondtype[bondtype == "un"] = "1"

    if len(mol.bonds) == 0:
        logger.info(
            "No bonds found in molecule. If you want your rdkit molecule to have bonds, use guessBonds=True"
        )

    bonds, bondtypes = calculateUniqueBonds(bonds, bondtype)
    for i in range(bonds.shape[0]):
        _rdmol.AddBond(
            int(bonds[i, 0]),
            int(bonds[i, 1]),
            bondtype_map[bondtypes[i]],
        )

    for f in range(mol.numFrames):
        conf = Conformer(mol.numAtoms)
        for i in range(mol.numAtoms):
            conf.SetAtomPosition(i, mol.coords[i, :, f].tolist())
        _rdmol.AddConformer(conf, assignId=True)

    if sanitize:
        Chem.SanitizeMol(_rdmol)
    if kekulize:
        Chem.Kekulize(_rdmol)
    if assignStereo:
        Chem.AssignStereochemistryFrom3D(_rdmol)

    if _logger:
        logger.info(
            f"Converted Molecule to RDKit mol with SMILES: {Chem.MolToSmiles(_rdmol, kekuleSmiles=True)}"
        )

    if mol.numAtoms != _rdmol.GetNumAtoms():
        raise RuntimeError("Number of atoms changed while converting to rdkit molecule")
    unique_resnames = np.unique(mol.resname)
    _rdmol.SetProp(
        "_Name",
        unique_resnames[0] if len(unique_resnames) == 1 else "+".join(unique_resnames),
    )

    # Return non-editable version
    return Chem.Mol(_rdmol)


def _try_strip_unmatched_terminals(
    rmol_smi, unmatched_heavy_idx, residue, atom_mapping, boundary_local_idxs
):
    """Strip unmatched terminal heavy atoms from a SMILES template.

    When the SMILES is a superset of the actual residue (e.g. a covalent
    inhibitor's leaving group, or a free amino acid templated mid-chain),
    the extra atoms are typically singly-bonded terminals attached to a
    residue atom whose valence has been taken up by an inter-residue bond
    instead. We strip such atoms in two recognized cases:

    (a) The matched neighbor in the residue carries an explicit covalent
        bond to another residue (e.g. C10/C11/C12 of LFI to CYS sulfurs,
        or a peptide bond when guessBonds has populated mol.bonds).
    (b) The unmatched atom is a carboxyl -OH and the residue is non-terminal
        (no OXT atom). This preserves the legacy behaviour for amino acids
        whose peptide bonds aren't present in mol.bonds.

    Returns the modified SMILES, or ``None`` if any unmatched atom does not
    fit the pattern (in which case the caller should error rather than strip
    speculatively).
    """
    no_oxt = "OXT" not in residue.name
    to_strip = []
    for idx in unmatched_heavy_idx:
        idx = int(idx)
        atom = rmol_smi.GetAtomWithIdx(idx)
        heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() != "H"]
        if len(heavy_neighbors) != 1:
            return None
        neighbor = heavy_neighbors[0]
        bond = rmol_smi.GetBondBetweenAtoms(idx, neighbor.GetIdx())

        # MCS uses ``CompareAny`` on bonds and ``CompareElements`` on atoms,
        # so on a ``C(=O)O`` carbonyl the residue's lone O is paired
        # arbitrarily with either the =O or the -OH. If MCS picked the OH
        # we end up here with the =O unmatched and double-bonded. Swap:
        # strip the OH (single-bonded) sibling instead, so the next-pass
        # MCS pairs the residue's O with the =O.
        if (
            bond.GetBondType() == Chem.BondType.DOUBLE
            and atom.GetSymbol() == "O"
            and neighbor.GetSymbol() == "C"
            and no_oxt
        ):
            sibling_oh_idx = None
            for sb in neighbor.GetBonds():
                sib = sb.GetOtherAtom(neighbor)
                if (
                    sib.GetIdx() != idx
                    and sib.GetSymbol() == "O"
                    and sb.GetBondType() == Chem.BondType.SINGLE
                    and sib.GetIdx() in atom_mapping
                ):
                    # Sibling must be a terminal -OH (carbonyl C is its
                    # only heavy neighbor). Refuses to strip an ester /
                    # ether O whose other side is alkyl, since stripping
                    # a non-terminal atom would disconnect the rest of
                    # the SMILES on next-pass MCS.
                    sib_heavy = [
                        n for n in sib.GetNeighbors() if n.GetSymbol() != "H"
                    ]
                    if len(sib_heavy) != 1:
                        continue
                    sibling_oh_idx = sib.GetIdx()
                    break
            if sibling_oh_idx is not None:
                to_strip.append(sibling_oh_idx)
                continue

        if bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        if neighbor.GetIdx() not in atom_mapping:
            return None
        neighbor_res_idx = atom_mapping[neighbor.GetIdx()]

        # Signal A: the matched residue neighbor carries an inter-residue
        # bond, so the missing SMILES atom was effectively replaced by it.
        if neighbor_res_idx in boundary_local_idxs:
            to_strip.append(idx)
            continue

        # Signal B: carboxyl -OH on a non-terminal amino acid.
        if (
            atom.GetSymbol() == "O"
            and neighbor.GetSymbol() == "C"
            and no_oxt
        ):
            has_double_o = any(
                b.GetOtherAtom(neighbor).GetSymbol() == "O"
                and b.GetBondType() == Chem.BondType.DOUBLE
                and b.GetOtherAtom(neighbor).GetIdx() != idx
                for b in neighbor.GetBonds()
            )
            if has_double_o:
                to_strip.append(idx)
                continue

        return None

    if not to_strip:
        return None

    edit_mol = Chem.RWMol(rmol_smi)
    for idx in sorted(to_strip, reverse=True):
        edit_mol.RemoveAtom(idx)

    try:
        return Chem.MolToSmiles(edit_mol)
    except Exception:
        return None


_BONDTYPE_ORDER = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "ar": 1, "un": 1, "mc": 1}


def _detect_cross_residue_bonds(mol, selidx):
    """Return ``(cross_bonds, sel_start, sel_end)`` for covalent bonds linking
    the selected residue to the rest of ``mol``.

    ``cross_bonds`` is a list of ``(local_residue_idx, external_global_idx,
    bondtype_str)``. Detects bonds already present in ``mol.bonds`` that cross
    the residue boundary, plus peptide (N-C) and nucleic-acid (P-O3') bonds
    inferred by proximity when ``mol.bonds`` lacks them. Pure read; ``mol`` is
    not mutated.
    """
    import numpy as np

    sel_start = int(selidx[0])
    sel_end = int(selidx[-1])
    cross_bonds = []  # (local_residue_idx, external_global_idx, bondtype_str)
    if len(mol.bonds):
        in_sel = (mol.bonds >= sel_start) & (mol.bonds <= sel_end)
        cross_mask = in_sel.sum(axis=1) == 1
        for bidx in np.where(cross_mask)[0]:
            a, b = int(mol.bonds[bidx, 0]), int(mol.bonds[bidx, 1])
            bt = str(mol.bondtype[bidx]) if len(mol.bondtype) > bidx else "1"
            if sel_start <= a <= sel_end:
                cross_bonds.append((a - sel_start, b, bt))
            else:
                cross_bonds.append((b - sel_start, a, bt))

    # PDB inputs often arrive without explicit peptide / phosphodiester bonds
    # (CONECT records cover only HET groups). Without them, the boundary
    # atoms look free-standing and AddHs over-protonates them. Mirror the
    # proximity check used by ``_has_peptide_neighbour`` in
    # nonstandard_residues so a caller doesn't have to run ``mol.guessBonds``
    # first.
    sel_names = mol.name[selidx]
    sel_elems = mol.element[selidx]

    def _add_cross_by_proximity(local_idx, other_mask_full, threshold):
        if local_idx in {li for li, _, _ in cross_bonds}:
            return
        own_pos = mol.coords[selidx[local_idx], :, mol.frame]
        other_mask = other_mask_full.copy()
        other_mask[selidx] = False
        candidates = np.where(other_mask)[0]
        if not len(candidates):
            return
        d = np.linalg.norm(
            mol.coords[candidates, :, mol.frame] - own_pos, axis=1
        )
        within = d < threshold
        if not within.any():
            return
        partner = int(candidates[np.argmin(np.where(within, d, np.inf))])
        cross_bonds.append((local_idx, partner, "1"))

    # Peptide N-C bonds (~1.33 A, threshold 1.6 A)
    if {"N", "CA", "C"}.issubset(sel_names):
        for own_side, other_name in (("N", "C"), ("C", "N")):
            hits = np.where(sel_names == own_side)[0]
            if len(hits):
                _add_cross_by_proximity(
                    int(hits[0]), mol.name == other_name, threshold=1.6
                )

    # Nucleic acid phosphodiester P-O3' bonds (~1.6 A, threshold 1.8 A).
    # Two directions: (1) own P to external O3' of previous residue,
    # (2) own O3' to external P of next residue. The O3' check also runs for
    # 5'-terminal residues that lack their own P but still bond to next.
    if "P" in sel_elems:
        for own_p_idx in np.where(sel_elems == "P")[0]:
            _add_cross_by_proximity(
                int(own_p_idx), mol.element == "O", threshold=1.8
            )
    for own_o3_idx in np.where(sel_names == "O3'")[0]:
        _add_cross_by_proximity(
            int(own_o3_idx), mol.element == "P", threshold=1.8
        )

    return cross_bonds, sel_start, sel_end


def _apply_template_mapping(
    mol,
    selidx,
    sel_start,
    sel_end,
    residue,
    rmol,
    ref_rdkit,
    at1,
    at2,
    atom_mapping,
    cross_bonds,
    addHs,
    onlyOnAtoms,
    _logger,
):
    """Apply a built atom mapping (residue RDKit mol ``rmol`` <-> template RDKit
    mol ``ref_rdkit``) to ``mol`` in place.

    Transfers formal charges and bond orders through the mapping, optionally
    adds hydrogens (with cross-residue boundary corrections), round-trips the
    templated residue back into ``mol``, and restores inter-residue bonds.
    ``at1``/``at2`` are matched atom-index lists into ``rmol``/``ref_rdkit``;
    ``atom_mapping`` maps ``ref_rdkit`` atom idx -> ``rmol`` atom idx.
    """
    from rdkit import Chem
    from moleculekit.molecule import Molecule
    import numpy as np

    # Transfer the formal charge, bonds and bond orders from ref_rdkit to rmol
    for i, j in zip(at1, at2):
        fch = ref_rdkit.GetAtomWithIdx(j).GetFormalCharge()
        rmol.GetAtomWithIdx(i).SetFormalCharge(fch)

    # The template drives every matched bond's type, including DATIVE arrows
    # (e.g. [N]->[Fe]). Those round-trip through fromRDKitMol as "mc".
    for bond in ref_rdkit.GetBonds():
        i = atom_mapping[bond.GetBeginAtomIdx()]
        j = atom_mapping[bond.GetEndAtomIdx()]
        rbond = rmol.GetBondBetweenAtoms(i, j)
        if rbond is None:
            raise RuntimeError(f"Bond between atoms {i} and {j} not found in residue")
        rbond.SetBondType(bond.GetBondType())

    # For atoms covalently bonded to other residues, the template assumes those
    # bonds are hydrogens. Reduce the explicit H count on each boundary atom by
    # the order of its external bonds so AddHs doesn't over-protonate. Bonds
    # whose external partner is a metal element are tracked separately:
    # coordinating an -SH / -OH / -COOH to a metal is chemically a
    # deprotonation, so the donor's formal charge is reduced by one per
    # metal-bond order to keep the residue's net charge correct.
    rmol_to_smi = dict(zip(at1, at2))
    boundary_ext_order = {}
    boundary_metal_order = {}
    for local_idx, ext_global_idx, bt in cross_bonds:
        order = _BONDTYPE_ORDER.get(bt, 1)
        boundary_ext_order[local_idx] = boundary_ext_order.get(local_idx, 0) + order
        if str(mol.element[ext_global_idx]) in _METAL_ELEMENTS:
            boundary_metal_order[local_idx] = (
                boundary_metal_order.get(local_idx, 0) + order
            )

    # Protonate the residue according to the template
    if addHs:
        if onlyOnAtoms is not None:
            onlyOnAtoms = residue.atomselect(onlyOnAtoms, indexes=True).tolist()

        # Don't sanitize to not lose bond orders
        rmol = Chem.RemoveHs(rmol, sanitize=True)

        # Boundary H counts must be set AFTER RemoveHs: RemoveHs increments
        # NumExplicitHs by the number of removed H neighbors, so applying
        # this before would double-count.
        for local_idx, ext_order in boundary_ext_order.items():
            if local_idx not in rmol_to_smi:
                continue
            smi_atom = ref_rdkit.GetAtomWithIdx(rmol_to_smi[local_idx])
            smi_hs = int(smi_atom.GetTotalNumHs())
            target_hs = max(0, smi_hs - int(ext_order))
            atom = rmol.GetAtomWithIdx(local_idx)
            atom.SetNumExplicitHs(target_hs)
            atom.SetNoImplicit(True)
            metal_order = boundary_metal_order.get(local_idx, 0)
            if metal_order:
                hs_stripped = smi_hs - target_hs
                delta = min(metal_order, hs_stripped)
                if delta:
                    atom.SetFormalCharge(atom.GetFormalCharge() - delta)

        rmol = Chem.AddHs(rmol, addCoords=True, onlyOnAtoms=onlyOnAtoms)
        Chem.Kekulize(rmol)  # Sanitization ruins kekulization

    new_residue = Molecule.fromRDKitMol(rmol)
    new_residue.resid[:] = residue.resid[0]
    new_residue.chain[:] = residue.chain[0]
    new_residue.segid[:] = residue.segid[0]
    new_residue.insertion[:] = residue.insertion[0]

    # Restore per-atom metadata that doesn't round-trip through RDKit
    # (beta, occupancy, record, altloc) by matching atoms back by name.
    orig_by_name = {}
    for orig_idx, name in enumerate(residue.name):
        if name in orig_by_name:
            orig_by_name[name] = None  # ambiguous; skip
        else:
            orig_by_name[name] = orig_idx
    for new_idx, name in enumerate(new_residue.name):
        orig_idx = orig_by_name.get(name)
        if orig_idx is None:
            continue
        for field in ("beta", "occupancy", "record", "altloc"):
            getattr(new_residue, field)[new_idx] = getattr(residue, field)[orig_idx]

    # If the residue is the entire molecule, ``mol.remove`` empties it and the
    # subsequent ``mol.insert`` adopts ``new_residue``'s zero-initialised box.
    # Snapshot them so we can restore the simulation cell after the round-trip.
    saved_box = mol.box.copy() if mol.box is not None else None
    saved_boxangles = mol.boxangles.copy() if mol.boxangles is not None else None

    mol.remove(selidx, _logger=False)
    mol.insert(new_residue, selidx[0])

    if saved_box is not None and np.any(saved_box):
        mol.box = saved_box
    if saved_boxangles is not None and np.any(saved_boxangles):
        mol.boxangles = saved_boxangles

    # Restore the inter-residue bonds. Atoms originally past the residue were
    # shifted by (new_residue.numAtoms - len(selidx)); atoms before are
    # unchanged.
    if cross_bonds:
        shift = new_residue.numAtoms - len(selidx)
        for local_idx, ext_global_idx, bt in cross_bonds:
            if ext_global_idx < sel_start:
                new_ext = ext_global_idx
            elif ext_global_idx > sel_end:
                new_ext = ext_global_idx + shift
            else:
                continue
            mol.addBond(sel_start + local_idx, new_ext, bt)


def template_residue_from_smiles(
    mol: Molecule,
    sel: str,
    smiles: str,
    sanitizeSmiles: bool = True,
    addHs: bool = False,
    onlyOnAtoms: str | None = None,
    guessBonds: bool = False,
    _logger: bool = True,
):
    """Assign bonds, bond orders, formal charges and (optionally) hydrogens
    to a residue from a SMILES template.

    See :meth:`moleculekit.molecule.Molecule.templateResidueFromSmiles` for
    the full description; this is the underlying implementation. The
    ``mol`` argument is mutated in place.

    Parameters
    ----------
    mol : Molecule
        The molecule containing the residue(s) to template. Mutated in
        place.
    sel : str or numpy.ndarray
        VMD-style atom selection or boolean mask. May span multiple
        residues with the same chemistry; each residue is templated in
        sequence with the same SMILES.
    smiles : str
        SMILES string of the template residue. RCSB-style (fully
        protonated, explicit charges, full heavy-atom set) works best.
    sanitizeSmiles : bool
        If True, sanitize the SMILES with RDKit before matching.
    addHs : bool
        If True, add hydrogens after bond orders are transferred. Boundary
        atoms (those involved in cross-residue covalent bonds) have their
        explicit H count reduced by the order of the external bond so they
        are not over-protonated.
    onlyOnAtoms : str
        VMD-style selection within the residue restricting which heavy
        atoms get hydrogens added. Only used when ``addHs=True``.
    guessBonds : bool
        If True, run distance-based bond guessing on the residue before
        templating. Use when ``mol.bonds`` is empty.

    Raises
    ------
    RuntimeError
        If the selection is empty, has gaps in atom indexes, contains
        multiple residues with conflicting metadata, has no bonds, or the
        SMILES contains heavy atoms that cannot be matched (even after
        stripping recognized terminal atoms).

    Examples
    --------
    >>> mol = Molecule("3ptb")
    >>> mol.templateResidueFromSmiles("resname BEN", "[NH2+]=C(N)c1ccccc1", addHs=True)
    >>> mol.templateResidueFromSmiles("resname GLY and resid 18", "C(C(=O))N", addHs=True)
    """
    from rdkit.Chem import rdFMCS
    from rdkit import Chem
    import numpy as np

    selidx = mol.atomselect(sel, indexes=True)
    if len(selidx) == 0:
        raise RuntimeError(
            f"Selection {sel!r} matched no atoms; nothing to template."
        )

    # Selections that span multiple residues (e.g. ``resname MLE`` when
    # the same resname appears at three different resids, or three
    # contiguous NAG residues) get templated one residue at a time with
    # the same SMILES. Identities are snapshotted up front because
    # templating mutates ``mol`` (removes the matched atoms and inserts
    # a re-protonated copy), which would invalidate any cached atom
    # indexes for the later residues.
    keys = list(
        dict.fromkeys(
            zip(
                mol.resid[selidx],
                mol.insertion[selidx],
                mol.chain[selidx],
                mol.segid[selidx],
            )
        )
    )
    if len(keys) > 1:
        for resid, insertion, chain, segid in keys:
            mask = (
                (mol.resid == resid)
                & (mol.insertion == insertion)
                & (mol.chain == chain)
                & (mol.segid == segid)
            )
            template_residue_from_smiles(
                mol,
                mask,
                smiles,
                sanitizeSmiles=sanitizeSmiles,
                addHs=addHs,
                onlyOnAtoms=onlyOnAtoms,
                guessBonds=guessBonds,
                _logger=_logger,
            )
        return
    if not np.all(np.diff(selidx) == 1):
        raise RuntimeError(
            "The selection contains gaps in the atom indexes. Please select a single molecule residue only."
        )

    cross_bonds, sel_start, sel_end = _detect_cross_residue_bonds(mol, selidx)

    residue = mol.copy(sel=selidx)
    if guessBonds:
        residue.guessBonds()
    for field in ("resname", "chain", "segid", "resid", "insertion"):
        if len(set(getattr(residue, field))) != 1:
            raise RuntimeError(
                f"The selection contains multiple {field}s. Please select a single molecule residue only."
            )

    if len(residue.bonds) == 0:
        raise RuntimeError(
            "The selection contains no bonds. Please set the bonds of the residue or guess them with guessBonds=True"
        )

    # _logger=False here always: the SMILES echoed by toRDKitMol is the
    # residue's pre-template SMILES, which is noise inside templateResidueFromSmiles
    # (the caller already knows the target SMILES they passed in).
    rmol = residue.toRDKitMol(
        sanitize=False, kekulize=False, assignStereo=False, _logger=False
    )
    rmol_smi = Chem.MolFromSmiles(smiles, sanitize=sanitizeSmiles)
    if rmol_smi is None:
        raise RuntimeError(
            f"RDKit failed to parse the SMILES template '{smiles}'. "
            "Check the RDKit parser error printed above for the offending position."
        )
    rmol_smi = Chem.RemoveAllHs(rmol_smi)
    if rmol_smi is None:
        raise RuntimeError(
            f"RDKit failed to remove hydrogens from the SMILES template '{smiles}'."
        )
    Chem.Kekulize(rmol_smi)

    # FindMCS with CompareAny finds the MCS regardless of bond types, but
    # ``res.smartsString`` encodes specific bond constraints (e.g. ``-,->``)
    # that DATIVE bonds don't roundtrip through ``GetSubstructMatch``
    # cleanly. ``res.queryMol`` is a true wildcard query (atoms ``*``,
    # bond types ``UNSPECIFIED``) and matches both sides regardless of
    # their bond types — no demote needed.
    res = rdFMCS.FindMCS(
        [rmol, rmol_smi],
        bondCompare=rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
    )
    patt = res.queryMol
    at1 = list(rmol.GetSubstructMatch(patt))
    at2 = list(rmol_smi.GetSubstructMatch(patt))
    atom_mapping = {j: i for i, j in zip(at1, at2)}

    # Check if any atoms in rmol_smi which are not in at2 are non-hydrogen atoms
    elem = np.array([a.GetSymbol() for a in rmol_smi.GetAtoms()])
    non_matched = np.setdiff1d(range(len(elem)), at2)
    if np.any(elem[non_matched] != "H"):
        unmatched_heavy_idx = non_matched[elem[non_matched] != "H"]
        boundary_local_idxs = {local_idx for local_idx, _, _ in cross_bonds}
        modified_smiles = _try_strip_unmatched_terminals(
            rmol_smi, unmatched_heavy_idx, residue, atom_mapping, boundary_local_idxs
        )
        if modified_smiles is not None:
            if _logger:
                logger.info(
                    f"Stripped unmatched terminal heavy atoms from SMILES "
                    f"template (e.g. leaving group displaced by a covalent "
                    f"link, or carboxyl -OH on a non-terminal amino acid). "
                    f"Modified SMILES: '{modified_smiles}'"
                )
            rmol_smi = Chem.MolFromSmiles(modified_smiles, sanitize=sanitizeSmiles)
            if rmol_smi is None:
                raise RuntimeError(
                    f"RDKit failed to parse the SMILES template '{modified_smiles}' "
                    f"(derived from '{smiles}' after stripping unmatched terminal atoms)."
                )
            Chem.Kekulize(rmol_smi)
            res = rdFMCS.FindMCS(
                [rmol, rmol_smi],
                bondCompare=rdFMCS.BondCompare.CompareAny,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
            )
            patt = res.queryMol
            at1 = list(rmol.GetSubstructMatch(patt))
            at2 = list(rmol_smi.GetSubstructMatch(patt))
            atom_mapping = {j: i for i, j in zip(at1, at2)}

            elem = np.array([a.GetSymbol() for a in rmol_smi.GetAtoms()])
            non_matched = np.setdiff1d(range(len(elem)), at2)
            if np.any(elem[non_matched] != "H"):
                raise RuntimeError(
                    f"The SMILES template '{smiles}' contains heavy atoms which could not be matched to the residue. "
                    "Even after stripping the recognized terminal atoms, some heavy atoms could not be matched."
                )
        else:
            raise RuntimeError(
                f"The SMILES template '{smiles}' contains heavy atoms which could not be matched to the residue. "
                "If the SMILES describes a fragment that loses atoms in the structure (e.g. a leaving group "
                "displaced by a covalent bond, or the OXT of a mid-chain amino acid), either drop those atoms "
                "from the SMILES or ensure the corresponding inter-residue bond is present in mol.bonds (call "
                "mol.guessBonds() before templating, or rely on LINK/CONECT records)."
            )

    _apply_template_mapping(
        mol,
        selidx,
        sel_start,
        sel_end,
        residue,
        rmol,
        rmol_smi,
        at1,
        at2,
        atom_mapping,
        cross_bonds,
        addHs,
        onlyOnAtoms,
        _logger,
    )


def template_residue_from_molecule(
    mol: Molecule,
    sel,
    refmol: Molecule,
    addHs: bool = False,
    onlyOnAtoms=None,
    guessBonds: bool = False,
    _logger: bool = True,
):
    """Assign bonds, bond orders, formal charges and (optionally) hydrogens to a
    residue from a reference :class:`Molecule` template, matched by atom name.

    Mirrors :func:`template_residue_from_smiles` but the template source is a
    reference Molecule (e.g. ``Molecule("LIG.cif")``) that already carries
    ``bonds``, ``bondtype`` and ``formalcharge``. Heavy atoms of the selected
    residue are mapped onto the reference by NAME (not MCS); bond orders and
    formal charges are transferred through that mapping. The reference is used
    only as a template and is never appended to ``mol``. ``mol`` is mutated in
    place.

    The residue and the reference must have the same set of heavy-atom names.

    Raises
    ------
    RuntimeError
        If the selection is empty/gapped/multi-residue, has no bonds, the
        reference has duplicate heavy-atom names, or the residue and reference
        heavy-atom names do not match.
    """
    from rdkit import Chem
    import numpy as np

    selidx = mol.atomselect(sel, indexes=True)
    if len(selidx) == 0:
        raise RuntimeError(f"Selection {sel!r} matched no atoms; nothing to template.")

    # Multi-residue dispatch: template each copy individually (indexes are
    # snapshotted because templating mutates mol). Mirrors the SMILES variant.
    keys = list(
        dict.fromkeys(
            zip(
                mol.resid[selidx],
                mol.insertion[selidx],
                mol.chain[selidx],
                mol.segid[selidx],
            )
        )
    )
    if len(keys) > 1:
        for resid, insertion, chain, segid in keys:
            mask = (
                (mol.resid == resid)
                & (mol.insertion == insertion)
                & (mol.chain == chain)
                & (mol.segid == segid)
            )
            template_residue_from_molecule(
                mol,
                mask,
                refmol,
                addHs=addHs,
                onlyOnAtoms=onlyOnAtoms,
                guessBonds=guessBonds,
                _logger=_logger,
            )
        return
    if not np.all(np.diff(selidx) == 1):
        raise RuntimeError(
            "The selection contains gaps in the atom indexes. Please select a single molecule residue only."
        )

    cross_bonds, sel_start, sel_end = _detect_cross_residue_bonds(mol, selidx)

    residue = mol.copy(sel=selidx)
    if guessBonds:
        residue.guessBonds()
    for field in ("resname", "chain", "segid", "resid", "insertion"):
        if len(set(getattr(residue, field))) != 1:
            raise RuntimeError(
                f"The selection contains multiple {field}s. Please select a single molecule residue only."
            )
    if len(residue.bonds) == 0:
        raise RuntimeError(
            "The selection contains no bonds. Please set the bonds of the residue or guess them with guessBonds=True"
        )

    # Build the residue and reference RDKit mols. The residue keeps its Hs (the
    # AddHs tail strips/re-adds them); the reference is reduced to heavy atoms so
    # the name mapping is over heavy atoms only.
    rmol = residue.toRDKitMol(
        sanitize=False, kekulize=False, assignStereo=False, _logger=False
    )
    ref_rdkit = refmol.toRDKitMol(
        sanitize=False, kekulize=False, assignStereo=False, _logger=False
    )
    ref_rdkit = Chem.RemoveAllHs(ref_rdkit)
    Chem.Kekulize(ref_rdkit)

    # Map residue heavy atoms onto the reference heavy atoms by NAME. RDKit atom
    # order follows Molecule atom order, so the residue's i-th atom is rmol atom
    # i, and the reference's k-th heavy atom is ref_rdkit atom k.
    res_heavy_mask = residue.element != "H"
    res_heavy_rdkit_idx = np.where(res_heavy_mask)[0]
    res_heavy_names = residue.name[res_heavy_mask]
    ref_heavy_names = refmol.name[refmol.element != "H"]

    ref_name_to_idx = {}
    for idx, nm in enumerate(ref_heavy_names):
        if nm in ref_name_to_idx:
            raise RuntimeError(
                f"Reference template has duplicate atom name {nm!r}; atom-name "
                "templating requires unique heavy-atom names."
            )
        ref_name_to_idx[nm] = idx

    at1, at2 = [], []  # rmol atom idx, ref_rdkit atom idx
    for rdkit_idx, nm in zip(res_heavy_rdkit_idx, res_heavy_names):
        if nm not in ref_name_to_idx:
            raise RuntimeError(
                f"Residue atom {nm!r} has no same-named atom in the reference "
                "template; cannot map by name. Provide a matching reference or "
                "use a SMILES template instead."
            )
        at1.append(int(rdkit_idx))
        at2.append(ref_name_to_idx[nm])

    if len(at2) != len(ref_heavy_names):
        extra = sorted(set(ref_heavy_names) - set(res_heavy_names))
        raise RuntimeError(
            f"The reference template has heavy atoms not present in the residue: "
            f"{extra}. Atom-name templating requires the reference and residue to "
            "have the same heavy-atom names."
        )

    atom_mapping = {j: i for i, j in zip(at1, at2)}

    _apply_template_mapping(
        mol,
        selidx,
        sel_start,
        sel_end,
        residue,
        rmol,
        ref_rdkit,
        at1,
        at2,
        atom_mapping,
        cross_bonds,
        addHs,
        onlyOnAtoms,
        _logger,
    )


def extend_residue_from_smiles(
    mol: Molecule,
    sel: str,
    extension_smiles: str | None = None,
    target_atom_sel: str | None = None,
    new_smiles: str | None = None,
    sanitizeSmiles: bool = True,
    minimize: bool = False,
    _logger: bool = True,
):
    """
    Extends a residue by attaching new atoms, preserving original 3D coordinates
    and relaxing new atoms via constrained embedding and MMFF minimization.

    Two modes are supported (mutually exclusive):

    1. Extension SMILES mode: provide ``extension_smiles`` (containing a dummy
       atom ``*`` at the attachment point) together with ``target_atom_sel``
       (the atom on the residue to attach to).

    2. New SMILES mode: provide ``new_smiles`` with the complete SMILES of the
       modified molecule. The function uses Maximum Common Substructure (MCS)
       matching to identify unchanged atoms and generates 3D coordinates for
       the new ones.

    Parameters
    ----------
    mol : Molecule
        The molecule containing the residue to extend
    sel : str
        Atom selection for the residue to extend
    extension_smiles : str, optional
        SMILES of the extension moiety with a dummy atom ``*``
    target_atom_sel : str, optional
        Atom selection of the attachment point (required with extension_smiles)
    new_smiles : str, optional
        Complete SMILES of the modified molecule (alternative to extension_smiles)
    sanitizeSmiles : bool
        If True the SMILES string will be sanitized
    minimize : bool
        If True and OpenMM is available, run a soft-potential energy
        minimization of the residue against its surroundings after insertion.

    Examples
    --------
    >>> mol = Molecule('3ptb')
    >>> mol.templateResidueFromSmiles("resname BEN", "[NH2+]=C(N)c1ccccc1", addHs=True)
    >>> mol.extendResidueFromSmiles("resname BEN", extension_smiles="*C(C)(C)C", target_atom_sel="resname BEN and name H6")

    Or equivalently using the full modified SMILES:

    >>> mol = Molecule('3ptb')
    >>> mol.templateResidueFromSmiles("resname BEN", "[NH2+]=C(N)c1ccccc1", addHs=True)
    >>> mol.extendResidueFromSmiles("resname BEN", new_smiles="[NH2+]=C(N)c1cc(C(C)(C)C)ccc1")
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    import numpy as np

    if extension_smiles is not None and new_smiles is not None:
        raise ValueError(
            "Cannot specify both 'extension_smiles' and 'new_smiles'. Use one or the other."
        )
    if extension_smiles is None and new_smiles is None:
        raise ValueError(
            "Must specify either 'extension_smiles' (with 'target_atom_sel') or 'new_smiles'."
        )
    if extension_smiles is not None and target_atom_sel is None:
        raise ValueError("'target_atom_sel' is required when using 'extension_smiles'.")

    selidx = mol.atomselect(sel, indexes=True)

    if extension_smiles is not None:
        target_atom_idx = mol.atomselect(target_atom_sel, indexes=True)
        if len(target_atom_idx) != 1:
            raise RuntimeError(
                f"The target atom selection contains multiple atoms {target_atom_idx}. Please select a single atom only."
            )
        target_atom_idx = target_atom_idx[0]
        if target_atom_idx not in selidx:
            raise RuntimeError(
                f"The target atom {target_atom_idx} is not in the selection {sel}. Please select a single atom only."
            )
        target_atom_idx = int(np.where(selidx == target_atom_idx)[0][0])

    if not np.all(np.diff(selidx) == 1):
        raise RuntimeError(
            "The selection contains gaps in the atom indexes. Please select a single molecule residue only."
        )

    residue = mol.copy(sel=selidx)
    if residue.numFrames > 1:
        raise RuntimeError(
            "The molecule has multiple frames. This operation is not supported for multiple frames. Please use Molecule.dropFrames(keep=[0]) to keep a single frame"
        )

    for field in ("resname", "chain", "segid", "resid", "insertion"):
        if len(set(getattr(residue, field))) != 1:
            raise RuntimeError(
                f"The selection contains multiple {field}s. Please select a single molecule residue only."
            )

    if len(residue.bonds) == 0:
        if extension_smiles is not None:
            raise RuntimeError(
                "The selection contains no bonds. In extension_smiles mode "
                "the residue's bond orders are carried into the output, so "
                "guessed single bonds would corrupt the result. Run "
                "Molecule.templateResidueFromSmiles on the residue first to "
                "assign correct bond orders, or use new_smiles mode instead "
                "(which takes bond orders from the SMILES and skips the "
                "templating step)."
            )
        if _logger:
            logger.info(
                "The selection contains no bonds. Guessing them from coordinates."
            )
        residue.guessBonds()

    if extension_smiles is not None:
        # --- Extension SMILES mode ---
        if residue.element[target_atom_idx] == "H":
            heavy = residue.getNeighbors(target_atom_idx)
            residue.remove(target_atom_idx, _logger=False)
            target_atom_idx = int(heavy[0])

        base_mol = residue.toRDKitMol(
            sanitize=False, kekulize=False, assignStereo=False, _logger=_logger
        )
        base_num_atoms = base_mol.GetNumAtoms()

        ext_mol = Chem.MolFromSmiles(extension_smiles, sanitize=sanitizeSmiles)
        if ext_mol is None:
            raise RuntimeError(
                f"RDKit failed to parse the extension SMILES '{extension_smiles}'."
            )
        ext_mol = Chem.AddHs(ext_mol)
        Chem.Kekulize(ext_mol)

        dummy_idx = -1
        ext_attach_idx = -1
        bond_order = Chem.rdchem.BondType.SINGLE
        for atom in ext_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx = atom.GetIdx()
                dummy_bond = atom.GetBonds()[0]
                bond_order = dummy_bond.GetBondType()
                ext_attach_idx = dummy_bond.GetOtherAtomIdx(dummy_idx)
                break

        if dummy_idx == -1:
            raise ValueError("Extension SMILES must contain a dummy atom '*'.")

        combined_mol = Chem.CombineMols(base_mol, ext_mol)
        rw_mol = Chem.RWMol(combined_mol)

        new_dummy_idx = dummy_idx + base_num_atoms
        new_ext_attach_idx = ext_attach_idx + base_num_atoms

        rw_mol.AddBond(target_atom_idx, new_ext_attach_idx, order=bond_order)
        rw_mol.RemoveAtom(new_dummy_idx)

        final_mol = rw_mol.GetMol()
        Chem.SanitizeMol(final_mol)

        conf = base_mol.GetConformer()
        coord_map = {}
        for i in range(base_num_atoms):
            coord_map[i] = conf.GetAtomPosition(i)

        alignment_map = [(i, i) for i in range(base_num_atoms)]
        fixed_indices = list(range(base_num_atoms))

    else:
        # --- SMILES mode (MCS-based on heavy atoms) ---
        from rdkit.Chem import rdFMCS

        base_mol = residue.toRDKitMol(
            sanitize=False, kekulize=False, assignStereo=False, _logger=_logger
        )

        new_mol_noh = Chem.MolFromSmiles(new_smiles, sanitize=sanitizeSmiles)
        if new_mol_noh is None:
            raise RuntimeError(
                f"RDKit failed to parse the SMILES template '{new_smiles}'."
            )
        Chem.Kekulize(new_mol_noh)

        # Match on heavy atoms only; including explicit Hs causes
        # combinatorial explosion because all H atoms share the same element.
        base_noh = Chem.RemoveHs(base_mol, sanitize=False)

        mcs_result = rdFMCS.FindMCS(
            [base_noh, new_mol_noh],
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            timeout=60,
        )
        patt = Chem.MolFromSmarts(mcs_result.smartsString)
        base_noh_matched = list(base_noh.GetSubstructMatch(patt))
        new_noh_matched = list(new_mol_noh.GetSubstructMatch(patt))

        if len(base_noh_matched) == 0:
            raise RuntimeError(
                "Could not find any common substructure between the residue "
                "and the new SMILES. Please check that the new SMILES is a "
                "modified version of the original molecule."
            )

        n_base_heavy = base_noh.GetNumAtoms()
        if len(base_noh_matched) < n_base_heavy * 0.5:
            logger.warning(
                f"Only {len(base_noh_matched)}/{n_base_heavy} heavy atoms "
                "matched between residue and new SMILES. The result may be "
                "unreliable."
            )

        # Map heavy-atom-only indices back to the full (with-H) molecules
        base_heavy_indices = [
            a.GetIdx() for a in base_mol.GetAtoms() if a.GetAtomicNum() != 1
        ]
        new_mol = Chem.AddHs(new_mol_noh)
        new_heavy_indices = [
            a.GetIdx() for a in new_mol.GetAtoms() if a.GetAtomicNum() != 1
        ]

        conf = base_mol.GetConformer()
        coord_map = {}
        matched_new_indices = []
        matched_base_indices = []
        for base_noh_idx, new_noh_idx in zip(base_noh_matched, new_noh_matched):
            base_orig_idx = base_heavy_indices[base_noh_idx]
            new_orig_idx = new_heavy_indices[new_noh_idx]
            coord_map[new_orig_idx] = conf.GetAtomPosition(base_orig_idx)
            matched_new_indices.append(new_orig_idx)
            matched_base_indices.append(base_orig_idx)

        final_mol = new_mol
        Chem.SanitizeMol(final_mol)

        alignment_map = list(zip(matched_new_indices, matched_base_indices))
        fixed_indices = matched_new_indices

    # Constrained embedding: generate 3D for new atoms, keep matched atoms fixed
    res = AllChem.EmbedMolecule(final_mol, coordMap=coord_map, randomSeed=42)
    if res == -1:
        raise RuntimeError(
            "RDKit failed to embed the molecule. The modification might be too constrained."
        )

    # Rigid-body alignment brings the whole molecule (including new atoms)
    # close to the original coordinate frame, so that the subsequent
    # coordinate snap only makes small corrections and new atoms remain
    # in physically reasonable positions for minimization.
    rdMolAlign.AlignMol(final_mol, base_mol, atomMap=alignment_map)

    # Snap matched atoms and their H neighbors to exact original coordinates.
    # EmbedMolecule treats coordMap as soft constraints and can alter
    # rotatable-bond dihedrals. Setting positions directly preserves the
    # original geometry exactly; MMFF then only relaxes the new atoms.
    # Extend alignment_map with H neighbors of matched heavy atoms
    already_matched_new = set(ni for ni, _ in alignment_map)
    already_matched_base = set(bi for _, bi in alignment_map)
    h_pairs = []
    for new_idx, base_idx in alignment_map:
        base_atom = base_mol.GetAtomWithIdx(base_idx)
        if base_atom.GetAtomicNum() == 1:
            continue
        base_hs = [
            n.GetIdx()
            for n in base_atom.GetNeighbors()
            if n.GetAtomicNum() == 1 and n.GetIdx() not in already_matched_base
        ]
        new_hs = [
            n.GetIdx()
            for n in final_mol.GetAtomWithIdx(new_idx).GetNeighbors()
            if n.GetAtomicNum() == 1 and n.GetIdx() not in already_matched_new
        ]
        for bh, nh in zip(base_hs, new_hs):
            h_pairs.append((nh, bh))

    alignment_map = alignment_map + h_pairs
    fixed_indices = fixed_indices + [nh for nh, _ in h_pairs]

    # Snap positions and preserve names for all matched atoms (heavy + H)
    final_conf = final_mol.GetConformer()
    base_conf = base_mol.GetConformer()
    for new_idx, base_idx in alignment_map:
        final_conf.SetAtomPosition(new_idx, base_conf.GetAtomPosition(base_idx))
        base_atom = base_mol.GetAtomWithIdx(base_idx)
        if base_atom.HasProp("_Name"):
            final_mol.GetAtomWithIdx(new_idx).SetProp(
                "_Name", base_atom.GetProp("_Name")
            )

    # MMFF minimization: freeze matched atoms, relax new atoms
    ff_props = AllChem.MMFFGetMoleculeProperties(final_mol)
    ff = AllChem.MMFFGetMoleculeForceField(final_mol, ff_props)

    if ff:
        for i in fixed_indices:
            ff.AddFixedPoint(i)
        ff.Minimize(maxIts=1000)
    else:
        logger.warning(
            "Could not set up MMFF force field. Molecule generated but clashes may exist."
        )

    Chem.Kekulize(final_mol)

    # Reorder: matched atoms first (in original base_mol order), then the rest
    sorted_pairs = sorted(alignment_map, key=lambda x: x[1])
    matched_ordered = [new_idx for new_idx, _base_idx in sorted_pairs]
    matched_set = set(matched_ordered)
    unmatched = [i for i in range(final_mol.GetNumAtoms()) if i not in matched_set]
    new_order = matched_ordered + unmatched
    final_mol = Chem.RenumberAtoms(final_mol, new_order)

    new_residue = Molecule.fromRDKitMol(final_mol)
    new_residue.resname[:] = residue.resname[0]
    new_residue.resid[:] = residue.resid[0]
    new_residue.chain[:] = residue.chain[0]
    new_residue.segid[:] = residue.segid[0]
    new_residue.insertion[:] = residue.insertion[0]

    mol.remove(selidx, _logger=False)
    mol.insert(new_residue, selidx[0])

    if minimize:
        from moleculekit.openmmtools import minimize_soft_potential

        # Only the newly-added atoms are mobile; original residue atoms
        # stay put so the extension relaxes against the existing geometry.
        n_matched = len(matched_ordered)
        new_atom_indices = set(
            range(selidx[0] + n_matched, selidx[0] + new_residue.numAtoms)
        )
        minimize_soft_potential(mol, new_atom_indices)
