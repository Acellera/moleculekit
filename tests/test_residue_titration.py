import numpy as np
import pytest
from rdkit import Chem
from moleculekit.molecule import Molecule, UniqueResidueID
from moleculekit.tools.nonstandard_residues import (
    ChainResidueSpec,
    CovalentLigandSpec,
    ScaffoldSpec,
    detectNonStandardResidues,
    getResidueMask,
)
from moleculekit.tools.residue_titration import (
    _cap_residue_smiles,
    _inter_residue_crosslinks,
    _isolated_residue_rdkit,
    _relaxed_query,
    _uncapped_residue_smiles,
)


def _single_residue_mol(resname, names, elements, coords):
    """Build a single-residue Molecule from explicit heavy-atom coordinates.

    Coordinates only need to be close enough to real amino-acid geometry for
    ``guessBonds()`` to recover the intended backbone connectivity; capping
    itself is coordinate-independent (RDKit valence assigns the hydrogens).
    """
    m = Molecule().empty(len(names))
    m.resname[:] = resname
    m.name[:] = names
    m.element[:] = elements
    m.resid[:] = 1
    m.record[:] = "ATOM"
    m.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    return m


def _dal_spec(mol, is_n_term=False, is_c_term=False):
    """Build a ChainResidueSpec for the (single) DAL residue in ``mol``."""
    rid = UniqueResidueID.fromMolecule(mol, idx=0)
    return ChainResidueSpec(
        resname="DAL", residue=rid, is_n_term=is_n_term, is_c_term=is_c_term
    )


def _dal_mol():
    # D-alanine backbone: N, CA, CB, C, O at amino-acid-like geometry so
    # guessBonds() recovers N-CA, CA-CB, CA-C, C=O.
    coords = [
        (-1.0, 1.0, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (0.3, -0.5, 1.4),  # CB
        (1.2, 0.6, -0.6),  # C
        (1.1, 1.8, -0.6),  # O
    ]
    return _single_residue_mol(
        "DAL", ["N", "CA", "CB", "C", "O"], ["N", "C", "C", "C", "O"], coords
    )


def test_get_residue_mask_selects_only_the_residue():
    mol = _dal_mol()
    spec = _dal_spec(mol)
    mask = getResidueMask(mol, spec)
    assert mask.dtype == bool
    assert mask.sum() == mol.numAtoms
    assert np.all(mask)


def _covalent_ligand_and_cys():
    """A two-residue fixture with a genuine non-peptide crosslink: a small
    non-canonical ``LIG`` residue whose last carbon is thioether-bonded
    (explicit ``mol.bonds``/``mol.bondtype`` entry, no distance guessing) to
    a free-standing ``CYS`` residue's ``SG``. Neither residue is peptide-
    bonded to anything, so the only inter-residue bond is the crosslink
    itself and ``detectNonStandardResidues`` reports ``LIG`` as a
    :class:`~moleculekit.tools.nonstandard_residues.CovalentLigandSpec`
    (exactly one non-peptide bond, not chain-resident).
    """
    names = ["C1", "C2", "C3"] + ["N", "CA", "CB", "SG", "C", "O"]
    elements = ["C", "C", "C"] + ["N", "C", "C", "S", "C", "O"]
    resnames = ["LIG"] * 3 + ["CYS"] * 6
    resids = [1] * 3 + [2] * 6
    coords = [
        (10.0, 0.0, 0.0),  # LIG C1
        (11.0, 0.0, 0.0),  # LIG C2
        (11.0, 1.0, 0.0),  # LIG C3 (bonded to CYS SG)
        (0.0, 0.0, 0.0),  # CYS N
        (1.0, 0.0, 0.0),  # CYS CA
        (1.5, 1.3, 0.0),  # CYS CB
        (3.0, 1.3, 0.0),  # CYS SG
        (1.5, -1.0, 0.5),  # CYS C
        (2.5, -1.5, 0.5),  # CYS O
    ]
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elements
    mol.resname[:] = resnames
    mol.resid[:] = resids
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.record[:] = ["HETATM"] * 3 + ["ATOM"] * 6
    mol.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    bonds = [
        (0, 1),
        (1, 2),  # LIG intra-residue: C1-C2, C2-C3
        (3, 4),
        (4, 5),
        (5, 6),
        (4, 7),
        (7, 8),  # CYS N-CA, CA-CB, CB-SG, CA-C, C=O
        (2, 6),  # the crosslink: LIG C3 - CYS SG
    ]
    mol.bonds = np.array(bonds, dtype=np.uint32)
    mol.bondtype = np.array(["1"] * len(bonds), dtype=object)
    return mol


def test_inter_residue_crosslinks_finds_nonpeptide_bond():
    # Build a two-residue mol: a LIG carbon bonded (thioether) to a CYS SG,
    # with explicit bonds so no guessing is needed.
    mol = _covalent_ligand_and_cys()
    specs = detectNonStandardResidues(mol)
    lig = next(s for s in specs if isinstance(s, (ScaffoldSpec, CovalentLigandSpec)))
    xs = _inter_residue_crosslinks(mol, lig)
    assert len(xs) == 1
    local, partner = xs[0]
    assert str(mol.resname[local]) == lig.resname
    assert str(mol.resname[partner]) != lig.resname  # crosses to the other residue


def _donor_with_metal_and_crosslink():
    """A ``RES`` residue whose ``OG`` coordinates the iron of a cofactor
    (a two-atom ``HEM``) via an explicit metal-coordination bond (the stored
    ``"mc"`` bond type) and whose ``CX`` carbon is genuinely thioether-
    crosslinked (bond type ``"1"``) to a free ``CYS`` ``SG``. Only the
    ``CX``-``SG`` bond is a real crosslink; the ``OG``-``Fe`` bond is
    coordination and is recognised as such purely from its ``"mc"`` type.
    """
    names = ["CA", "CB", "OG", "CX"] + ["FE", "C1"] + ["SG", "CB"]
    elements = ["C", "C", "O", "C"] + ["Fe", "C"] + ["S", "C"]
    resnames = ["RES"] * 4 + ["HEM"] * 2 + ["CYS"] * 2
    resids = [1] * 4 + [2] * 2 + [3] * 2
    coords = [(float(i), 0.0, 0.0) for i in range(len(names))]
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elements
    mol.resname[:] = resnames
    mol.resid[:] = resids
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.record[:] = "HETATM"
    mol.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    bonds = [
        (0, 1),  # RES CA-CB
        (1, 2),  # RES CB-OG
        (0, 3),  # RES CA-CX
        (2, 4),  # OG - HEM FE (metal coordination)
        (4, 5),  # HEM FE-C1 (intra-cofactor)
        (3, 6),  # RES CX - CYS SG (the genuine crosslink)
        (6, 7),  # CYS SG-CB
    ]
    mol.bonds = np.array(bonds, dtype=np.uint32)
    mol.bondtype = np.array(
        ["1", "1", "1", "mc", "1", "1", "1"], dtype=object
    )
    return mol


def test_inter_residue_crosslinks_excludes_metal_coordination():
    mol = _donor_with_metal_and_crosslink()
    rid = UniqueResidueID.fromMolecule(mol, idx=0)
    res = ChainResidueSpec(resname="RES", residue=rid)
    xs = _inter_residue_crosslinks(mol, res)
    # Only the covalent CX-SG crosslink is returned; the OG-Fe bond is excluded
    # because it carries the stored "mc" (metal-coordination) bond type.
    assert len(xs) == 1
    local, partner = xs[0]
    assert str(mol.name[local]) == "CX"
    assert str(mol.resname[partner]) == "CYS"
    # No returned bond touches the iron atom.
    fe_idx = int(np.where(mol.element == "Fe")[0][0])
    assert all(fe_idx not in pair for pair in xs)


def test_cap_midchain_adds_ace_and_nme():
    mol = _dal_mol()
    spec = _dal_spec(mol, is_n_term=False, is_c_term=False)
    capped = _cap_residue_smiles(mol, spec, "C[C@H](C(=O)O)N")
    m = Chem.MolFromSmiles(capped)
    assert m is not None
    # Mid-chain cap = acetyl (adds one extra terminal methyl-carbonyl) + N-methyl
    # amide. Count amide bonds: two backbone-facing amides (ACE-N and C-NME).
    amide = Chem.MolFromSmarts("[CX3](=O)[NX3]")
    assert len(m.GetSubstructMatches(amide)) == 2
    # No free carboxylic acid remains (C side is capped as an amide).
    acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
    assert len(m.GetSubstructMatches(acid)) == 0


def test_cap_cterm_leaves_free_acid():
    mol = _dal_mol()
    spec = _dal_spec(mol, is_n_term=False, is_c_term=True)
    capped = _cap_residue_smiles(mol, spec, "C[C@H](C(=O)O)N")
    m = Chem.MolFromSmiles(capped)
    assert m is not None
    acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
    assert len(m.GetSubstructMatches(acid)) == 1  # free acid kept on C side
    amide = Chem.MolFromSmarts("[CX3](=O)[NX3]")
    assert len(m.GetSubstructMatches(amide)) == 1  # only the ACE amide on N side


def test_cap_nterm_leaves_free_amine():
    mol = _dal_mol()
    spec = _dal_spec(mol, is_n_term=True, is_c_term=False)
    capped = _cap_residue_smiles(mol, spec, "C[C@H](C(=O)O)N")
    m = Chem.MolFromSmiles(capped)
    assert m is not None
    amide = Chem.MolFromSmarts("[CX3](=O)[NX3]")
    assert len(m.GetSubstructMatches(amide)) == 1  # only the C-NME amide
    amine = Chem.MolFromSmarts("[NX3;H2;!$(NC=O)]")
    assert len(m.GetSubstructMatches(amine)) == 1  # free primary amine kept on N side
    acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
    assert len(m.GetSubstructMatches(acid)) == 0  # C side is capped as an amide


def test_cap_nmethyl_backbone_sarcosine():
    # Sarcosine (N-methyl-glycine): backbone N carries a methyl (CN). Capping
    # must still attach ACE to the atom named N without error, giving a
    # tertiary amide (no N-H) on that side.
    coords = [
        (-1.0, 1.0, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.2, 0.6, -0.6),  # C
        (1.1, 1.8, -0.6),  # O
        (-2.039, 2.039, 0.0),  # CN (N-methyl carbon)
    ]
    mol = _single_residue_mol(
        "SAR", ["N", "CA", "C", "O", "CN"], ["N", "C", "C", "O", "C"], coords
    )
    rid = UniqueResidueID.fromMolecule(mol, idx=0)
    spec = ChainResidueSpec(
        resname="SAR", residue=rid, is_n_term=False, is_c_term=False
    )
    capped = _cap_residue_smiles(mol, spec, "CNCC(=O)O")
    m = Chem.MolFromSmiles(capped)
    assert m is not None
    amide = Chem.MolFromSmarts("[CX3](=O)[NX3]")
    assert len(m.GetSubstructMatches(amide)) == 2


def _dal_mol_missing_backbone_atom(missing_name):
    """A D-alanine-like residue with no atom literally named ``missing_name``
    (``"N"`` or ``"C"``): that atom is renamed so it no longer matches, while
    keeping the same elements/geometry so bond-guessing and SMILES-template
    matching (which key off connectivity, not names) still succeed. This is
    the "uncappable backbone" case: :func:`_capped_residue_rdkit` cannot
    locate the backbone atom to cap from.
    """
    names = ["N", "CA", "CB", "C", "O"]
    elements = ["N", "C", "C", "C", "O"]
    coords = [
        (-1.0, 1.0, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (0.3, -0.5, 1.4),  # CB
        (1.2, 0.6, -0.6),  # C
        (1.1, 1.8, -0.6),  # O
    ]
    renamed = [f"{n}X" if n == missing_name else n for n in names]
    return _single_residue_mol("DAL", renamed, elements, coords)


def test_isolated_residue_rdkit_maps_backbone_atoms():
    # Direct test of _isolated_residue_rdkit's own contract: it returns
    # (rw, res_to_smi) where res_to_smi maps a structure residue's global
    # heavy-atom indices onto valid atom indices in the RWMol built from
    # base_smiles. Every caller (_capped_residue_rdkit's backbone-cap and
    # crosslink-cap branches) depends on that mapping resolving correctly, but
    # nothing exercised it directly before, only indirectly through those
    # higher-level cap/strip functions.
    mol = _dal_mol()
    spec = _dal_spec(mol)
    base_smiles = "C[C@H](C(=O)O)N"
    rw, res_to_smi = _isolated_residue_rdkit(mol, spec, base_smiles)

    n_g = int(np.where(mol.name == "N")[0][0])
    c_g = int(np.where(mol.name == "C")[0][0])
    assert n_g in res_to_smi
    assert c_g in res_to_smi

    n_smi, c_smi = res_to_smi[n_g], res_to_smi[c_g]
    assert 0 <= n_smi < rw.GetNumAtoms()
    assert 0 <= c_smi < rw.GetNumAtoms()
    assert rw.GetAtomWithIdx(n_smi).GetSymbol() == "N"
    assert rw.GetAtomWithIdx(c_smi).GetSymbol() == "C"


@pytest.mark.parametrize("missing_name", ["N", "C"])
def test_isolated_residue_rdkit_raises_clear_error_on_missing_backbone_atom(
    missing_name,
):
    # The backbone N/C lookup now lives in _capped_residue_rdkit (reached via
    # _cap_residue_smiles), not in _isolated_residue_rdkit itself: that
    # function only builds the RWMol and the res_to_smi index map.
    mol = _dal_mol_missing_backbone_atom(missing_name)
    spec = _dal_spec(mol)
    with pytest.raises(ValueError, match="DAL"):
        _cap_residue_smiles(mol, spec, "C[C@H](C(=O)O)N")


def _substruct_matches(anchor_smiles, capped_smiles):
    """True if ``anchor_smiles`` substructure-matches ``capped_smiles`` using
    the same relaxed (bond-order-agnostic) query the downstream strip step
    uses to locate a residue's atoms inside its capped titration result.
    """
    from rdkit.Chem import rdmolops

    params = rdmolops.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    query = rdmolops.AdjustQueryProperties(Chem.MolFromSmiles(anchor_smiles), params)
    return bool(Chem.MolFromSmiles(capped_smiles).GetSubstructMatch(query))


@pytest.mark.parametrize(
    "is_n_term,is_c_term",
    [(False, False), (True, False), (False, True)],
    ids=["mid", "nterm", "cterm"],
)
def test_uncapped_residue_smiles_matches_capped_residue(is_n_term, is_c_term):
    # The real contract: _uncapped_residue_smiles is used downstream as a
    # relaxed substructure query to locate the residue's own atoms inside
    # the capped SMILES that _cap_residue_smiles produces for the same
    # spec, so the anchor must be a subgraph of that capped molecule in
    # every chain context - mid-chain and N-terminal (amide-capped C side,
    # one oxygen) just as much as C-terminal (free-acid C side, two
    # oxygens). Asserting acid-vs-aldehyde in isolation (the previous
    # version of this test) does not verify that; it let a regression that
    # always forced a free acid onto the anchor slip through, breaking the
    # match for mid-chain and N-terminal residues.
    mol = _dal_mol()
    spec = _dal_spec(mol, is_n_term=is_n_term, is_c_term=is_c_term)
    base_smiles = "C[C@H](C(=O)O)N"
    capped = _cap_residue_smiles(mol, spec, base_smiles)
    anchor = _uncapped_residue_smiles(mol, spec, base_smiles)
    assert _substruct_matches(anchor, capped) is True


def _dipeptide_with_ncaa(with_ligand=False):
    """ALA-DAL-ALA tripeptide: DAL (D-alanine, non-canonical) sits mid-chain,
    peptide-bonded on both sides via EXPLICIT ``mol.bonds``/``mol.bondtype``
    (no distance-based bond guessing), so ``detectNonStandardResidues``
    classifies it as a chain-resident, mid-chain ``ChainResidueSpec`` purely
    from real connectivity. The two flanking ALA residues are canonical and
    are skipped by the detector entirely (no spec emitted for them).

    Only the DAL residue's own coordinates need to look like a real residue
    (N-CA-CB / CA-C=O geometry): it is the only residue that goes through
    ``_cap_residue_smiles`` / ``_uncapped_residue_smiles``, which re-derive
    that single residue's internal bonds from its coordinates via
    ``guessBonds``. The flanking ALA placeholders never take that path, so
    their coordinates are arbitrary (just kept far apart so nothing spurious
    would be guessed if it ever were).

    When ``with_ligand`` is True, a free, unbonded "LIG" residue is appended
    on its own chain so the detector also emits a ``LigandSpec``.
    """
    names = ["N", "CA", "C", "O"] + ["N", "CA", "CB", "C", "O"] + ["N", "CA", "C", "O"]
    elements = ["N", "C", "C", "O"] + ["N", "C", "C", "C", "O"] + ["N", "C", "C", "O"]
    resnames = ["ALA"] * 4 + ["DAL"] * 5 + ["ALA"] * 4
    resids = [1] * 4 + [2] * 5 + [3] * 4
    coords = (
        [(-100.0, 0.0, 0.0), (-99.0, 0.0, 0.0), (-98.0, 0.0, 0.0), (-98.0, 1.0, 0.0)]
        + [
            (-1.0, 1.0, 0.0),  # N
            (0.0, 0.0, 0.0),  # CA
            (0.3, -0.5, 1.4),  # CB
            (1.2, 0.6, -0.6),  # C
            (1.1, 1.8, -0.6),  # O
        ]
        + [(98.0, 0.0, 0.0), (99.0, 0.0, 0.0), (100.0, 0.0, 0.0), (100.0, 1.0, 0.0)]
    )
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elements
    mol.resname[:] = resnames
    mol.resid[:] = resids
    mol.chain[:] = "A"
    mol.segid[:] = "P0"
    mol.record[:] = "ATOM"
    mol.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    bonds = [
        (0, 1), (1, 2), (2, 3),  # ALA1 N-CA, CA-C, C=O
        (4, 5), (5, 6), (5, 7), (7, 8),  # DAL2 N-CA, CA-CB, CA-C, C=O
        (9, 10), (10, 11), (11, 12),  # ALA3 N-CA, CA-C, C=O
        (2, 4), (7, 9),  # peptide bonds: ALA1.C-DAL2.N, DAL2.C-ALA3.N
    ]
    if with_ligand:
        lig = Molecule().empty(3)
        lig.name[:] = ["C", "C", "O"]
        lig.element[:] = ["C", "C", "O"]
        lig.resname[:] = "LIG"
        lig.resid[:] = 1
        lig.chain[:] = "L"
        lig.segid[:] = "L"
        lig.record[:] = "HETATM"
        lig.coords = np.array(
            [(50.0, 0.0, 0.0), (51.0, 0.0, 0.0), (51.0, 1.0, 0.0)], dtype=np.float32
        ).reshape(-1, 3, 1)
        mol.append(lig, collisions=False)
    mol.bonds = np.array(bonds, dtype=np.uint32)
    mol.bondtype = np.array(["1"] * len(bonds), dtype=object)
    return mol


def test_inter_residue_crosslinks_excludes_peptide_bond():
    # DAL2 sits mid-chain, peptide-bonded on both sides via explicit N-C
    # bonds to the flanking ALA residues. Those are the only inter-residue
    # bonds touching it, so the backbone-exclusion rule must leave nothing.
    mol = _dipeptide_with_ncaa()
    specs = detectNonStandardResidues(mol)
    dal = next(s for s in specs if isinstance(s, ChainResidueSpec) and s.resname == "DAL")
    assert _inter_residue_crosslinks(mol, dal) == []


def test_capForTitration_returns_capped_smiles_by_key(monkeypatch):
    mol = _dipeptide_with_ncaa()
    specs = detectNonStandardResidues(mol)
    from moleculekit.tools import residue_titration as rt
    from rdkit import Chem

    # Avoid network: stub the RCSB fetch.
    monkeypatch.setattr(rt, "rcsbFetchLigandSmiles", lambda c, **k: "C[C@H](C(=O)O)N")
    titration = rt.capNonstandardResiduesForTitration(mol, specs)
    # Single-context NCAA -> plain resname key; value is its capped titration SMILES.
    assert set(titration) == {"DAL"}
    # DAL is mid-chain here, so both backbone sides are capped as amides.
    m = Chem.MolFromSmiles(titration["DAL"])
    assert m is not None
    amide = Chem.MolFromSmarts("[CX3](=O)[NX3]")
    assert len(m.GetSubstructMatches(amide)) == 2


def test_capForTitration_requires_smiles(monkeypatch):
    mol = _dipeptide_with_ncaa()
    specs = detectNonStandardResidues(mol)
    from moleculekit.tools import residue_titration as rt

    def _raise(c, **k):
        raise RuntimeError("no RCSB entry")

    monkeypatch.setattr(rt, "rcsbFetchLigandSmiles", _raise)
    with pytest.raises(RuntimeError):
        rt.capNonstandardResiduesForTitration(mol, specs)  # no override supplied


def test_capForTitration_ligand_passthrough(monkeypatch):
    mol = _dipeptide_with_ncaa(with_ligand=True)
    specs = detectNonStandardResidues(mol)
    from moleculekit.tools import residue_titration as rt

    fetched = {"LIG": "c1ccccc1C(=O)O", "DAL": "C[C@H](C(=O)O)N"}
    monkeypatch.setattr(rt, "rcsbFetchLigandSmiles", lambda c, **k: fetched[c])
    titration = rt.capNonstandardResiduesForTitration(mol, specs)
    assert set(titration) == {"DAL", "LIG"}
    # Free ligands pass through uncapped: the titration SMILES is the raw fetch.
    assert titration["LIG"] == fetched["LIG"]


def test_capForTitration_falls_back_uncapped_on_uncappable_backbone(caplog):
    # A chain-resident spec whose residue has no atom literally named "C"
    # cannot be capped (_capped_residue_rdkit raises ValueError). This must
    # be caught and fall back to titrating the residue uncapped (context
    # "ligand"), with a warning naming the residue, rather than crashing or
    # silently swallowing an unrelated bug.
    mol = _dal_mol_missing_backbone_atom("C")
    spec = _dal_spec(mol)
    from moleculekit.tools import residue_titration as rt

    with caplog.at_level("WARNING"):
        titration = rt.capNonstandardResiduesForTitration(
            mol, [spec], smiles={"DAL": "C[C@H](C(=O)O)N"}
        )
    # Titrated whole: the value is the uncapped base SMILES, and a warning names it.
    assert titration["DAL"] == "C[C@H](C(=O)O)N"
    assert any("DAL" in r.message for r in caplog.records)


def test_capForTitration_does_not_swallow_unexpected_errors(monkeypatch):
    # A non-ValueError raised while capping (e.g. a genuine bug) must NOT be
    # treated as the uncappable-backbone fallback signal: it should propagate
    # instead of silently reverting the residue to uncapped titration.
    mol = _dal_mol()
    spec = _dal_spec(mol)
    from moleculekit.tools import residue_titration as rt

    def _boom(mol, spec, base_smiles):
        raise RuntimeError("genuine bug, not an uncappable backbone")

    monkeypatch.setattr(rt, "_cap_residue_smiles", _boom)
    with pytest.raises(RuntimeError, match="genuine bug"):
        rt.capNonstandardResiduesForTitration(
            mol, [spec], smiles={"DAL": "C[C@H](C(=O)O)N"}
        )


def test_capForTitration_dedup_same_key(monkeypatch):
    # Two separate mid-chain DAL occurrences (two independent tripeptides)
    # must collapse to a single manifest entry keyed "DAL".
    mol_a = _dipeptide_with_ncaa()
    mol_b = _dipeptide_with_ncaa()
    mol_b.chain[:] = "B"
    mol_b.segid[:] = "P1"
    mol = mol_a.copy()
    mol.append(mol_b, collisions=False)
    specs = detectNonStandardResidues(mol)
    dal_specs = [s for s in specs if s.resname == "DAL"]
    assert len(dal_specs) == 2  # sanity: detector really found both

    from moleculekit.tools import residue_titration as rt

    calls = []

    def _fetch(c, **k):
        calls.append(c)
        return "C[C@H](C(=O)O)N"

    monkeypatch.setattr(rt, "rcsbFetchLigandSmiles", _fetch)
    titration = rt.capNonstandardResiduesForTitration(mol, specs)

    assert list(titration).count("DAL") == 1
    assert calls.count("DAL") == 1  # fetched (and capped) only once


def test_capForTitration_override_skips_rcsb(monkeypatch):
    mol = _dipeptide_with_ncaa()
    specs = detectNonStandardResidues(mol)
    from moleculekit.tools import residue_titration as rt

    def _raise(c, **k):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(rt, "rcsbFetchLigandSmiles", _raise)
    titration = rt.capNonstandardResiduesForTitration(
        mol, specs, smiles={"DAL": "C[C@H](C(=O)O)N"}
    )
    assert set(titration) == {"DAL"}


def test_capForTitration_skips_renamed_canonical_crosslink(monkeypatch):
    # A disulfide-bonded CYS renamed to CYX (or any canonical AA renamed at a
    # non-peptide junction) keeps its canonical resname and is already
    # handled by the force field: it must never be templated. Only the
    # genuine NCAA (DAL) should reach the manifest, and rcsbFetchLigandSmiles
    # must never be called for CYS/CYX.
    mol = _dipeptide_with_ncaa()
    specs = detectNonStandardResidues(mol)
    dal_spec = next(s for s in specs if s.resname == "DAL")
    cyx_spec = ChainResidueSpec(
        resname="CYS",
        residue=dal_spec.residue,
        new_resname="CYX",
        anchor_atom="SG",
    )

    from moleculekit.tools import residue_titration as rt

    calls = []

    def _fetch(c, **k):
        calls.append(c)
        return "C[C@H](C(=O)O)N"

    monkeypatch.setattr(rt, "rcsbFetchLigandSmiles", _fetch)
    titration = rt.capNonstandardResiduesForTitration(mol, [cyx_spec, dal_spec])

    assert set(titration) == {"DAL"}  # the renamed-canonical crosslink is skipped
    assert calls == ["DAL"]  # CYS/CYX was never fetched from RCSB


def test_templatesFromTitration_strips_caps_keeps_sidechain(monkeypatch):
    from moleculekit.tools import residue_titration as rt
    from rdkit import Chem

    # Exercises the strip via the (mol, specs, protonated) API. The anchor
    # derivation is covered by test_uncapped_residue_smiles_matches_capped_residue,
    # so stub it to an aspartate-like skeleton and focus on the strip: a pKa
    # tool deprotonated the sidechain carboxyl; the caps (ACE + NME) must be
    # removed and the sidechain carboxylate (-1) preserved.
    mol = _dal_mol()
    spec = _dal_spec(mol)  # ChainResidueSpec, key "DAL"
    monkeypatch.setattr(rt, "_uncapped_residue_smiles", lambda m, s, b: "NC(CC(=O)O)C=O")
    protonated = {"DAL": "CC(=O)NC(CC(=O)[O-])C(=O)NC"}
    out = rt.templatesFromTitration(
        mol, [spec], protonated, smiles={"DAL": "C[C@H](C(=O)O)N"}
    )
    m = Chem.MolFromSmiles(out["DAL"])
    assert m is not None
    assert any(a.GetFormalCharge() == -1 for a in m.GetAtoms())
    assert Chem.MolToSmiles(m) == Chem.MolToSmiles(
        Chem.MolFromSmiles("NC(CC(=O)[O-])C=O")
    )


def test_templatesFromTitration_raises_when_anchor_absent(monkeypatch):
    from moleculekit.tools import residue_titration as rt

    # The anchor is a sulfur-bearing skeleton that simply does not occur in the
    # (benzene) protonated SMILES, so the substructure match must fail with a
    # clear error naming the key.
    mol = _dal_mol()
    spec = _dal_spec(mol)
    monkeypatch.setattr(rt, "_uncapped_residue_smiles", lambda m, s, b: "NC(CS)C=O")
    with pytest.raises(RuntimeError, match="DAL"):
        rt.templatesFromTitration(
            mol, [spec], {"DAL": "c1ccccc1"}, smiles={"DAL": "C[C@H](C(=O)O)N"}
        )


def test_templatesFromTitration_raises_on_unparseable_protonated_smiles(monkeypatch):
    from moleculekit.tools import residue_titration as rt

    # A protonated value RDKit cannot parse must raise a clear RuntimeError
    # naming the key, not a cryptic AttributeError on a None molecule.
    mol = _dal_mol()
    spec = _dal_spec(mol)
    monkeypatch.setattr(rt, "_uncapped_residue_smiles", lambda m, s, b: "NC(CS)C=O")
    with pytest.raises(RuntimeError, match="DAL"):
        rt.templatesFromTitration(
            mol, [spec], {"DAL": "not_a_smiles"}, smiles={"DAL": "C[C@H](C(=O)O)N"}
        )


def test_templatesFromTitration_raises_on_missing_key(monkeypatch):
    from moleculekit.tools import residue_titration as rt

    # A key present in the specs but absent from the protonated dict is a clear
    # caller error, not a KeyError.
    mol = _dal_mol()
    spec = _dal_spec(mol)
    monkeypatch.setattr(rt, "_uncapped_residue_smiles", lambda m, s, b: "NC(C)C=O")
    with pytest.raises(RuntimeError, match="DAL"):
        rt.templatesFromTitration(mol, [spec], {}, smiles={"DAL": "C[C@H](C(=O)O)N"})


def test_capForTitration_handles_covalent_ligand(monkeypatch):
    # LIG's only inter-residue bond is its C3-CYS.SG thioether crosslink, so
    # detectNonStandardResidues reports it as a CovalentLigandSpec (the CYS
    # side is canonical and filtered out by requiresTemplate). A
    # CovalentLigandSpec is not a LigandSpec, so it must take the SAME cap
    # path as a ChainResidueSpec: its crosslink is inert-capped, not passed
    # through untouched.
    from moleculekit.tools import residue_titration as rt

    mol = _covalent_ligand_and_cys()
    specs = detectNonStandardResidues(mol)
    lig_spec = next(s for s in specs if s.resname == "LIG")
    assert isinstance(lig_spec, CovalentLigandSpec)

    # A straight-chain propylamine-like LIG: three heavy atoms line up with
    # the residue's own C1-C2-C3 path (an aromatic ring SMILES would let the
    # MCS map the crosslink carbon onto an already-trisubstituted ring atom,
    # which cannot take a fourth substituent and would kekulization-fail).
    base_smiles = "CCCN"
    monkeypatch.setattr(rt, "rcsbFetchLigandSmiles", lambda c, **k: base_smiles)
    titration = rt.capNonstandardResiduesForTitration(mol, specs)

    assert "LIG" in titration
    m = Chem.MolFromSmiles(titration["LIG"])
    assert m is not None
    # LIG carries no sulfur itself (the SG is on the CYS side of the
    # crosslink), so "no free thiol" would be trivially true here and prove
    # nothing. The meaningful check: the severed crosslink carbon (C3) is not
    # a nitrogen or an amide carbonyl, so _classify_junction calls it "other"
    # and _attach_methyl caps it with a plain [CH3] - one extra carbon versus
    # the uncapped base, giving the exact expected capped structure.
    assert titration["LIG"] != base_smiles  # actually capped, not passed through
    assert Chem.MolToSmiles(m) == Chem.CanonSmiles("CCC(C)N")
    base_atoms = Chem.MolFromSmiles(base_smiles).GetNumAtoms()
    assert m.GetNumAtoms() == base_atoms + 1


def test_covalent_ligand_roundtrip(monkeypatch):
    # cap -> (echo pKa) -> strip yields a per-resname template for the
    # one-bond CovalentLigandSpec case (LIG's single C3-CYS.SG crosslink).
    # This is NOT a ScaffoldSpec case: detectNonStandardResidues classifies
    # a residue with exactly one non-peptide crosslink as CovalentLigandSpec,
    # never ScaffoldSpec (that requires two or more); see
    # test_scaffold_two_crosslinks_caps_both_junctions below for the genuine
    # 2+-crosslink scaffold path.
    from moleculekit.tools import residue_titration as rt

    mol = _covalent_ligand_and_cys()
    specs = detectNonStandardResidues(mol)
    lig_spec = next(s for s in specs if s.resname == "LIG")
    assert isinstance(lig_spec, CovalentLigandSpec)
    monkeypatch.setattr(rt, "rcsbFetchLigandSmiles", lambda c, **k: "CCCN")
    titration = rt.capNonstandardResiduesForTitration(mol, specs)
    protonated = dict(titration)  # non-ionizable here, echo is faithful
    templates = rt.templatesFromTitration(mol, specs, protonated)
    assert "LIG" in templates
    assert Chem.MolFromSmiles(templates["LIG"]) is not None


def _scaffold_two_crosslinks():
    """A genuine :class:`~moleculekit.tools.nonstandard_residues.ScaffoldSpec`
    fixture: a non-canonical ``SCF`` hub, a straight 5-carbon aliphatic chain
    (``C1``-``C2``-``C3``-``C4``-``C5``), with BOTH terminal carbons carrying
    an explicit, non-peptide inter-residue bond (``mol.bonds``/``mol.bondtype``
    entries, no distance guessing) out to two separate one-atom partner
    residues (``LNK`` and ``LN2``). Two non-peptide crosslinks on a residue
    that is not chain-resident is exactly the ScaffoldSpec case (a
    CovalentLigandSpec has exactly one).

    The hub's own coordinates are a straight line spaced at a realistic C-C
    bond length so ``guessBonds()`` (run internally on the isolated ``SCF``
    residue by ``_isolated_residue_rdkit``) recovers the intended C1-C2-C3-
    C4-C5 chain; the crosslink partners are filtered out before that step, so
    their coordinates are irrelevant and are simply placed further out.

    Both crosslink carbons are terminal (aliphatic, unsubstituted) methyls in
    the hub's own chain, deliberately avoiding an aromatic or already-
    trisubstituted junction atom (those hit the known uncappable fallback).
    """
    names = ["C1", "C2", "C3", "C4", "C5"] + ["PA"] + ["PB"]
    elements = ["C"] * 5 + ["C"] + ["C"]
    resnames = ["SCF"] * 5 + ["LNK"] + ["LN2"]
    resids = [1] * 5 + [2] + [3]
    coords = [
        (0.0, 0.0, 0.0),  # C1 (crosslink 1)
        (1.54, 0.0, 0.0),  # C2
        (3.08, 0.0, 0.0),  # C3
        (4.62, 0.0, 0.0),  # C4
        (6.16, 0.0, 0.0),  # C5 (crosslink 2)
    ] + [
        (-1.54, 0.0, 0.0),  # LNK PA, crosslink partner for C1
        (7.70, 0.0, 0.0),  # LN2 PB, crosslink partner for C5
    ]
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elements
    mol.resname[:] = resnames
    mol.resid[:] = resids
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.record[:] = "HETATM"
    mol.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    bonds = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # SCF hub chain: C1-C2-C3-C4-C5
        (0, 5),  # crosslink 1: SCF C1 - LNK PA
        (4, 6),  # crosslink 2: SCF C5 - LN2 PB
    ]
    mol.bonds = np.array(bonds, dtype=np.uint32)
    mol.bondtype = np.array(["1"] * len(bonds), dtype=object)
    return mol


def test_scaffold_two_crosslinks_caps_both_junctions():
    # A real ScaffoldSpec (2+ non-peptide crosslinks), built directly (not
    # via detectNonStandardResidues, which would also run anchor validation
    # this fixture does not need to exercise). Both hub-terminal carbons are
    # "other"-kind junctions (aliphatic C, not an amide N or carbonyl C), so
    # _classify_junction routes both to a plain methyl cap: the capped SMILES
    # must therefore carry exactly two more carbons than the base, one per
    # crosslink, not just one (which would mean only one junction was capped).
    from moleculekit.tools import residue_titration as rt

    mol = _scaffold_two_crosslinks()
    rid = UniqueResidueID.fromMolecule(mol, idx=0)  # SCF's C1 atom
    spec = ScaffoldSpec(resname="SCF", residue=rid)

    xs = _inter_residue_crosslinks(mol, spec)
    assert len(xs) == 2  # both hub-terminal bonds found, not just one

    base_smiles = "CCCCC"
    titration = rt.capNonstandardResiduesForTitration(
        mol, [spec], smiles={"SCF": base_smiles}
    )
    assert "SCF" in titration
    assert titration["SCF"] != base_smiles  # actually capped, not passed through

    m = Chem.MolFromSmiles(titration["SCF"])
    assert m is not None
    base_atoms = Chem.MolFromSmiles(base_smiles).GetNumAtoms()
    assert m.GetNumAtoms() == base_atoms + 2  # two methyl caps, one per crosslink
    assert Chem.MolToSmiles(m) == Chem.CanonSmiles("CCCCCCC")


def _cys_ncaa_thioether_crosslink():
    """A mid-chain non-canonical Cys-like residue ("CY2") whose ``SG`` carries
    an explicit thioether bond out to a separate residue's atom.

    Mirrors ``_dal_mol()``'s single-hand-built-residue pattern (backbone
    ``N``, ``CA``, ``C``, ``O`` at the same amino-acid-like geometry so
    ``guessBonds()`` recovers the same backbone), plus a sidechain ``CB``
    bonded to ``SG`` (a genuine, non-terminal C-S bond so ``guessBonds()``
    also recovers it), and a second, one-atom "LNK" residue supplying the
    crosslink partner via an explicit ``mol.bonds`` entry (the same
    explicit-bonds pattern already used by ``_covalent_ligand_and_cys()``).
    """
    names = ["CX"] + ["N", "CA", "CB", "SG", "C", "O"]
    elements = ["C"] + ["N", "C", "C", "S", "C", "O"]
    resnames = ["LNK"] + ["CY2"] * 6
    resids = [1] + [2] * 6
    coords = [
        (5.0, -0.5, 1.4),  # LNK CX (thioether crosslink partner)
    ] + [
        (-1.0, 1.0, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (0.3, -0.5, 1.4),  # CB
        (2.1, -0.5, 1.4),  # SG (~1.8 A from CB, a real C-S bond)
        (1.2, 0.6, -0.6),  # C
        (1.1, 1.8, -0.6),  # O
    ]
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elements
    mol.resname[:] = resnames
    mol.resid[:] = resids
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.record[:] = ["HETATM"] + ["ATOM"] * 6
    mol.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    bonds = [
        (1, 2), (2, 3), (3, 4), (2, 5), (5, 6),  # CY2: N-CA, CA-CB, CB-SG, CA-C, C=O
        (4, 0),  # the crosslink: CY2 SG - LNK CX
    ]
    mol.bonds = np.array(bonds, dtype=np.uint32)
    mol.bondtype = np.array(["1"] * len(bonds), dtype=object)
    return mol


def _cys_ncaa_spec(mol, is_n_term=False, is_c_term=False):
    """Hand-built ``ChainResidueSpec`` for the ``CY2`` residue in
    :func:`_cys_ncaa_thioether_crosslink`.

    Built directly (the same way ``_dal_spec`` hand-builds a spec for
    ``_dal_mol``) rather than through ``detectNonStandardResidues``, which
    would require real chain peptide bonds on a non-canonical resname to
    classify it as chain-resident; the fixture only needs to exercise
    ``_capped_residue_rdkit``'s crosslink-capping path for a
    ``ChainResidueSpec``, not the detector itself.
    """
    rid = UniqueResidueID.fromMolecule(mol, idx=1)  # CY2's N atom
    return ChainResidueSpec(
        resname="CY2",
        residue=rid,
        anchor_atom="SG",
        is_n_term=is_n_term,
        is_c_term=is_c_term,
    )


def test_capped_residue_inert_caps_sidechain_crosslink():
    # A crosslinked chain NCAA (a Cys-like residue whose SG is thioether-
    # bonded to a partner) must NOT be titrated as a free thiol: the SG
    # crosslink is inert-capped to a thioether (S-CH3), so no free [SX2H]
    # remains.
    mol = _cys_ncaa_thioether_crosslink()
    spec = _cys_ncaa_spec(mol)
    capped = _cap_residue_smiles(mol, spec, "C(CS)(C(=O)O)N")  # free-cysteine-like
    m = Chem.MolFromSmiles(capped)
    assert m is not None
    assert not m.HasSubstructMatch(Chem.MolFromSmarts("[SX2H]"))  # no free thiol
    assert m.HasSubstructMatch(Chem.MolFromSmarts("[SX2]([#6])[#6]"))  # thioether present


def test_capped_residue_backbone_unchanged_for_plain_ncaa():
    # Regression: a plain mid-chain NCAA with no crosslink still caps to two
    # amides, unchanged by the crosslink-capping addition.
    mol = _dal_mol()
    spec = _dal_spec(mol, is_n_term=False, is_c_term=False)
    m = Chem.MolFromSmiles(_cap_residue_smiles(mol, spec, "C[C@H](C(=O)O)N"))
    assert m is not None
    assert len(m.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]"))) == 2


def test_classify_junction_kinds():
    import numpy as np
    from rdkit import Chem
    from moleculekit.molecule import Molecule
    from moleculekit.tools.residue_titration import _classify_junction

    # A tiny structure mol carrying just the elements the classifier reads.
    mol = Molecule().empty(3)
    mol.name[:] = ["N", "C", "S"]
    mol.element[:] = ["N", "C", "S"]
    mol.coords = np.zeros((3, 3, 1), dtype=np.float32)

    # rw = an acetyl-alanine-like fragment so the mapped C carries a =O.
    rw = Chem.RWMol(Chem.MolFromSmiles("CC(=O)NC(C)C(=O)O"))
    carbonyl_c = [a.GetIdx() for a in rw.GetAtoms()
                  if a.GetSymbol() == "C"
                  and any(b.GetBondType() == Chem.BondType.DOUBLE
                          and b.GetOtherAtom(a).GetSymbol() == "O"
                          for b in a.GetBonds())][0]
    amine_n = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "N"][0]

    # local N (mol idx 0), any partner -> amide_n
    assert _classify_junction(mol, 0, 1, rw, amine_n) == "amide_n"
    # local carbonyl C (mol idx 1) whose partner is N (mol idx 0) -> amide_c
    assert _classify_junction(mol, 1, 0, rw, carbonyl_c) == "amide_c"
    # local S (mol idx 2), partner C -> other
    assert _classify_junction(mol, 2, 1, rw, carbonyl_c) == "other"


def _amine_crosslink_mol():
    """A small non-canonical residue ("LY2") whose terminal amine nitrogen
    (``NZ``, an isopeptide-acceptor analog of a Lys sidechain) carries an
    explicit, non-peptide inter-residue bond (``mol.bonds``/``mol.bondtype``,
    no distance guessing) out to a separate one-atom partner residue
    ("LNK"). The LOCAL crosslink atom is itself a nitrogen, so
    ``_classify_junction`` must route it to ``"amide_n"`` (acetyl cap), not
    the catch-all methyl cap that every other crosslink kind gets: a methyl
    on a nitrogen would leave a free tertiary amine, not the inert,
    non-titratable amide a severed isopeptide/amide bond must become.

    ``C1``-``C2``-``C3``-``NZ`` mirrors a short alkylamine chain
    (propylamine, ``base_smiles="CCCN"``) at realistic bond-length spacing so
    ``guessBonds()`` recovers the intended chain when
    ``_isolated_residue_rdkit`` isolates just the ``LY2`` residue (the
    ``LNK`` partner is filtered out first, so its own coordinates are
    irrelevant).
    """
    names = ["C1", "C2", "C3", "NZ"] + ["CX"]
    elements = ["C", "C", "C", "N"] + ["C"]
    resnames = ["LY2"] * 4 + ["LNK"]
    resids = [1] * 4 + [2]
    coords = [
        (0.0, 0.0, 0.0),  # C1
        (1.54, 0.0, 0.0),  # C2
        (3.08, 0.0, 0.0),  # C3
        (4.55, 0.0, 0.0),  # NZ (crosslink)
        (6.0, 0.0, 0.0),  # LNK CX, crosslink partner
    ]
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elements
    mol.resname[:] = resnames
    mol.resid[:] = resids
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.record[:] = "HETATM"
    mol.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    bonds = [
        (0, 1), (1, 2), (2, 3),  # LY2: C1-C2, C2-C3, C3-NZ
        (3, 4),  # the crosslink: LY2 NZ - LNK CX
    ]
    mol.bonds = np.array(bonds, dtype=np.uint32)
    mol.bondtype = np.array(["1"] * len(bonds), dtype=object)
    return mol


def test_amide_crosslink_stays_nontitratable_amide():
    # A severed amide/isopeptide crosslink whose LOCAL atom is a nitrogen
    # must come out as an amide (acetylated), never a free primary or
    # secondary amine: that free amine would otherwise look like a
    # titratable group to a downstream pKa tool.
    from moleculekit.tools.nonstandard_residues import CovalentLigandSpec

    mol = _amine_crosslink_mol()
    rid = UniqueResidueID.fromMolecule(mol, idx=0)  # LY2's C1 atom
    spec = CovalentLigandSpec(resname="LY2", residue=rid)

    xs = _inter_residue_crosslinks(mol, spec)
    assert len(xs) == 1
    local, partner = xs[0]
    assert str(mol.name[local]) == "NZ"

    capped = _cap_residue_smiles(mol, spec, "CCCN")
    m = Chem.MolFromSmiles(capped)
    assert m is not None

    free_amine = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
    assert not m.HasSubstructMatch(free_amine)  # no free primary/secondary amine

    amide = Chem.MolFromSmarts("[NX3][CX3]=O")
    assert m.HasSubstructMatch(amide)  # the severed NZ is now an amide nitrogen


# Cyclosporin's six non-canonical residues as their neutral RCSB free-acid /
# free-amine ligand descriptors (what an RCSB fetch returns): the alpha-amine
# is a bare -N / -NC and the alpha-carboxyl a free -C(=O)O, i.e. NOT the
# pH-7.4 form. Fed uncapped to a pKa tool these titrate to the backbone
# zwitterion ([NH3+]...C(=O)[O-]); capping is what prevents that.
_CYCLOSPORIN_RCSB_SMILES = {
    "DAL": "C[C@H](C(=O)O)N",
    "ABA": "CC[C@@H](C(=O)O)N",
    "SAR": "CNCC(=O)O",
    "MLE": "CC(C)C[C@@H](C(=O)O)NC",
    "MVA": "CC(C)[C@@H](C(=O)O)NC",
    "BMT": "C/C=C/C[C@@H](C)[C@H]([C@@H](C(=O)O)NC)O",
}

# Expected per-residue templates after cap -> (echo) -> strip. Five match the
# hand-written mid-chain SMILES in htmd's test_full_pipeline_1m63 verbatim; BMT
# additionally carries the two sidechain stereocentres (the 4-methyl and
# 3-hydroxyl carbons) that htmd's reference leaves unspecified but the RCSB
# descriptor defines, so the pipeline's form is the more complete one.
_CYCLOSPORIN_MIDCHAIN_SMILES = {
    "DAL": "C[C@H](C=O)N",
    "ABA": "CC[C@@H](C=O)N",
    "SAR": "O=CCNC",
    "MLE": "CC(C)C[C@@H](C=O)NC",
    "MVA": "CC(C)[C@@H](C=O)NC",
    "BMT": "C/C=C/C[C@@H](C)[C@@H](O)[C@@H](C=O)NC",
}


@pytest.fixture(scope="module")
def cyclosporin_1m63():
    """The 1M63 structure (calcineurin / cyclophilin + cyclosporin), waters
    removed. Fetched once from RCSB; the whole module skips if it is
    unreachable, so the round-trip tests never fail on a network hiccup."""
    try:
        mol = Molecule("1M63")
    except Exception as e:
        pytest.skip(f"could not fetch 1M63 from RCSB: {e}")
    mol.remove("water", _logger=False)
    return mol


def test_cyclosporin_1m63_capping_neutralizes_backbone(cyclosporin_1m63):
    """Fed their neutral RCSB SMILES, 1M63's six cyclosporin NCAAs cap to fully
    amide-terminated molecules: no free alpha-carboxyl and no free alpha-amine
    survive, so a pKa tool has nothing to titrate into the backbone zwitterion
    that broke the original uncapped flow."""
    from rdkit import Chem
    from moleculekit.tools import residue_titration as rt

    specs = detectNonStandardResidues(cyclosporin_1m63)
    capped = rt.capNonstandardResiduesForTitration(
        cyclosporin_1m63, specs, smiles=_CYCLOSPORIN_RCSB_SMILES
    )
    assert set(capped) == set(_CYCLOSPORIN_RCSB_SMILES)

    free_acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    free_amine = Chem.MolFromSmarts("[NX3;H2]")
    for resname, smi in capped.items():
        m = Chem.MolFromSmiles(smi)
        assert m is not None, f"{resname} capped SMILES unparseable: {smi}"
        assert not m.HasSubstructMatch(free_acid), (
            f"{resname} still has a free carboxylic acid after capping: {smi}"
        )
        assert not m.HasSubstructMatch(free_amine), (
            f"{resname} still has a free primary amine after capping: {smi}"
        )


def test_cyclosporin_1m63_roundtrip_yields_midchain_templates(cyclosporin_1m63):
    """cap -> (echo) -> strip on 1M63's six cyclosporin NCAAs, fed their neutral
    RCSB SMILES, reproduces the mid-chain build templates. Cyclosporin's
    sidechains carry no group ionizable at pH 7.4 (BMT's beta-hydroxyl is an
    alcohol), so a real AcePka pass would echo the capped SMILES back unchanged;
    echoing here exercises the strip half without running a pKa job."""
    from rdkit import Chem
    from moleculekit.tools import residue_titration as rt

    specs = detectNonStandardResidues(cyclosporin_1m63)
    capped = rt.capNonstandardResiduesForTitration(
        cyclosporin_1m63, specs, smiles=_CYCLOSPORIN_RCSB_SMILES
    )
    templates = rt.templatesFromTitration(
        cyclosporin_1m63, specs, dict(capped), smiles=_CYCLOSPORIN_RCSB_SMILES
    )
    assert set(templates) == set(_CYCLOSPORIN_MIDCHAIN_SMILES)
    for resname, expected in _CYCLOSPORIN_MIDCHAIN_SMILES.items():
        assert Chem.CanonSmiles(templates[resname]) == Chem.CanonSmiles(expected), (
            f"{resname}: got {templates[resname]!r}, expected {expected!r}"
        )


@pytest.mark.parametrize(
    "anchor,capped",
    [
        # 4EFP's 0AF (7-hydroxytryptophan): an indole [nH]
        (
            "N[C@H](C=O)Cc1c[nH]c2c(O)cccc12",
            "CNC(=O)[C@H](Cc1c[nH]c2c(O)cccc12)NC(C)=O",
        ),
        # a methylhistidine: an imidazole [nH]
        (
            "N[C@H](C=O)Cc1c[nH]c(C)n1",
            "CNC(=O)[C@H](Cc1c[nH]c(C)n1)NC(C)=O",
        ),
    ],
)
def test_relaxed_query_locates_aromatic_nh_sidechain(anchor, capped):
    """An aromatic N-H's hydrogen is what donates the lone pair its ring needs
    to kekulize, so the relaxed query has to keep it: clearing every atom's
    hydrogen count left the five-membered ring unkekulizable and the query
    failed to build at all. Only modified residues reach this path, which is
    why canonical Trp and His never showed it."""
    query = _relaxed_query(anchor)
    match = Chem.MolFromSmiles(capped).GetSubstructMatch(query)
    assert len(match) == query.GetNumAtoms()


@pytest.fixture(scope="module")
def hydroxytryptophan_4efp():
    """The 4EFP structure (peptidylglycine alpha-hydroxylating monooxygenase),
    waters removed, whose two 0AF residues are 7-hydroxytryptophan: a chain NCAA
    carrying an indole [nH]. Fetched once from RCSB; the test skips if it is
    unreachable, so it never fails on a network hiccup."""
    try:
        mol = Molecule("4EFP")
    except Exception as e:
        pytest.skip(f"could not fetch 4EFP from RCSB: {e}")
    mol.remove("water", _logger=False)
    return mol


def test_hydroxytryptophan_4efp_roundtrip_yields_midchain_template(
    hydroxytryptophan_4efp,
):
    """cap -> (echo) -> strip on 4EFP's 0AF residues, fed their neutral RCSB
    SMILES, reproduces the mid-chain build template. The sidechain's phenol is
    not ionizable at pH 7.4, so a real AcePka pass would echo the capped SMILES
    back unchanged; echoing here exercises the strip half without running a pKa
    job."""
    from moleculekit.tools import residue_titration as rt

    specs = [
        s
        for s in detectNonStandardResidues(hydroxytryptophan_4efp)
        if s.resname == "0AF"
    ]
    base = {"0AF": "c1cc2c(c[nH]c2c(c1)O)C[C@@H](C(=O)O)N"}
    capped = rt.capNonstandardResiduesForTitration(
        hydroxytryptophan_4efp, specs, smiles=base
    )
    templates = rt.templatesFromTitration(
        hydroxytryptophan_4efp, specs, dict(capped), smiles=base
    )
    assert Chem.CanonSmiles(templates["0AF"]) == Chem.CanonSmiles(
        "N[C@H](C=O)Cc1c[nH]c2c(O)cccc12"
    )


def _leaving_group_amine_crosslink_mol():
    """A residue ``ACL`` whose crosslink carbon is fully substituted in its base
    SMILES (a tert-butanol-like carbon bearing a hydroxyl) and is covalently
    bonded to a partner nitrogen. Models a condensation crosslink (e.g. an
    N-glycosidic bond) where the residue's -OH is the leaving group displaced by
    the C-N bond: the free-form SMILES still carries the -OH, the deposited
    structure does not. Capping such an atom used to over-valence it (the -OH
    plus the added cap) and fall back to titrating the residue whole.
    """
    names = ["CX", "CA", "CB", "CC", "NP"]
    elements = ["C", "C", "C", "C", "N"]
    resnames = ["ACL", "ACL", "ACL", "ACL", "PNR"]
    resids = [1, 1, 1, 1, 2]
    coords = [
        (0.0, 0.0, 0.0),  # CX (crosslink carbon)
        (0.87, 0.87, 0.87),  # CA
        (0.87, -0.87, -0.87),  # CB
        (-0.87, 0.87, -0.87),  # CC
        (-0.87, -0.87, 0.87),  # NP (partner nitrogen)
    ]
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elements
    mol.resname[:] = resnames
    mol.resid[:] = resids
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.record[:] = "HETATM"
    mol.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    mol.bonds = np.array([(0, 1), (0, 2), (0, 3), (0, 4)], dtype=np.uint32)
    mol.bondtype = np.array(["1"] * 4, dtype=object)
    return mol


def test_condensation_crosslink_to_nitrogen_caps_as_amide():
    # A crosslink carbon carrying a hydroxyl leaving group in its base SMILES,
    # bonded to a partner N (an N-glycosidic-style condensation bond), must cap
    # to an amide: the -OH is stripped and the severed C-N junction is
    # acetylated, so no free amine and no free hydroxyl remain and the residue
    # is not silently titrated whole.
    mol = _leaving_group_amine_crosslink_mol()
    rid = UniqueResidueID.fromMolecule(mol, idx=0)
    spec = CovalentLigandSpec(resname="ACL", residue=rid)
    capped = _cap_residue_smiles(mol, spec, "OC(C)(C)C")
    m = Chem.MolFromSmiles(capped)
    assert m is not None
    assert m.HasSubstructMatch(Chem.MolFromSmarts("[NX3][CX3]=O"))  # amide junction
    assert not m.HasSubstructMatch(
        Chem.MolFromSmarts("[NX3;H1,H2;!$(NC=O)]")
    )  # no free primary/secondary amine
    assert not m.HasSubstructMatch(Chem.MolFromSmarts("[OX2H]"))  # -OH leaving group gone


def _phosphoester_crosslink_mol():
    """A residue ``POX`` whose sidechain oxygen is covalently bonded to a
    partner phosphorus, as in a phosphodiester backbone. The oxygen has an open
    valence (it loses only a hydrogen, no heavy leaving group), so no stripping
    is needed; the cap must reflect the real partner (a phosphate), not a plain
    methyl ether.
    """
    names = ["C1", "C2", "O3", "P"]
    elements = ["C", "C", "O", "P"]
    resnames = ["POX", "POX", "POX", "PHO"]
    resids = [1, 1, 1, 2]
    coords = [
        (0.0, 0.0, 0.0),  # C1
        (1.5, 0.0, 0.0),  # C2
        (2.4, 1.2, 0.0),  # O3 (crosslink oxygen)
        (3.9, 1.5, 0.0),  # P (partner phosphorus)
    ]
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elements
    mol.resname[:] = resnames
    mol.resid[:] = resids
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.record[:] = "HETATM"
    mol.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    mol.bonds = np.array([(0, 1), (1, 2), (2, 3)], dtype=np.uint32)
    mol.bondtype = np.array(["1"] * 3, dtype=object)
    return mol


def test_phosphoester_crosslink_caps_as_phosphate():
    # A sidechain oxygen bonded to a partner phosphorus (a phosphodiester
    # crosslink) must cap as a phosphate reflecting the real partner element,
    # not the plain methyl ether the element-agnostic heuristic would produce.
    mol = _phosphoester_crosslink_mol()
    rid = UniqueResidueID.fromMolecule(mol, idx=0)
    spec = CovalentLigandSpec(resname="POX", residue=rid)
    capped = _cap_residue_smiles(mol, spec, "CCO")
    m = Chem.MolFromSmiles(capped)
    assert m is not None
    assert any(a.GetSymbol() == "P" for a in m.GetAtoms())  # partner P reflected
    assert m.HasSubstructMatch(Chem.MolFromSmarts("[#6][OX2]P(=O)"))  # O-P phosphate


def _ether_crosslink_mol():
    """A residue ``ETX`` whose sp3 carbon is bonded to a partner oxygen (an
    ether crosslink). The carbon has an open valence, so nothing is stripped;
    the partner oxygen must give a hydroxyl, not the element-agnostic methyl."""
    names = ["C1", "C2", "OP"]
    elements = ["C", "C", "O"]
    resnames = ["ETX", "ETX", "OXR"]
    resids = [1, 1, 2]
    coords = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.4, 1.2, 0.0)]
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elements
    mol.resname[:] = resnames
    mol.resid[:] = resids
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.record[:] = "HETATM"
    mol.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    mol.bonds = np.array([(0, 1), (1, 2)], dtype=np.uint32)
    mol.bondtype = np.array(["1", "1"], dtype=object)
    return mol


def test_oxygen_crosslink_caps_as_hydroxyl():
    # A carbon bonded to a partner oxygen (an ether crosslink) caps to a
    # hydroxyl reflecting the real partner element, not the plain methyl the
    # element-agnostic heuristic would add.
    mol = _ether_crosslink_mol()
    rid = UniqueResidueID.fromMolecule(mol, idx=0)
    spec = CovalentLigandSpec(resname="ETX", residue=rid)
    capped = _cap_residue_smiles(mol, spec, "CC")
    m = Chem.MolFromSmiles(capped)
    assert m is not None
    assert m.HasSubstructMatch(Chem.MolFromSmarts("[CX4][OX2H]"))  # hydroxyl, not methyl


def _ester_crosslink_mol():
    """A residue ``EST`` whose sidechain carbonyl carbon is bonded to a partner
    oxygen (an ester crosslink). Its free-acid base SMILES carries the hydroxyl
    that ester formation displaced; capping must not leave a titratable free
    carboxylic acid on the severed carbonyl."""
    names = ["CM", "CO", "OD", "OX"]
    elements = ["C", "C", "O", "O"]
    resnames = ["EST", "EST", "EST", "ALX"]
    resids = [1, 1, 1, 2]
    coords = [
        (0.0, 0.0, 0.0),  # CM (methyl)
        (1.5, 0.0, 0.0),  # CO (carbonyl carbon)
        (2.1, 1.0, 0.0),  # OD (carbonyl oxygen)
        (2.1, -1.2, 0.0),  # OX (partner ester oxygen)
    ]
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elements
    mol.resname[:] = resnames
    mol.resid[:] = resids
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.record[:] = "HETATM"
    mol.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    mol.bonds = np.array([(0, 1), (1, 2), (1, 3)], dtype=np.uint32)
    mol.bondtype = np.array(["1", "2", "1"], dtype=object)
    return mol


def test_ester_crosslink_does_not_leave_free_acid():
    # A carbonyl carbon bonded to a partner oxygen (an ester crosslink) must not
    # cap to a free carboxylic acid: a hydroxyl on the carbonyl would be a
    # titratable acid, so this junction stays non-titratable (a methyl ketone).
    mol = _ester_crosslink_mol()
    rid = UniqueResidueID.fromMolecule(mol, idx=0)
    spec = CovalentLigandSpec(resname="EST", residue=rid)
    capped = _cap_residue_smiles(mol, spec, "CC(=O)O")
    m = Chem.MolFromSmiles(capped)
    assert m is not None
    assert not m.HasSubstructMatch(
        Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
    )  # no free carboxylic acid / carboxylate


def _stereo_leaving_group_crosslink_mol():
    """A residue ``STG`` whose crosslink carbon is a stereocentre carrying a
    hydroxyl leaving group in its base SMILES (a butan-2-ol-like carbon) and is
    bonded to a partner nitrogen. The stereocentre's explicit hydrogen (from the
    ``[C@@H]`` notation) is frozen, so after the leaving group and the cap are
    stripped the residue-skeleton anchor must still relocate the atom inside the
    capped molecule despite its now-inconsistent hydrogen count.
    """
    names = ["C1", "C2", "C3", "C4", "NP"]
    elements = ["C", "C", "C", "C", "N"]
    resnames = ["STG", "STG", "STG", "STG", "PNR"]
    resids = [1, 1, 1, 1, 2]
    coords = [
        (0.0, 0.0, 0.0),  # C1 (methyl)
        (1.5, 0.0, 0.0),  # C2 (stereocentre crosslink carbon)
        (2.2, 1.3, 0.0),  # C3
        (3.7, 1.3, 0.0),  # C4
        (1.5, -1.4, 0.0),  # NP (partner nitrogen)
    ]
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elements
    mol.resname[:] = resnames
    mol.resid[:] = resids
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.record[:] = "HETATM"
    mol.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    mol.bonds = np.array([(0, 1), (1, 2), (2, 3), (1, 4)], dtype=np.uint32)
    mol.bondtype = np.array(["1"] * 4, dtype=object)
    return mol


def test_stereocenter_crosslink_roundtrips_to_template():
    # cap -> (echo) -> strip must succeed even when the crosslink atom is a
    # stereocentre whose leaving group was stripped: the residue-skeleton anchor
    # gains a hydrogen the capped molecule does not have, so the strip step must
    # match on connectivity, not exact hydrogen count.
    from moleculekit.tools import residue_titration as rt

    mol = _stereo_leaving_group_crosslink_mol()
    rid = UniqueResidueID.fromMolecule(mol, idx=0)
    spec = CovalentLigandSpec(resname="STG", residue=rid)
    base = "O[C@@H](C)CC"
    capped = rt.capNonstandardResiduesForTitration(
        mol, [spec], smiles={"STG": base}, _logger=False
    )
    templates = rt.templatesFromTitration(mol, [spec], dict(capped), smiles={"STG": base})
    assert Chem.MolFromSmiles(templates["STG"]) is not None


def _own_phosphate_crosslink_mol():
    """A residue ``PHP`` whose own phosphate phosphorus is the crosslink atom
    (its phosphate bonds out to a partner oxygen, as an internal nucleotide's
    5'-phosphate bonds to the previous residue's O3'). The phosphorus is already
    a complete phosphate in the base SMILES, so its outgoing bond is represented
    by an -OH that must survive the cap -> strip round-trip intact, not collapse
    to a spurious P-H."""
    names = ["C", "OE", "P", "OD", "ON", "OX"]
    elements = ["C", "O", "P", "O", "O", "O"]
    resnames = ["PHP", "PHP", "PHP", "PHP", "PHP", "OXP"]
    resids = [1, 1, 1, 1, 1, 2]
    coords = [
        (0.0, 0.0, 0.0),  # C (methyl)
        (1.4, 0.0, 0.0),  # OE (ester oxygen)
        (2.9, 0.5, 0.0),  # P
        (3.5, 1.9, 0.0),  # OD (=O)
        (4.3, -0.3, 0.0),  # ON (non-bridging O)
        (2.3, -1.0, 0.0),  # OX (partner oxygen)
    ]
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elements
    mol.resname[:] = resnames
    mol.resid[:] = resids
    mol.chain[:] = "A"
    mol.segid[:] = "A"
    mol.record[:] = "HETATM"
    mol.coords = np.array(coords, dtype=np.float32).reshape(-1, 3, 1)
    mol.bonds = np.array([(0, 1), (1, 2), (2, 3), (2, 4), (2, 5)], dtype=np.uint32)
    mol.bondtype = np.array(["1"] * 5, dtype=object)
    return mol


def test_own_phosphate_crosslink_template_stays_phosphate():
    # When the crosslink atom is the residue's own (already-complete) phosphate
    # phosphorus bonded to a partner oxygen, the cap -> strip round-trip must
    # yield a proper phosphate, not an H-phosphonate: RDKit must not be left to
    # fill an under-coordinated phosphorus with a spurious hydrogen.
    from moleculekit.tools import residue_titration as rt

    mol = _own_phosphate_crosslink_mol()
    rid = UniqueResidueID.fromMolecule(mol, idx=0)
    spec = CovalentLigandSpec(resname="PHP", residue=rid)
    base = "COP(=O)(O)O"
    capped = rt.capNonstandardResiduesForTitration(
        mol, [spec], smiles={"PHP": base}, _logger=False
    )
    templates = rt.templatesFromTitration(mol, [spec], dict(capped), smiles={"PHP": base})
    m = Chem.MolFromSmiles(templates["PHP"])
    assert m is not None
    p = next(a for a in m.GetAtoms() if a.GetSymbol() == "P")
    assert p.GetTotalNumHs() == 0  # a phosphate, not an H-phosphonate


# ---------------------------------------------------------------------------
# File round-trip: titration.csv out, protonated.csv in, templates.json out.
# ---------------------------------------------------------------------------


def test_capForTitration_outfile_writes_key_smiles_base(tmp_path):
    import csv

    from moleculekit.tools import residue_titration as rt

    mol = _dal_mol()
    spec = _dal_spec(mol)
    base = "C[C@H](C(=O)O)N"
    out_csv = tmp_path / "titration.csv"
    titration = rt.capNonstandardResiduesForTitration(
        mol, [spec], smiles={"DAL": base}, outfile=str(out_csv)
    )
    with open(out_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [set(r) for r in rows] == [{"key", "SMILES", "base"}]
    assert rows[0]["key"] == "DAL"
    assert rows[0]["SMILES"] == titration["DAL"]
    assert rows[0]["base"] == base


def test_templatesFromTitration_reads_protonated_csv_with_echoed_base(
    tmp_path, monkeypatch
):
    from moleculekit.tools import residue_titration as rt

    mol = _dal_mol()
    spec = _dal_spec(mol)

    # No smiles= override and network disabled: the anchor's base SMILES must
    # come from the CSV's echoed `base` column, not from an RCSB fetch.
    def _no_net(*a, **k):
        raise AssertionError("RCSB fetch attempted despite an echoed base column")

    monkeypatch.setattr(rt, "rcsbFetchLigandSmiles", _no_net)

    seen_bases = []

    def _fake_uncapped(m, s, b):
        seen_bases.append(b)
        return "NC(CC(=O)O)C=O"

    monkeypatch.setattr(rt, "_uncapped_residue_smiles", _fake_uncapped)

    csv_path = tmp_path / "protonated.csv"
    csv_path.write_text(
        "key,SMILES,base\nDAL,CC(=O)NC(CC(=O)[O-])C(=O)NC,C[C@H](C(=O)O)N\n"
    )
    out = rt.templatesFromTitration(mol, [spec], str(csv_path))
    assert seen_bases == ["C[C@H](C(=O)O)N"]
    m = Chem.MolFromSmiles(out["DAL"])
    assert Chem.MolToSmiles(m) == Chem.MolToSmiles(
        Chem.MolFromSmiles("NC(CC(=O)[O-])C=O")
    )


def test_templatesFromTitration_outfile_writes_wrapped_json(tmp_path, monkeypatch):
    import json

    from moleculekit.tools import residue_titration as rt

    mol = _dal_mol()
    spec = _dal_spec(mol)
    monkeypatch.setattr(
        rt, "_uncapped_residue_smiles", lambda m, s, b: "NC(CC(=O)O)C=O"
    )
    protonated = {"DAL": "CC(=O)NC(CC(=O)[O-])C(=O)NC"}
    out_json = tmp_path / "templates.json"
    out = rt.templatesFromTitration(
        mol,
        [spec],
        protonated,
        smiles={"DAL": "C[C@H](C(=O)O)N"},
        outfile=str(out_json),
    )
    with open(out_json) as fh:
        assert json.load(fh) == {"DAL": {"smiles": out["DAL"]}}
