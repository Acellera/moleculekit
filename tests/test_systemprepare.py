from moleculekit.molecule import Molecule, mol_equal
from moleculekit.tools.preparation import systemPrepare, _table_dtypes
import numpy as np
import os
import pytest

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _compare_results(refpdb, refdf_f, pmol: Molecule, df):
    from moleculekit.util import tempname
    import pandas as pd

    # # Use this to update tests
    # pmol.filter("not water", _logger=False)
    # pmol.write(refpdb, writebonds=False)
    # df.to_csv(refdf_f, index=False)

    refdf = pd.read_csv(
        refdf_f, dtype=_table_dtypes, keep_default_na=False, na_values=[""]
    )
    refdf = refdf.fillna("")
    df = df.fillna("")
    try:
        pd.testing.assert_frame_equal(refdf, df)
    except AssertionError:
        if len(df) != len(refdf):
            raise AssertionError(
                "Different number of residues were found in the DataFrames!"
            )
        try:
            for key in [
                "resname",
                "protonation",
                "resid",
                "insertion",
                "chain",
                "segid",
                "pKa",
                "buried",
            ]:
                refv = refdf[key].values
                newv = df[key].values
                diff = refv != newv
                if key in ("pKa", "buried"):
                    nans = refv == ""
                    refv = refv[~nans].astype(np.float32)
                    newv = newv[~nans].astype(np.float32)
                    diff = np.abs(refv - newv) > 1e-4
                if any(diff):
                    print(f"Found difference in field: {key}")
                    print(f"Reference values: {refv[diff]}")
                    print(f"New values:       {newv[diff]}")
        except Exception:
            pass

        df_out = tempname(suffix=".csv")
        df.to_csv(df_out, index=False)
        raise AssertionError(f"Failed comparison of {refdf_f} to {df_out}")

    refmol = Molecule(refpdb)
    refmol.filter("not water", _logger=False)
    pmol.filter("not water", _logger=False)
    coords_prec = 1e-3
    if not mol_equal(
        refmol, pmol, exceptFields=["serial"], fieldPrecision={"coords": coords_prec}
    ):
        pmol_out = tempname(suffix=".pdb")
        pmol.write(pmol_out, writebonds=False)
        raise AssertionError(f"Failed comparison of {refpdb} vs {pmol_out}")


# Explicit-protonation SMILES for cofactors so formal charges are
# well-defined before PDB2PQR. HEM: ferric heme b with [Fe+3], two
# deprotonated pyrroles ([n-]) and two propionates ([O-]); all four
# pyrrole Ns donate via dative bonds. Net charge -1, matching MCPB.py
# (porphyrin -4 + Fe+3) and the ferric resting state of catalase HPII.
_LIGAND_SMILES = {
    "HEM": "C=CC1=C(C)c2cc3c(C)c(CCC(=O)[O-])c4cc5[n]6->[Fe@SP2+3]7(<-[n]2c1cc1c(C)c(C=C)c(cc6C(C)=C5CCC(=O)[O-])[n-]->71)<-[n-]34",
}


@pytest.mark.parametrize("pdb", ["3PTB", "1A25", "1U5U", "1UNC", "6A5J"])
def _test_systemPrepare(pdb):
    test_home = os.path.join(curr_dir, "test_systemprepare", pdb)
    mol = Molecule(os.path.join(test_home, f"{pdb}.pdb"))
    ligand_resnames = set(mol.resname.tolist()) & _LIGAND_SMILES.keys()
    if ligand_resnames:
        mol.remove("element H", _logger=False)
        for resn in ligand_resnames:
            mol.templateResidueFromSmiles(
                mol.resname == resn,
                _LIGAND_SMILES[resn],
                addHs=True,
                _logger=False,
            )
    pmol, _, df = systemPrepare(mol, return_details=True)
    _compare_results(
        os.path.join(test_home, f"{pdb}_prepared.pdb"),
        os.path.join(test_home, f"{pdb}_prepared.csv"),
        pmol,
        df,
    )


def _test_systemprepare_1u5u_heme_tyr_coordination_end_to_end():
    """End-to-end check of the heme-Tyr coordination workflow on 1U5U
    chain A (catalase HPII monomer):

    1. Read mc bonds from bcif (TYR353-OH -> Fe and HOH -> Fe).
    2. Template HEM with the canonical dative-arrow SMILES so the four
       pyrrole N -> Fe bonds also arrive as DATIVE / 'mc' in mol.bonds.
    3. detectNonStandardResidues flags TYR353 (anchor=OH) for re-templating
       as a tyrosinate and HEM as a CovalentLigandSpec.
    4. systemPrepare(detect_specs=...) renames TYR353 -> custom and templates
       it with the Tyr SMILES; the 'mc' cross-bond drives the deprotonation
       logic so the hydroxyl H is dropped AND the OH gets formal charge -1.

    Asserted invariants:
      - TYR353 was renamed (no longer 'TYR')
      - TYR353 has no HH and its OH carries formal charge -1 (tyrosinate)
      - HEM net formal charge -1 (porphyrin -4 + Fe+3, matches MCPB.py)
      - Fe is in the +3 state
      - All 6 Fe coordinations (4 pyrrole N + Tyr-O + axial water O) are 'mc'
    """
    from moleculekit.tools.nonstandard_residues import detectNonStandardResidues

    hem_smiles = "C=CC1=C(C)c2cc3c(C)c(CCC(=O)[O-])c4cc5[n]6->[Fe@SP2+3]7(<-[n]2c1cc1c(C)c(C=C)c(cc6C(C)=C5CCC(=O)[O-])[n-]->71)<-[n-]34"

    mol = Molecule("1u5u")
    mol.filter("chain A")
    mol.remove("element H", _logger=False)
    mol.templateResidueFromSmiles(
        mol.resname == "HEM", hem_smiles, addHs=True, _logger=False
    )
    specs = detectNonStandardResidues(mol)
    assert sum(1 for s in specs if s.resname == "TYR") == 1
    assert sum(1 for s in specs if s.resname == "HEM") == 1

    pmol, _ = systemPrepare(mol, detect_specs=specs)

    # TYR353 - tyrosinate
    m = (pmol.resid == 353) & (pmol.chain == "A")
    assert m.any(), "TYR353/A not found"
    assert "TYR" not in pmol.resname[m].tolist(), (
        f"TYR353 should be renamed, got {set(pmol.resname[m].tolist())}"
    )
    assert not (m & (pmol.name == "HH")).any(), (
        "HH should be stripped on tyrosinate"
    )
    oh = m & (pmol.name == "OH")
    assert oh.sum() == 1
    assert pmol.formalcharge[oh].tolist() == [-1], (
        f"OH should carry -1 charge, got {pmol.formalcharge[oh].tolist()}"
    )

    # HEM - net -1, Fe+3, 4 Fe-N + Fe-Tyr-O + Fe-HOH-O all "mc"
    mh = (pmol.resid == 999) & (pmol.chain == "A") & (pmol.resname == "HEM")
    assert int(pmol.formalcharge[mh].sum()) == -1, (
        f"HEM net charge should be -1, got {pmol.formalcharge[mh].sum()}"
    )
    fe = mh & (pmol.name == "FE")
    assert fe.sum() == 1
    assert pmol.formalcharge[fe].tolist() == [3], (
        f"Fe should be +3, got {pmol.formalcharge[fe].tolist()}"
    )
    fe_idx = int(np.where(fe)[0][0])
    b_mask = (pmol.bonds[:, 0] == fe_idx) | (pmol.bonds[:, 1] == fe_idx)
    partners = []
    for bi in np.where(b_mask)[0]:
        a, c = pmol.bonds[bi]
        partner = int(c if a == fe_idx else a)
        partners.append(partner)
        assert pmol.bondtype[bi] == "mc", (
            f"Fe-{pmol.name[partner]} should be 'mc', got {pmol.bondtype[bi]!r}"
        )
    assert len(partners) == 6, f"expected 6 Fe coordinations, got {len(partners)}"
    partner_names = sorted(pmol.name[partners].tolist())
    # 4 pyrrole N + 1 Tyr-OH + 1 water-O
    assert partner_names.count("NA") == 1
    assert partner_names.count("NB") == 1
    assert partner_names.count("NC") == 1
    assert partner_names.count("ND") == 1
    assert "OH" in partner_names  # tyrosinate
    assert "O" in partner_names   # axial water


def _test_systemprepare_ligand():
    test_home = os.path.join(curr_dir, "test_systemprepare", "test-prepare-with-ligand")
    mol = Molecule(os.path.join(test_home, "5EK0_A.pdb"))
    pmol, _, df = systemPrepare(mol, return_details=True)
    _compare_results(
        os.path.join(test_home, "5EK0_A_prepared.pdb"),
        os.path.join(test_home, "5EK0_A_prepared.csv"),
        pmol,
        df,
    )
    # Now remove the ligands and check again what the pka is
    mol.filter('not resname PX4 "5P2"')
    pmol, _, df = systemPrepare(mol, return_details=True)
    _compare_results(
        os.path.join(test_home, "5EK0_A_prepared_nolig.pdb"),
        os.path.join(test_home, "5EK0_A_prepared_nolig.csv"),
        pmol,
        df,
    )


def _test_reprotonate():
    pmol, _, df = systemPrepare(Molecule("3PTB"), return_details=True)
    assert df.protonation[df.resid == 40].iloc[0] == "HIE"
    assert df.protonation[df.resid == 57].iloc[0] == "HIP"
    assert df.protonation[df.resid == 91].iloc[0] == "HID"

    pmol.mutateResidue("protein and resid 40", "HID")
    _, _, df2 = systemPrepare(pmol, titration=False, return_details=True)
    assert df2.protonation[df2.resid == 40].iloc[0] == "HID"
    assert df2.protonation[df2.resid == 57].iloc[0] == "HIP"
    assert df2.protonation[df2.resid == 91].iloc[0] == "HID"

    pmol, _, df = systemPrepare(
        Molecule("3PTB"),
        force_protonation=(
            ("protein and resid 40", "HID"),
            ("protein and resid 91", "HIE"),
        ),
        return_details=True,
    )
    assert df.protonation[df.resid == 40].iloc[0] == "HID"
    assert df.protonation[df.resid == 57].iloc[0] == "HIP"
    assert df.protonation[df.resid == 91].iloc[0] == "HIE"


def _test_auto_freezing():
    test_home = os.path.join(curr_dir, "test_systemprepare", "test-auto-freezing")
    mol = Molecule(os.path.join(test_home, "2B5I.pdb"))

    pmol, _, df = systemPrepare(mol, return_details=True, hold_nonpeptidic_bonds=True)
    _compare_results(
        os.path.join(test_home, "2B5I_prepared.pdb"),
        os.path.join(test_home, "2B5I_prepared.csv"),
        pmol,
        df,
    )


def _test_auto_freezing_and_force():
    test_home = os.path.join(curr_dir, "test_systemprepare", "test-auto-freezing")
    mol = Molecule(os.path.join(test_home, "5DPX_A.pdb"))

    pmol, _, df = systemPrepare(
        mol,
        return_details=True,
        hold_nonpeptidic_bonds=True,
        force_protonation=[
            ("protein and resid 105 and chain A", "HIE"),
            ("protein and resid 107 and chain A", "HIE"),
            ("protein and resid 110 and chain A", "HIE"),
            ("protein and resid 181 and chain A", "HIE"),
            ("protein and resid 246 and chain A", "HIE"),
        ],
    )
    _compare_results(
        os.path.join(test_home, "5DPX_A_prepared.pdb"),
        os.path.join(test_home, "5DPX_A_prepared.csv"),
        pmol,
        df,
    )


@pytest.mark.parametrize("system", ("1A4W", "5VBL", "2QRV"))
def _test_nonstandard_residues(tmp_path, system):
    """The expected flow: caller runs ``detectNonStandardResidues`` to find
    chain-resident NCAAs, templates each one via
    ``mol.templateResidueFromSmiles``, then passes the spec list back to
    ``systemPrepare`` via ``detect_specs=``. Reference data was generated
    with the now-removed ``residue_smiles=`` argument; the new flow must
    produce equivalent output.

    2QRV's only non-canonical is the SAH cofactor (a free ligand, not an
    NCAA). systemPrepare no longer adds free ligands to its FF library,
    but we still call ``templateResidueFromSmiles`` on SAH so its bonds
    survive the bond-capture/restore round-trip across PDB2PQR.
    """
    from moleculekit.tools.nonstandard_residues import detectNonStandardResidues
    from moleculekit.tools.autosegment import autoSegment2
    from moleculekit.molecule import mol_equal
    import glob

    test_home = os.path.join(
        curr_dir, "test_systemprepare", "test-nonstandard-residues", system
    )

    # RCSB-style carboxyl SMILES; ``_try_strip_unmatched_terminals`` in
    # rdkittools drops the OXT -OH for mid-chain residues automatically.
    res_smiles = {
        "200": "c1cc(ccc1C[C@@H](C(=O)O)N)Cl",
        "ALC": "C1CCC(CC1)C[C@@H](C(=O)O)N",
        "HRG": "C(CCNC(=N)N)C[C@@H](C(=O)O)N",
        "NLE": "CCCC[C@@H](C(=O)O)N",
        "OIC": "C1CC[C@H]2[C@@H](C1)C[C@H](N2)C(=O)O",
        "TYS": "c1cc(ccc1C[C@@H](C(=O)O)N)OS(=O)(=O)O",
        "SAH": "c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CSCC[C@@H](C(=O)O)N)O)O)N",
    }
    mol = Molecule(os.path.join(test_home, f"{system}.pdb"))
    if system == "2QRV":
        mol.filter("chain D E K M")
        mol = autoSegment2(mol, fields=("chain", "segid"))
    mol.remove("element H", _logger=False)
    mol.set("chain", "W", sel="water")

    specs = detectNonStandardResidues(mol)
    # Template every detected non-canonical residue we have a SMILES for
    # so its connectivity and protonation are well-defined before
    # systemPrepare. Only NCAA / crosslinked-NCAA specs get FF-templated
    # inside systemPrepare; free ligands like SAH only need their bonds
    # set so the bond-restore step preserves them.
    for resn in {s.resname for s in specs}:
        if resn in res_smiles:
            mol.templateResidueFromSmiles(
                f"resname '{resn}'", res_smiles[resn], addHs=True, _logger=False
            )

    pmol, _, df = systemPrepare(
        mol,
        return_details=True,
        hold_nonpeptidic_bonds=True,
        detect_specs=specs,
    )
    pmol.write(os.path.join(tmp_path, "prepared.pdb"))
    if system == "2QRV":
        # The LYS hydrogens are placed differently on Mac/Windows builds
        # of pdb2pqr. Remove them to make the test more stable.
        _ = pmol.remove("resname LYS and element H")

    _compare_results(
        os.path.join(test_home, f"{system}_prepared.pdb"),
        os.path.join(test_home, f"{system}_prepared.csv"),
        pmol,
        df,
    )
    for ff in glob.glob(os.path.join(tmp_path, "residue_cifs", "*.cif")):
        cif1 = Molecule(ff)
        ffref = os.path.join(test_home, "residue_cifs", os.path.basename(ff))
        cif2 = Molecule(ffref)
        assert mol_equal(
            cif1,
            cif2,
            checkFields=Molecule._all_fields,
            exceptFields=["fileloc"],
            fieldPrecision={"coords": 1e-3},
        ), f"Failed comparison of {ff} vs {ffref}"


def _test_nonstandard_residue_hard_ignore_ns():
    test_home = os.path.join(
        curr_dir, "test_systemprepare", "test-nonstandard-residues"
    )
    mol = Molecule(os.path.join(test_home, "5VBL", "5VBL.pdb"))

    pmol, _, df = systemPrepare(
        mol,
        return_details=True,
        hold_nonpeptidic_bonds=True,
        detect_specs=[],
    )
    _compare_results(
        os.path.join(test_home, "5VBL", "5VBL_prepared_ignore_ns.pdb"),
        os.path.join(test_home, "5VBL", "5VBL_prepared_ignore_ns.csv"),
        pmol,
        df,
    )


def _test_rna_protein_complex():
    test_home = os.path.join(curr_dir, "test_systemprepare", "test-rna-protein-complex")
    mol = Molecule(os.path.join(test_home, "3WBM.pdb"))

    pmol, _, df = systemPrepare(mol, return_details=True)

    _compare_results(
        os.path.join(test_home, "3WBM_prepared.pdb"),
        os.path.join(test_home, "3WBM_prepared.csv"),
        pmol,
        df,
    )


def _test_dna():
    test_home = os.path.join(curr_dir, "test_systemprepare", "test-dna")
    mol = Molecule(os.path.join(test_home, "1BNA.pdb"))

    pmol, _, df = systemPrepare(mol, return_details=True)

    _compare_results(
        os.path.join(test_home, "1BNA_prepared.pdb"),
        os.path.join(test_home, "1BNA_prepared.csv"),
        pmol,
        df,
    )


def _test_cyclic_peptides():
    test_home = os.path.join(curr_dir, "test_systemprepare", "test-cyclic-peptides")
    mol = Molecule(os.path.join(test_home, "5VAV.pdb"))

    pmol, _, df = systemPrepare(mol, return_details=True)

    _compare_results(
        os.path.join(test_home, "5VAV_prepared.pdb"),
        os.path.join(test_home, "5VAV_prepared.csv"),
        pmol,
        df,
    )


def _test_cyclic_peptides_noncanonical():
    """4TOT_E is a cyclic peptide of seven NCAAs. Strip any input
    hydrogens up-front so the test follows the canonical
    ``templateResidueFromSmiles(addHs=True)`` pattern shared with the
    other ``detect_specs``-based tests."""
    from moleculekit.tools.nonstandard_residues import (
        detectNonStandardResidues,
        ChainResidueSpec,
    )

    test_home = os.path.join(curr_dir, "test_systemprepare", "test-cyclic-peptides")
    mol = Molecule(os.path.join(test_home, "4TOT_E.pdb"))
    mol.remove("element H", _logger=False)

    smiles = {
        "DAL": "C[C@@H](C(=O)O)N",
        "MLE": "CC(C)C[C@@H](C(=O)O)NC",
        "MVA": "CC(C)[C@@H](C(=O)O)NC",
        "BMT": "CC=CC[C@@H](C)[C@H](O)[C@@H](NC)C(=O)O",
        "ABA": "CC[C@@H](C(=O)O)N",
        "33X": "CN[C@H](C)C(=O)O",
        "34E": "CN[C@@H]([C@H](C)CN1CCN(CCOC)CC1)C(=O)O",
    }

    specs = detectNonStandardResidues(mol)
    ncaa_resnames = {s.resname for s in specs if isinstance(s, ChainResidueSpec)}
    for resn in ncaa_resnames:
        mol.templateResidueFromSmiles(
            f"resname '{resn}'", smiles[resn], addHs=True, _logger=False
        )

    pmol, _, df = systemPrepare(mol, return_details=True, detect_specs=specs)

    _compare_results(
        os.path.join(test_home, "4TOT_E_prepared.pdb"),
        os.path.join(test_home, "4TOT_E_prepared.csv"),
        pmol,
        df,
    )


def _test_nucleiclike_ligand():
    # The nucleic preparation of systemPrepare was accidentally removing the P atom from the ligand
    # since it looked like a terminal nucleic acid phosphate. This test checks that the P atom is still there
    test_home = os.path.join(curr_dir, "test_systemprepare", "3U5S")
    mol = Molecule(os.path.join(test_home, "3U5S.pdb"))

    pmol, _, df = systemPrepare(mol, return_details=True, detect_specs=[])

    _compare_results(
        os.path.join(test_home, "3U5S_prepared.pdb"),
        os.path.join(test_home, "3U5S_prepared.csv"),
        pmol,
        df,
    )


def _test_disabling_titration():
    test_home = os.path.join(curr_dir, "test_systemprepare", "1AID")
    mol = Molecule(os.path.join(test_home, "1AID.pdb"))
    mol.remove("water")

    pmol_ref, _, df_ref = systemPrepare(mol, return_details=True)

    assert df_ref.protonation[df_ref.resid == 25].iloc[0] == "ASH"
    assert df_ref.protonation[df_ref.resid == 69].iloc[0] == "HID"
    assert df_ref.protonation[df_ref.resid == 69].iloc[1] == "HID"

    pmol, _, df = systemPrepare(mol, return_details=True, titrate="HIS")

    assert df.protonation[df.resid == 25].iloc[0] == "ASP"
    assert df.protonation[df.resid == 69].iloc[0] == "HID"
    assert df.protonation[df.resid == 69].iloc[1] == "HID"

    # Delete the resid 25 from both dataframes and compare
    df_ref = df_ref[df_ref.resid != 25]
    df = df[df.resid != 25]
    assert df_ref.equals(df)

    # Delete resid 25 from pmol_ref and pmol and compare
    pmol_ref.remove("resid 25")
    pmol.remove("resid 25")
    assert mol_equal(pmol_ref, pmol, exceptFields=["fileloc"])


def _test_backbone_fixing():
    from moleculekit.tools.backbone import check_backbone
    from moleculekit.molecule import Molecule

    # Remove lots of heavy atoms from the terminal residues to check if they got deleted
    mol = Molecule("3PTB")
    assert np.where(mol.resid == 16)[0].size > 0
    mol.remove("resid 16 and not name N")  # Keep just the N atom
    check_backbone(mol)
    assert np.where(mol.resid == 16)[0].size == 0

    # Remove a backbone O atom from a residue and check if it got reconstructed
    mol = Molecule("3PTB")
    mol.remove("resid 20 and name O")
    assert not np.any((mol.name == "O") & (mol.resid == 20))
    check_backbone(mol)
    assert np.sum((mol.name == "O") & (mol.resid == 20)) == 1

    # Remove a backbone N atom from a residue and check if it got reconstructed
    mol = Molecule("3PTB")
    mol.remove("resid 20 and name N")
    assert not np.any((mol.name == "N") & (mol.resid == 20))
    check_backbone(mol)
    assert np.sum((mol.name == "N") & (mol.resid == 20)) == 1

    # Remove a backbone C atom from a residue and check if it got reconstructed
    mol = Molecule("3PTB")
    mol.remove("resid 20 and name C")
    assert not np.any((mol.name == "C") & (mol.resid == 20))
    check_backbone(mol)
    assert np.sum((mol.name == "C") & (mol.resid == 20)) == 1

    # Remove a backbone CA atom from a residue and check if it got reconstructed
    mol = Molecule("3PTB")
    mol.remove("resid 20 and name CA")
    assert not np.any((mol.name == "CA") & (mol.resid == 20))
    check_backbone(mol)
    assert np.sum((mol.name == "CA") & (mol.resid == 20)) == 1

    # Remove everything except the N CA of a c-terminal residue and check if the C atom got added back
    mol = Molecule("3PTB")
    mol.filter("protein")
    mol.remove("resid 245 and not name N CA")
    assert not np.any((mol.name == "C") & (mol.resid == 245))
    check_backbone(mol)
    assert np.sum((mol.name == "C") & (mol.resid == 245)) == 1


def _test_capture_and_restore_bonds():
    """_capture_bonds must capture every bond touching a non-canonical
    or spec-listed residue, drop bonds entirely inside canonical / non-
    spec residues, and _restore_bonds must silently drop bonds whose
    missing endpoint is hydrogen while warning when a heavy atom goes
    missing.
    """
    import logging
    from moleculekit.tools.preparation import _capture_bonds, _restore_bonds
    from moleculekit.tools.nonstandard_residues import LigandSpec
    from moleculekit.molecule import UniqueResidueID

    class _CaptureWarnings(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.WARNING)
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    prep_logger = logging.getLogger("moleculekit.tools.preparation")

    def _attach_handler():
        h = _CaptureWarnings()
        prep_logger.addHandler(h)
        return h

    # Build a 4-atom mol: C1-C2 (intra), C2-H1 (H bond), C1-N1 (inter-residue).
    mol = Molecule().empty(4)
    mol.coords = np.zeros((4, 3, 1), dtype=np.float32)
    mol.coords[:, 0, 0] = [0.0, 1.5, 2.5, 3.0]
    mol.name[:] = ["C1", "C2", "H1", "N1"]
    mol.element[:] = ["C", "C", "H", "N"]
    mol.resname[:] = ["LIG", "LIG", "LIG", "RES"]
    mol.resid[:] = [1, 1, 1, 2]
    mol.chain[:] = ["A"] * 4
    mol.segid[:] = ["L"] * 4
    mol.bonds = np.array([[0, 1], [1, 2], [0, 3]], dtype=np.uint32)
    mol.bondtype = np.array(["2", "1", "ar"], dtype=object)  # mixed types

    # Non-canonical resnames trigger capture even with empty specs.
    captured = _capture_bonds(mol, detect_specs=[])
    assert len(captured) == 3, "non-canonical resnames must trigger capture"
    assert [t[3] for t in captured] == ["2", "1", "ar"], "bondtype must be captured"

    h_flags = [t[2] for t in captured]
    assert h_flags == [False, True, False], "is_h_bond must be set per bond"

    # Restore onto a copy with a resname rename (CYS->CYX style); H still present.
    mol2 = mol.copy()
    mol2.resname[mol2.resid == 1] = "LGX"
    mol2.bonds = np.zeros((0, 2), dtype=np.uint32)
    mol2.bondtype = np.array([], dtype=object)
    _restore_bonds(mol2, captured)
    assert len(mol2.bonds) == 3, "all bonds must be restored when atoms still present"
    assert list(mol2.bondtype) == ["2", "1", "ar"], "bondtype must round-trip"

    # Restore onto a copy where the hydrogen was removed: drop silently.
    mol3 = mol.copy()
    mol3.remove("name H1", _logger=False)
    mol3.bonds = np.zeros((0, 2), dtype=np.uint32)
    mol3.bondtype = np.array([], dtype=object)
    h = _attach_handler()
    try:
        _restore_bonds(mol3, captured)
    finally:
        prep_logger.removeHandler(h)
    assert len(mol3.bonds) == 2, "C2-H bond must be dropped, heavy bonds kept"
    assert not h.messages, f"missing H must not warn; got {h.messages}"

    # Restore onto a copy where a heavy atom was removed: warn.
    mol4 = mol.copy()
    mol4.remove("name N1", _logger=False)
    mol4.bonds = np.zeros((0, 2), dtype=np.uint32)
    mol4.bondtype = np.array([], dtype=object)
    h = _attach_handler()
    try:
        _restore_bonds(mol4, captured)
    finally:
        prep_logger.removeHandler(h)
    assert len(mol4.bonds) == 2, "C1-N bond must be dropped"
    assert h.messages, "missing heavy atom must warn"

    # Bonds entirely inside canonical, non-spec residues must be dropped.
    # Builders rebuild them from FF templates more correctly than a
    # name-based restore can (PDB2PQR renames RNA OP1/OP2 -> O1P/O2P).
    mol_can = Molecule().empty(3)
    mol_can.coords = np.zeros((3, 3, 1), dtype=np.float32)
    mol_can.coords[:, 0, 0] = [0.0, 1.5, 3.0]
    mol_can.name[:] = ["N", "CA", "C"]
    mol_can.element[:] = ["N", "C", "C"]
    mol_can.resname[:] = ["ALA", "ALA", "ALA"]
    mol_can.resid[:] = [1, 1, 1]
    mol_can.chain[:] = ["A"] * 3
    mol_can.segid[:] = ["P"] * 3
    mol_can.bonds = np.array([[0, 1], [1, 2]], dtype=np.uint32)
    mol_can.bondtype = np.array(["1", "1"], dtype=object)
    assert _capture_bonds(mol_can, detect_specs=[]) == [], (
        "bonds inside a canonical, non-spec residue must be dropped"
    )

    # Listing the canonical residue in detect_specs flips capture back on.
    spec = LigandSpec(
        resname="ALA",
        residue=UniqueResidueID(
            resname="ALA", chain="A", resid=1, insertion="", segid="P"
        ),
    )
    captured_can = _capture_bonds(mol_can, detect_specs=[spec])
    assert len(captured_can) == 2, "spec-listed canonical residue must be captured"


def _test_restore_termini_bonds_terminal_atoms():
    """``_restore_termini_bonds`` must reattach the standard terminal
    atoms PDB2PQR adds after ``_capture_bonds`` has run:

    - CTERM: OXT bonded to C
    - NEUTRAL-CTERM: HO bonded to OXT
    - NTERM: H2 and H3 bonded to N (H is the pre-existing amide H,
              still bonded via ``_restore_bonds``)

    Atoms already bonded (e.g. existing OXT recovered by name) must be
    left alone; partner atoms missing from the residue (e.g. no C) skip
    the bond silently.
    """
    from moleculekit.tools.preparation import _restore_termini_bonds

    # Residue 1: N-terminal CYS-like. Pre-existing N-H bond; H2 and H3
    # need to be reattached to N.
    # Residue 2: C-terminal residue with OXT and HO orphans. C is the
    # backbone carbon to bond OXT to.
    # Residue 3: a stray orphan H that's NOT named H2/H3 - must not be
    # touched.
    mol = Molecule().empty(11)
    mol.coords = np.zeros((11, 3, 1), dtype=np.float32)
    mol.name[:] = ["N", "H", "H2", "H3", "C", "O", "OXT", "HO", "N", "CA", "HG"]
    mol.element[:] = ["N", "H", "H", "H", "C", "O", "O", "H", "N", "C", "H"]
    mol.resname[:] = ["XX1"] * 4 + ["XX2"] * 4 + ["RES"] * 3
    mol.resid[:] = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
    mol.chain[:] = ["A"] * 11
    mol.segid[:] = ["P"] * 11
    mol.insertion[:] = [""] * 11

    # Pre-existing: N-H (captured by _restore_bonds) and C=O.
    mol.bonds = np.array([[0, 1], [4, 5]], dtype=np.uint32)
    mol.bondtype = np.array(["1", "2"], dtype=object)

    _restore_termini_bonds(mol)

    bondset = {frozenset((int(a), int(b))) for a, b in mol.bonds}
    assert frozenset((0, 1)) in bondset, "pre-existing N-H must be preserved"
    assert frozenset((4, 5)) in bondset, "pre-existing C=O must be preserved"
    assert frozenset((0, 2)) in bondset, "H2 must be bonded to N (NTERM patch)"
    assert frozenset((0, 3)) in bondset, "H3 must be bonded to N (NTERM patch)"
    assert frozenset((4, 6)) in bondset, "OXT must be bonded to C (CTERM patch)"
    assert frozenset((6, 7)) in bondset, "HO must be bonded to OXT (NEUTRAL-CTERM)"
    # The stray HG in residue 3 has no standard-terminus name; it must
    # remain unbonded.
    assert not any(10 in pair for pair in bondset), "non-terminal orphan must be left alone"
    assert len(mol.bonds) == len(mol.bondtype), "bonds/bondtype length mismatch"


def _test_restore_termini_bonds_idempotent():
    """Running ``_restore_termini_bonds`` twice must be a no-op the
    second time - the terminal atoms are already bonded."""
    from moleculekit.tools.preparation import _restore_termini_bonds

    mol = Molecule().empty(4)
    mol.coords = np.zeros((4, 3, 1), dtype=np.float32)
    mol.name[:] = ["N", "H2", "H3", "C"]
    mol.element[:] = ["N", "H", "H", "C"]
    mol.resname[:] = ["XX1"] * 4
    mol.resid[:] = [1, 1, 1, 1]
    mol.chain[:] = ["A"] * 4
    mol.segid[:] = ["P"] * 4
    mol.bonds = np.zeros((0, 2), dtype=np.uint32)
    mol.bondtype = np.array([], dtype=object)

    _restore_termini_bonds(mol)
    n_after_first = len(mol.bonds)
    _restore_termini_bonds(mol)
    assert len(mol.bonds) == n_after_first, "second pass must not add bonds"
    assert n_after_first == 2, "expected exactly N-H2 and N-H3"


def _heavy_bond_signatures(mol, sel):
    """Return a set of frozenset signatures for heavy-atom bonds whose
    endpoints both belong to ``sel``. Each signature is a pair of
    (segid, chain, resid, insertion, name) tuples — order-insensitive.
    """
    idx = set(int(i) for i in mol.atomselect(sel, indexes=True))
    sigs = set()
    for a, b in mol.bonds:
        a, b = int(a), int(b)
        if a not in idx or b not in idx:
            continue
        if mol.element[a] == "H" or mol.element[b] == "H":
            continue
        ka = (
            str(mol.segid[a]),
            str(mol.chain[a]),
            int(mol.resid[a]),
            str(mol.insertion[a]),
            str(mol.name[a]),
        )
        kb = (
            str(mol.segid[b]),
            str(mol.chain[b]),
            int(mol.resid[b]),
            str(mol.insertion[b]),
            str(mol.name[b]),
        )
        sigs.add(frozenset([ka, kb]))
    return sigs


def _test_5vbl_templated_bonds_preserved():
    """systemPrepare must preserve every heavy-atom bond of templated
    non-canonical residues across the PDB2PQR roundtrip. 5VBL contains
    five NCAAs (HRG, ALC, OIC, NLE, 200) plus a Zn ion and an OLC
    ligand; we template the NCAAs via templateResidueFromSmiles (the
    canonical entry point — ``residue_smiles=`` on systemPrepare is
    being deprecated) and run with ``detect_specs=[]`` so pdb2pqr leaves
    the NCAAs alone but the bond capture/restore round-trip still has
    to put their connectivity back.
    """
    from moleculekit.tools.preparation import systemPrepare

    smiles = {
        "HRG": "C(CCNC(=N)N)C[C@@H](C(=O)O)N",
        "ALC": "C1CCC(CC1)CC(C(=O)O)N",
        "OIC": "C1CCC2C(C1)CC(N2)C(=O)O",
        "NLE": "CCCC[C@@H](C(=O)O)N",
        "200": "c1cc(ccc1CC(C(=O)O)N)Cl",
    }

    mol = Molecule("5VBL")
    for resname, smi in smiles.items():
        mol.templateResidueFromSmiles(
            f"resname '{resname}'", smi, addHs=True, _logger=False
        )

    sel = "resname " + " ".join(f"'{r}'" for r in smiles)
    expected = _heavy_bond_signatures(mol, sel)
    assert expected, "templated NCAAs must have heavy bonds in input"

    pmol, _ = systemPrepare(mol, detect_specs=[], verbose=False)
    got = _heavy_bond_signatures(pmol, sel)

    missing = expected - got
    assert not missing, (
        f"systemPrepare dropped {len(missing)} heavy bond(s) of templated "
        f"residues: {sorted(missing)}"
    )


def _test_systemprepare_errors_on_untemplated_ncaa():
    """Regression: when ``detect_specs`` contains a chain-resident NCAA
    that the caller forgot to template via
    ``Molecule.templateResidueFromSmiles``, ``systemPrepare`` must fail
    fast with an actionable message — not let PDB2PQR match the NCAA to
    its nearest canonical (e.g. ALC -> LEU, NLE -> LYS) and then crash
    late in ``_assert_specs_bonded`` blaming ``_restore_termini_bonds``.

    Triggering pattern: 5VBL with HRG/OIC/200 templated but ALC and NLE
    left untemplated. Before the fix this raised
    ``RuntimeError: ... renamed canonical residues have unbonded atoms
    ... ['ALC9:A:H', 'NLE15:A:H']`` from inside the PDB2PQR roundtrip.
    After the fix, ``_assert_specs_templated`` fires before PDB2PQR with
    a message naming the untemplated NCAAs and pointing at
    ``templateResidueFromSmiles``.
    """
    from moleculekit.tools.preparation import systemPrepare

    # Template every NCAA in 5VBL EXCEPT ALC and NLE.
    smiles = {
        "HRG": "C(CCNC(=N)N)C[C@@H](C(=O)O)N",
        "OIC": "C1CCC2C(C1)CC(N2)C(=O)O",
        "200": "c1cc(ccc1CC(C(=O)O)N)Cl",
    }

    mol = Molecule("5VBL")
    mol.filter("not water", _logger=False)
    for resname, smi in smiles.items():
        mol.templateResidueFromSmiles(
            f"resname '{resname}'", smi, addHs=True, _logger=False
        )

    with pytest.raises(RuntimeError, match="have not been templated") as exc:
        systemPrepare(mol, verbose=False)

    msg = str(exc.value)
    assert "ALC9:A" in msg, f"error should name ALC9:A, got: {msg}"
    assert "NLE15:A" in msg, f"error should name NLE15:A, got: {msg}"
    assert "templateResidueFromSmiles" in msg, (
        f"error should point at templateResidueFromSmiles, got: {msg}"
    )


def _test_5vbl_restore_missing_sidechains():
    """``systemPrepare(restore_missing_sidechains=True)`` rebuilds the
    full heavy-atom sidechain of every canonical residue whose input
    sidechain is stripped down to backbone + CB.

    Fixture ``5VBL_A.cif`` is a hand-trimmed chain-A copy of 5VBL where
    five canonical residues (LYS1, PHE2, ARG3, ARG4, LYS12) are
    truncated to N/CA/C/O/CB. PDB2PQR would normally reject this
    structure (>10%% heavy atoms missing); with
    ``restore_missing_sidechains=True`` moleculekit's Dunbrack-rotamer
    mutator regrows the sidechains before the PDB2PQR roundtrip, the
    5 NCAAs (HRG, ALC, OIC, NLE, 200) keep their templated atoms, and
    every truncated residue ends up with its canonical heavy-atom set.
    """
    from moleculekit.tools.preparation import systemPrepare

    fixture = os.path.join(curr_dir, "test_systemprepare", "5VBL_A.cif")
    mol = Molecule(fixture)

    # NCAAs in 5VBL — same SMILES as the rest of the 5VBL suite.
    smiles = {
        "HRG": "C(CCNC(=N)N)C[C@@H](C(=O)O)N",
        "ALC": "C1CCC(CC1)CC(C(=O)O)N",
        "OIC": "C1CCC2C(C1)CC(N2)C(=O)O",
        "NLE": "CCCC[C@@H](C(=O)O)N",
        "200": "c1cc(ccc1CC(C(=O)O)N)Cl",
    }
    for resname, smi in smiles.items():
        mol.templateResidueFromSmiles(
            f"resname '{resname}'", smi, addHs=True, _logger=False
        )

    # Canonical heavy-atom sets we expect after reconstruction. Keys are
    # the truncated residues in the fixture (resid -> resname).
    expected_heavy = {
        (1, "LYS"): {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
        (2, "PHE"): {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
        (3, "ARG"): {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
        (4, "ARG"): {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
        (12, "LYS"): {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    }

    # Sanity-check the fixture: each target residue must be truncated to
    # backbone + CB on input, otherwise the test no longer exercises
    # reconstruction.
    for (resid, resname), _ in expected_heavy.items():
        mask = (mol.resid == resid) & (mol.resname == resname)
        heavy = set(mol.name[mask & (mol.element != "H")])
        assert heavy == {"N", "CA", "C", "O", "CB"}, (
            f"fixture {resname}{resid} should be backbone+CB only, got {heavy}"
        )

    pmol, _ = systemPrepare(mol, verbose=False, restore_missing_sidechains=True)

    for (resid, resname), expected in expected_heavy.items():
        mask = (pmol.resid == resid) & (pmol.resname == resname)
        assert mask.any(), f"{resname}{resid} missing from output"
        heavy = set(pmol.name[mask & (pmol.element != "H")])
        assert heavy == expected, (
            f"{resname}{resid} heavy atoms after systemPrepare: "
            f"got {sorted(heavy)}, expected {sorted(expected)}"
        )

    # Templated NCAAs must survive the roundtrip with their heavy atoms
    # intact (separate guarantee from sidechain reconstruction).
    ncaa_residues = {
        (8, "HRG"): 12,
        (9, "ALC"): 11,
        (14, "OIC"): 11,
        (15, "NLE"): 8,
        (17, "200"): 13,
    }
    for (resid, resname), n_heavy in ncaa_residues.items():
        mask = (pmol.resid == resid) & (pmol.resname == resname)
        assert mask.any(), f"NCAA {resname}{resid} lost from output"
        heavy = (mask & (pmol.element != "H")).sum()
        assert heavy == n_heavy, (
            f"{resname}{resid}: expected {n_heavy} heavy atoms, got {heavy}"
        )


def _test_no_oxt_on_midchain_residue():
    """Regression: PDB2PQR must not place OXT on a residue that's
    followed by another protein residue in the same chain. This guards
    against the bug where renaming a C-terminal residue to a custom
    resname (XX## via the spec pipeline) caused PDB2PQR to demote the
    previous residue to 'C-terminus' and add OXT to it.

    Triggering structure: 8QFZ chain B, where CYS 22 is the C-terminus
    and gets renamed to XX# by the new spec pipeline. Before the fix,
    PHE 21 (the residue before XX#) ended up with OXT because PDB2PQR
    didn't recognise XX# as a protein residue. After the fix (always
    registering custom-named canonical anchors via
    _generate_nonstandard_residues_ff) XX# is recognised and OXT
    correctly lands on it instead of PHE 21.
    """
    from moleculekit.util import sequenceID

    fixture = os.path.join(
        curr_dir, "test_nonstandard_residues", "8QFZ_B.cif"
    )
    mol = Molecule(fixture)
    pmol, _ = systemPrepare(mol, verbose=False)

    # Group atoms into residues, then identify the last residue per
    # (segid, chain) protein segment.
    res_idx = sequenceID((pmol.resid, pmol.insertion, pmol.chain, pmol.segid))
    protein_mask = pmol.atomselect("protein")

    bad = []
    for i in np.where(pmol.name == "OXT")[0]:
        if not protein_mask[i]:
            continue  # OXT on non-protein (e.g. nucleic O3') is unrelated
        seg, chain = str(pmol.segid[i]), str(pmol.chain[i])
        seg_mask = (pmol.segid == seg) & (pmol.chain == chain) & protein_mask
        last_res = int(res_idx[seg_mask].max())
        if int(res_idx[i]) != last_res:
            bad.append(
                f"{pmol.resname[i]}{pmol.resid[i]}{pmol.insertion[i]}:{chain}"
            )
    assert not bad, (
        f"Found OXT on non-C-terminal residue(s): {bad}. "
        f"PDB2PQR placed OXT on a residue that's followed by another "
        f"protein residue in the same chain."
    )


def _test_hydrogen_bonds_match_geometry():
    """Regression: every restored H bond must connect the H to its
    geometrically nearest heavy atom (within typical covalent X-H
    distance ~1.3 A).

    Trigger: 8QFZ chain B XX1 (N-terminal CYS templated as a canonical
    anchor with a sidechain crosslink). Before the fix, bond capture
    grabbed rdkit's generic name 'H3' for CA's hydrogen, and PDB2PQR
    later created its own atom named 'H3' (the third N-terminal NH3+
    H) at a different position. The name-based bond restore resolved
    the captured CA-H3 bond to PDB2PQR's new H3, leaving CA with 5
    bonds and a 2.04 A phantom C-H bond. Antechamber then typed CA as
    DU and the cluster parameterization crashed.

    The fix moves _canonicalize_ncaa_h_names (H1->H, H3->HA) to run
    BEFORE bond capture so captured names are stable AMBER names that
    don't collide with anything PDB2PQR may add later.
    """
    fixture = os.path.join(
        curr_dir, "test_nonstandard_residues", "8QFZ_B.cif"
    )
    mol = Molecule(fixture)
    pmol, _ = systemPrepare(mol, verbose=False)

    bad = []
    for a, b in pmol.bonds:
        a, b = int(a), int(b)
        ea, eb = pmol.element[a], pmol.element[b]
        if ea != "H" and eb != "H":
            continue
        d = float(np.linalg.norm(pmol.coords[a, :, 0] - pmol.coords[b, :, 0]))
        if d > 1.3:
            h_idx, hv_idx = (a, b) if ea == "H" else (b, a)
            bad.append(
                f"{pmol.resname[hv_idx]}{pmol.resid[hv_idx]}:"
                f"{pmol.chain[hv_idx]}:{pmol.name[hv_idx]}-{pmol.name[h_idx]} "
                f"d={d:.2f}"
            )
    assert not bad, (
        f"Hydrogens bonded to topologically wrong heavy atom (d > 1.3 A): "
        f"{bad}"
    )


def _build_spec_mol():
    """Two-residue mol: a spec residue LIG (resid 1) and a canonical
    residue ALA (resid 2). Used by the formal-charge unit tests."""
    mol = Molecule().empty(5)
    mol.coords = np.zeros((5, 3, 1), dtype=np.float32)
    mol.name[:] = ["C1", "N1", "O1", "N", "CA"]
    mol.element[:] = ["C", "N", "O", "N", "C"]
    mol.resname[:] = ["LIG", "LIG", "LIG", "ALA", "ALA"]
    mol.resid[:] = [1, 1, 1, 2, 2]
    mol.chain[:] = ["A"] * 5
    mol.segid[:] = ["P"] * 5
    mol.insertion[:] = [""] * 5
    mol.formalcharge[:] = [0, 1, -1, 1, 0]  # ALA N intentionally charged
    return mol


def _test_capture_formal_charges_empty_specs():
    """``_capture_formal_charges`` must return an empty list when
    ``detect_specs`` is empty - the function is scoped deliberately to
    spec residues so callers without specs (free ligands only) skip the
    capture cost entirely."""
    from moleculekit.tools.preparation import _capture_formal_charges

    mol = _build_spec_mol()
    assert _capture_formal_charges(mol, detect_specs=[]) == []


def _test_capture_formal_charges_scoped_to_specs():
    """Only non-zero charges on atoms inside spec residues must be
    captured. Zero charges are skipped, and charges on canonical
    residues outside ``detect_specs`` must be ignored - PDB2PQR is
    allowed to re-protonate canonicals, so we must not pin their formal
    charges."""
    from moleculekit.tools.preparation import _capture_formal_charges
    from moleculekit.tools.nonstandard_residues import LigandSpec
    from moleculekit.molecule import UniqueResidueID

    mol = _build_spec_mol()
    spec = LigandSpec(
        resname="LIG",
        residue=UniqueResidueID(
            resname="LIG", chain="A", resid=1, insertion="", segid="P"
        ),
    )

    captured = _capture_formal_charges(mol, detect_specs=[spec])
    captured_by_name = {uaid.name: charge for uaid, charge in captured}
    assert captured_by_name == {"N1": 1, "O1": -1}, (
        f"only non-zero charges inside the spec residue must be captured; "
        f"got {captured_by_name}"
    )


def _test_restore_formal_charges_survives_rename():
    """``_restore_formal_charges`` must use the relaxed atom lookup so
    captured charges still land on the right atom after the residue has
    been renamed (CYS->CYX, LIG->LGX). The roundtrip target is a mol
    whose formal charges have been zeroed (PDB2PQR resets them)."""
    from moleculekit.tools.preparation import (
        _capture_formal_charges,
        _restore_formal_charges,
    )
    from moleculekit.tools.nonstandard_residues import LigandSpec
    from moleculekit.molecule import UniqueResidueID

    mol = _build_spec_mol()
    spec = LigandSpec(
        resname="LIG",
        residue=UniqueResidueID(
            resname="LIG", chain="A", resid=1, insertion="", segid="P"
        ),
    )
    captured = _capture_formal_charges(mol, detect_specs=[spec])

    mol2 = mol.copy()
    mol2.resname[mol2.resid == 1] = "LGX"  # rename like CYS->CYX
    mol2.formalcharge[:] = 0  # simulate PDB2PQR zeroing
    _restore_formal_charges(mol2, captured)

    by_name = {str(n): int(c) for n, c in zip(mol2.name, mol2.formalcharge)}
    assert by_name["N1"] == 1, "N1 charge must be restored after rename"
    assert by_name["O1"] == -1, "O1 charge must be restored after rename"
    assert by_name["N"] == 0, (
        "canonical ALA N must not be touched (not in detect_specs)"
    )


def _test_restore_formal_charges_drops_missing_atom():
    """If an atom captured in the input mol has been removed by PDB2PQR
    (rare, but possible for stray Hs), the restore must silently skip
    it rather than raising. Mirrors the behaviour of ``_restore_bonds``
    via the shared ``_find_atom_relaxed`` helper."""
    from moleculekit.tools.preparation import (
        _capture_formal_charges,
        _restore_formal_charges,
    )
    from moleculekit.tools.nonstandard_residues import LigandSpec
    from moleculekit.molecule import UniqueResidueID

    mol = _build_spec_mol()
    spec = LigandSpec(
        resname="LIG",
        residue=UniqueResidueID(
            resname="LIG", chain="A", resid=1, insertion="", segid="P"
        ),
    )
    captured = _capture_formal_charges(mol, detect_specs=[spec])

    mol2 = mol.copy()
    mol2.remove("name N1", _logger=False)
    mol2.formalcharge[:] = 0
    _restore_formal_charges(mol2, captured)  # must not raise

    by_name = {str(n): int(c) for n, c in zip(mol2.name, mol2.formalcharge)}
    assert by_name["O1"] == -1, "remaining captured atom must still be restored"
    assert "N1" not in by_name


def _build_terminus_mol(n_term_h_count, c_term_oxt_h_count):
    """Build a single-residue protein-like mol with both N and OXT
    backbone atoms, parametrised by how many hydrogens hang off each.

    - ``n_term_h_count``: 3 = charged NH3+, 2 = neutral NH2.
    - ``c_term_oxt_h_count``: 0 = COO-, 1 = neutral COOH.
    """
    n_h = n_term_h_count
    oxt_h = c_term_oxt_h_count
    n_atoms = 4 + n_h + oxt_h  # N, CA, C, OXT + Hs on N and OXT
    mol = Molecule().empty(n_atoms)
    mol.coords = np.zeros((n_atoms, 3, 1), dtype=np.float32)

    names = ["N", "CA", "C", "OXT"]
    elems = ["N", "C", "C", "O"]
    bonds = [(0, 1), (1, 2), (2, 3)]  # N-CA, CA-C, C-OXT
    next_idx = 4
    for k in range(n_h):
        names.append(["H", "H2", "H3"][k])
        elems.append("H")
        bonds.append((0, next_idx))  # N-H
        next_idx += 1
    for _ in range(oxt_h):
        names.append("HO")
        elems.append("H")
        bonds.append((3, next_idx))  # OXT-H
        next_idx += 1

    mol.name[:] = names
    mol.element[:] = elems
    mol.resname[:] = ["GLY"] * n_atoms
    mol.resid[:] = [1] * n_atoms
    mol.chain[:] = ["A"] * n_atoms
    mol.segid[:] = ["P"] * n_atoms
    mol.insertion[:] = [""] * n_atoms
    mol.formalcharge[:] = 0  # PDB2PQR zeroes these on output
    mol.bonds = np.array(bonds, dtype=np.uint32)
    mol.bondtype = np.array(["1"] * len(bonds), dtype=object)
    return mol


def _test_apply_terminal_formal_charges_charged_nterm_charged_cterm():
    """3 H on backbone N -> NH3+ (formalcharge +1). 0 H on OXT -> COO-
    (formalcharge -1). These are PDB2PQR's default CTERM / NTERM
    patches at pH 7."""
    from moleculekit.tools.preparation import _apply_terminal_formal_charges
    from moleculekit.tools.nonstandard_residues import ChainResidueSpec
    from moleculekit.molecule import UniqueResidueID

    mol = _build_terminus_mol(n_term_h_count=3, c_term_oxt_h_count=0)
    spec = ChainResidueSpec(
        resname="GLY",
        residue=UniqueResidueID(
            resname="GLY", chain="A", resid=1, insertion="", segid="P"
        ),
        is_n_term=True,
        is_c_term=True,
    )
    _apply_terminal_formal_charges(mol, detect_specs=[spec])

    by_name = {str(n): int(c) for n, c in zip(mol.name, mol.formalcharge)}
    assert by_name["N"] == 1, "3 H on N must give NH3+ (+1)"
    assert by_name["OXT"] == -1, "0 H on OXT must give COO- (-1)"


def _test_apply_terminal_formal_charges_neutral_termini():
    """The NEUTRAL-NTERM / NEUTRAL-CTERM patches leave fewer Hs on N (2)
    and add an H to OXT, both of which the helper must read as neutral
    and leave formalcharge at 0."""
    from moleculekit.tools.preparation import _apply_terminal_formal_charges
    from moleculekit.tools.nonstandard_residues import ChainResidueSpec
    from moleculekit.molecule import UniqueResidueID

    mol = _build_terminus_mol(n_term_h_count=2, c_term_oxt_h_count=1)
    spec = ChainResidueSpec(
        resname="GLY",
        residue=UniqueResidueID(
            resname="GLY", chain="A", resid=1, insertion="", segid="P"
        ),
        is_n_term=True,
        is_c_term=True,
    )
    _apply_terminal_formal_charges(mol, detect_specs=[spec])

    by_name = {str(n): int(c) for n, c in zip(mol.name, mol.formalcharge)}
    assert by_name["N"] == 0, "2 H on N is neutral NH2 - no charge"
    assert by_name["OXT"] == 0, "1 H on OXT is neutral COOH - no charge"


def _test_apply_terminal_formal_charges_skips_non_chain_spec():
    """Only :class:`ChainResidueSpec` entries are touched. A
    :class:`LigandSpec` (a free ligand) must be ignored even if it
    happens to have backbone-named atoms."""
    from moleculekit.tools.preparation import _apply_terminal_formal_charges
    from moleculekit.tools.nonstandard_residues import LigandSpec
    from moleculekit.molecule import UniqueResidueID

    mol = _build_terminus_mol(n_term_h_count=3, c_term_oxt_h_count=0)
    spec = LigandSpec(
        resname="GLY",
        residue=UniqueResidueID(
            resname="GLY", chain="A", resid=1, insertion="", segid="P"
        ),
    )
    _apply_terminal_formal_charges(mol, detect_specs=[spec])

    by_name = {str(n): int(c) for n, c in zip(mol.name, mol.formalcharge)}
    assert by_name["N"] == 0, "LigandSpec must be skipped, N untouched"
    assert by_name["OXT"] == 0, "LigandSpec must be skipped, OXT untouched"


def _test_apply_terminal_formal_charges_skips_midchain_spec():
    """A ChainResidueSpec without either terminus flag (mid-chain NCAA)
    must be left alone - terminal patches don't apply to it."""
    from moleculekit.tools.preparation import _apply_terminal_formal_charges
    from moleculekit.tools.nonstandard_residues import ChainResidueSpec
    from moleculekit.molecule import UniqueResidueID

    mol = _build_terminus_mol(n_term_h_count=3, c_term_oxt_h_count=0)
    spec = ChainResidueSpec(
        resname="GLY",
        residue=UniqueResidueID(
            resname="GLY", chain="A", resid=1, insertion="", segid="P"
        ),
        is_n_term=False,
        is_c_term=False,
    )
    _apply_terminal_formal_charges(mol, detect_specs=[spec])

    by_name = {str(n): int(c) for n, c in zip(mol.name, mol.formalcharge)}
    assert by_name["N"] == 0, "mid-chain spec must not get NTERM treatment"
    assert by_name["OXT"] == 0, "mid-chain spec must not get CTERM treatment"


def _test_apply_terminal_formal_charges_uses_new_resname():
    """When a spec has been renamed (``new_resname`` set, e.g. CYS->CYX
    or LIG->XX1 for a custom anchor), the helper must match on the
    *renamed* resname - because that's what's in ``mol.resname`` after
    ``_apply_detect_spec_renames``."""
    from moleculekit.tools.preparation import _apply_terminal_formal_charges
    from moleculekit.tools.nonstandard_residues import ChainResidueSpec
    from moleculekit.molecule import UniqueResidueID

    mol = _build_terminus_mol(n_term_h_count=3, c_term_oxt_h_count=0)
    # Mimic _apply_detect_spec_renames having renamed the residue:
    mol.resname[:] = "CYX"

    spec = ChainResidueSpec(
        resname="CYS",  # original
        residue=UniqueResidueID(
            resname="CYS", chain="A", resid=1, insertion="", segid="P"
        ),
        new_resname="CYX",  # current resname in mol
        is_n_term=True,
        is_c_term=True,
    )
    _apply_terminal_formal_charges(mol, detect_specs=[spec])

    by_name = {str(n): int(c) for n, c in zip(mol.name, mol.formalcharge)}
    assert by_name["N"] == 1, "must match by new_resname; expected NH3+"
    assert by_name["OXT"] == -1, "must match by new_resname; expected COO-"
