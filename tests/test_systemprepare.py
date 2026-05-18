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


@pytest.mark.parametrize("pdb", ["3PTB", "1A25", "1U5U", "1UNC", "6A5J"])
def _test_systemPrepare(pdb):
    test_home = os.path.join(curr_dir, "test_systemprepare", pdb)
    mol = Molecule(os.path.join(test_home, f"{pdb}.pdb"))
    pmol, _, df = systemPrepare(mol, return_details=True)
    _compare_results(
        os.path.join(test_home, f"{pdb}_prepared.pdb"),
        os.path.join(test_home, f"{pdb}_prepared.csv"),
        pmol,
        df,
    )


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
    """_capture_bonds must capture every bond (intra- and inter-residue),
    and _restore_bonds must silently drop bonds whose missing endpoint
    is hydrogen while still warning when a heavy atom goes missing.
    """
    import logging
    from moleculekit.tools.preparation import _capture_bonds, _restore_bonds

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

    captured = _capture_bonds(mol)
    assert len(captured) == 3, "all bonds (incl. intra-residue) must be captured"
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
