from moleculekit.molecule import Molecule, mol_equal
from moleculekit.tools.preparation import systemPrepare, _table_dtypes
import numpy as np
import os
import pytest
import sys

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
    assert mol_equal(
        refmol, pmol, exceptFields=["serial"], fieldPrecision={"coords": coords_prec}
    ), f"Failed comparison of {refpdb} vs {pmol.fileloc}"


@pytest.mark.parametrize("pdb", ["3PTB", "1A25", "1U5U", "1UNC", "6A5J"])
def _test_systemPrepare(pdb):
    test_home = os.path.join(curr_dir, "test_systemprepare", pdb)
    mol = Molecule(os.path.join(test_home, f"{pdb}.pdb"))
    pmol, df = systemPrepare(mol, return_details=True)
    _compare_results(
        os.path.join(test_home, f"{pdb}_prepared.pdb"),
        os.path.join(test_home, f"{pdb}_prepared.csv"),
        pmol,
        df,
    )


def _test_systemprepare_ligand():
    test_home = os.path.join(curr_dir, "test_systemprepare", "test-prepare-with-ligand")
    mol = Molecule(os.path.join(test_home, "5EK0_A.pdb"))
    pmol, df = systemPrepare(mol, return_details=True)
    _compare_results(
        os.path.join(test_home, "5EK0_A_prepared.pdb"),
        os.path.join(test_home, "5EK0_A_prepared.csv"),
        pmol,
        df,
    )
    # Now remove the ligands and check again what the pka is
    mol.filter('not resname PX4 "5P2"')
    pmol, df = systemPrepare(mol, return_details=True)
    _compare_results(
        os.path.join(test_home, "5EK0_A_prepared_nolig.pdb"),
        os.path.join(test_home, "5EK0_A_prepared_nolig.csv"),
        pmol,
        df,
    )


def _test_reprotonate():
    pmol, df = systemPrepare(Molecule("3PTB"), return_details=True)
    assert df.protonation[df.resid == 40].iloc[0] == "HIE"
    assert df.protonation[df.resid == 57].iloc[0] == "HIP"
    assert df.protonation[df.resid == 91].iloc[0] == "HID"

    pmol.mutateResidue("protein and resid 40", "HID")
    _, df2 = systemPrepare(pmol, titration=False, return_details=True)
    assert df2.protonation[df2.resid == 40].iloc[0] == "HID"
    assert df2.protonation[df2.resid == 57].iloc[0] == "HIP"
    assert df2.protonation[df2.resid == 91].iloc[0] == "HID"

    pmol, df = systemPrepare(
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

    pmol, df = systemPrepare(mol, return_details=True, hold_nonpeptidic_bonds=True)
    _compare_results(
        os.path.join(test_home, "2B5I_prepared.pdb"),
        os.path.join(test_home, "2B5I_prepared.csv"),
        pmol,
        df,
    )


def _test_auto_freezing_and_force():
    test_home = os.path.join(curr_dir, "test_systemprepare", "test-auto-freezing")
    mol = Molecule(os.path.join(test_home, "5DPX_A.pdb"))

    pmol, df = systemPrepare(
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
    from moleculekit.tools.autosegment import autoSegment2
    from moleculekit.molecule import mol_equal
    import glob

    test_home = os.path.join(
        curr_dir, "test_systemprepare", "test-nonstandard-residues", system
    )

    res_smiles = {
        "200": "c1cc(ccc1C[C@@H](C(=O)O)N)Cl",
        "ALC": "C1CCC(CC1)C[C@@H](C=O)N",
        "HRG": "C(CCNC(=N)N)C[C@@H](C=O)N",
        "NLE": "CCCC[C@@H](C=O)N",
        "OIC": "C1CC[C@H]2[C@@H](C1)C[C@H](N2)C=O",
        "TYS": "c1cc(ccc1C[C@@H](C=O)N)OS(=O)(=O)O",
        "SAH": "c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CSCC[C@@H](C(=O)O)N)O)O)N",
    }
    mol = Molecule(os.path.join(test_home, f"{system}.pdb"))
    if system == "2QRV":
        mol.filter("chain D E K M")
        mol = autoSegment2(mol, fields=("chain", "segid"))
    mol.set("chain", "W", sel="water")

    pmol, df = systemPrepare(
        mol,
        return_details=True,
        hold_nonpeptidic_bonds=True,
        residue_smiles=res_smiles,
        outdir=os.path.join(tmp_path, "residue_cifs"),
    )
    pmol.write(os.path.join(tmp_path, "prepared.pdb"))
    if system == "2QRV":
        # The LYS hydrogens are placed differently on Mac/Windows builds of pdb2pqr
        # Remove them to make the test more stable
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

    pmol, df = systemPrepare(
        mol,
        return_details=True,
        hold_nonpeptidic_bonds=True,
        _molkit_ff=False,
        ignore_ns=True,
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

    pmol, df = systemPrepare(mol, return_details=True)

    _compare_results(
        os.path.join(test_home, "3WBM_prepared.pdb"),
        os.path.join(test_home, "3WBM_prepared.csv"),
        pmol,
        df,
    )


def _test_dna():
    test_home = os.path.join(curr_dir, "test_systemprepare", "test-dna")
    mol = Molecule(os.path.join(test_home, "1BNA.pdb"))

    pmol, df = systemPrepare(mol, return_details=True)

    _compare_results(
        os.path.join(test_home, "1BNA_prepared.pdb"),
        os.path.join(test_home, "1BNA_prepared.csv"),
        pmol,
        df,
    )


def _test_cyclic_peptides():
    test_home = os.path.join(curr_dir, "test_systemprepare", "test-cyclic-peptides")
    mol = Molecule(os.path.join(test_home, "5VAV.pdb"))

    pmol, df = systemPrepare(mol, return_details=True)

    _compare_results(
        os.path.join(test_home, "5VAV_prepared.pdb"),
        os.path.join(test_home, "5VAV_prepared.csv"),
        pmol,
        df,
    )


def _test_cyclic_peptides_noncanonical():
    test_home = os.path.join(curr_dir, "test_systemprepare", "test-cyclic-peptides")
    mol = Molecule(os.path.join(test_home, "4TOT_E.pdb"))

    smiles = {
        "33X": "CC(CO)NC",
        "34E": "CN[C@@H]([C@H](C)CN1CCN(CCOC)CC1)C(O)",
        "BMT": "C/C=C/C[C@@H](C)[C@H]([C@@H](CO)NC)O",
        "DAL": "C[C@H](CO)N",
    }
    pmol, df = systemPrepare(mol, return_details=True, residue_smiles=smiles)

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

    pmol, df = systemPrepare(mol, return_details=True, ignore_ns=True)

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

    pmol_ref, df_ref = systemPrepare(mol, return_details=True)

    assert df_ref.protonation[df_ref.resid == 25].iloc[0] == "ASH"
    assert df_ref.protonation[df_ref.resid == 69].iloc[0] == "HID"
    assert df_ref.protonation[df_ref.resid == 69].iloc[1] == "HID"

    pmol, df = systemPrepare(mol, return_details=True, titrate="HIS")

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
        ka = (str(mol.segid[a]), str(mol.chain[a]), int(mol.resid[a]),
              str(mol.insertion[a]), str(mol.name[a]))
        kb = (str(mol.segid[b]), str(mol.chain[b]), int(mol.resid[b]),
              str(mol.insertion[b]), str(mol.name[b]))
        sigs.add(frozenset([ka, kb]))
    return sigs


def _test_5vbl_templated_bonds_preserved():
    """systemPrepare must preserve every heavy-atom bond of templated
    non-canonical residues across the PDB2PQR roundtrip. 5VBL contains
    five NCAAs (HRG, ALC, OIC, NLE, 200) plus a Zn ion and an OLC
    ligand; we template the NCAAs via templateResidueFromSmiles (the
    canonical entry point — ``residue_smiles=`` on systemPrepare is
    being deprecated) and run with ``ignore_ns=True`` so pdb2pqr leaves
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

    pmol = systemPrepare(mol, ignore_ns=True, verbose=False)
    got = _heavy_bond_signatures(pmol, sel)

    missing = expected - got
    assert not missing, (
        f"systemPrepare dropped {len(missing)} heavy bond(s) of templated "
        f"residues: {sorted(missing)}"
    )



