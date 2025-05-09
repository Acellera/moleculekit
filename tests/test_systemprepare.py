from moleculekit.molecule import Molecule, mol_equal
from moleculekit.tools.preparation import systemPrepare, _table_dtypes
import numpy as np
import os
import pytest

curr_dir = os.path.dirname(os.path.abspath(__file__))

# The below is used for testing only
try:
    from aceprep.prepare import rdk_prepare
except ImportError:
    ACEPREP_EXISTS = False
else:
    ACEPREP_EXISTS = True


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
    assert mol_equal(
        refmol, pmol, exceptFields=["serial"], fieldPrecision={"coords": 1e-3}
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


@pytest.mark.skipif(not ACEPREP_EXISTS, reason="Can only run with aceprep installed")
@pytest.mark.parametrize(
    "files",
    (
        ("1A4W.pdb", "1A4W_prepared"),
        ("5VBL.pdb", "5VBL_prepared"),
        ("2QRV.pdb", "2QRV_prepared"),
    ),
)
def _test_nonstandard_residues(tmp_path, files):
    from moleculekit.tools.autosegment import autoSegment2

    inf, outf = files
    test_home = os.path.join(
        curr_dir, "test_systemprepare", "test-nonstandard-residues"
    )

    res_smiles = {
        "200": "c1cc(ccc1C[C@@H](C(=O)O)N)Cl",
        "HRG": "C(CCNC(=N)N)C[C@@H](C(=O)O)N",
        "OIC": "C1CC[C@H]2[C@@H](C1)C[C@H](N2)C(=O)O",
        "TYS": "c1cc(ccc1C[C@@H](C(=O)O)N)OS(=O)(=O)O",
        "SAH": "c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CSCC[C@@H](C(=O)O)N)O)O)N",
    }
    mol = Molecule(os.path.join(test_home, inf))
    if inf == "2QRV.pdb":
        mol = autoSegment2(mol, fields=("chain", "segid"))

    pmol, df = systemPrepare(
        mol,
        return_details=True,
        hold_nonpeptidic_bonds=True,
        residue_smiles=res_smiles,
    )
    pmol.fileloc.append(os.path.join(tmp_path, "prepared.pdb"))
    pmol.write(pmol.fileloc[0])

    _compare_results(
        os.path.join(test_home, f"{outf}.pdb"),
        os.path.join(test_home, f"{outf}.csv"),
        pmol,
        df,
    )

    pmol, df = systemPrepare(mol, return_details=True, hold_nonpeptidic_bonds=True)
    pmol.fileloc.append(os.path.join(tmp_path, "prepared.pdb"))
    pmol.write(pmol.fileloc[0])

    _compare_results(
        os.path.join(test_home, f"{outf}.pdb"),
        os.path.join(test_home, f"{outf}.csv"),
        pmol,
        df,
    )


@pytest.mark.skipif(ACEPREP_EXISTS, reason="Can only run WITHOUT aceprep installed")
def _test_nonstandard_residue_hard_ignore_ns():
    test_home = os.path.join(
        curr_dir, "test_systemprepare", "test-nonstandard-residues"
    )
    mol = Molecule(os.path.join(test_home, "5VBL.pdb"))

    pmol, df = systemPrepare(
        mol,
        return_details=True,
        hold_nonpeptidic_bonds=True,
        _molkit_ff=False,
        ignore_ns_errors=True,
    )
    _compare_results(
        os.path.join(test_home, "5VBL_prepared_ignore_ns.pdb"),
        os.path.join(test_home, "5VBL_prepared_ignore_ns.csv"),
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


@pytest.mark.skipif(not ACEPREP_EXISTS, reason="Can only run with aceprep installed")
def _test_cyclic_peptides_noncanonical():
    test_home = os.path.join(curr_dir, "test_systemprepare", "test-cyclic-peptides")
    mol = Molecule(os.path.join(test_home, "4TOT_E.pdb"))

    pmol, df = systemPrepare(mol, return_details=True)

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
