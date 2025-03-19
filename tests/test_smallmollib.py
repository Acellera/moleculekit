from moleculekit.smallmol.smallmollib import SmallMolLib
from moleculekit.smallmol.smallmol import SmallMol
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

SDF_N_MOLS = 100

SDF_IDS_DELETE = [1, 10, 15, 16]
SDF_MOLNAME_DELETE = ["ZINC02583363", "ZINC86860147", "ZINC04342657", "ZINC02023420"]
SDF_FIELDS = ["ligname", "_mol"]
SDF_LOC_0_99 = "ZINC02141008"


def _test_loaSdffile():
    sdffile = os.path.join(curr_dir, "test_smallmol", "fda_drugs_light.sdf")
    lib = SmallMolLib(sdffile)
    n_mols = lib.numMols
    assert (
        n_mols == SDF_N_MOLS
    ), f"Molecules not correctly loaded. Expected: {SDF_N_MOLS}; Now: {n_mols}"

    sdffile = os.path.join(curr_dir, "test_smallmol", "fda_drugs_light.sdf.gz")
    lib = SmallMolLib(sdffile)
    n_mols = lib.numMols
    assert (
        n_mols == SDF_N_MOLS
    ), f"Molecules not correctly loaded. Expected: {SDF_N_MOLS}; Now: {n_mols}"


def _test_writeSdf(tmp_path):
    sdffile = os.path.join(curr_dir, "test_smallmol", "fda_drugs_light.sdf")
    lib = SmallMolLib(sdffile)

    sdfname = os.path.join(tmp_path, "test.sdf")
    lib.writeSdf(sdfname)

    sdf_exists = os.path.isfile(sdfname)

    assert sdf_exists, "The sdf written was not found"

    sdf = SmallMolLib(sdfname)

    assert isinstance(sdf, SmallMolLib), (
        "The sdf written was not correctly loaded. Probably the previous"
        "writing went wrong"
    )


def _test_appendSmallMolLib():
    sdffile = os.path.join(curr_dir, "test_smallmol", "fda_drugs_light.sdf")
    lib = SmallMolLib(sdffile)
    lib2 = SmallMolLib(sdffile)

    lib.appendSmallLib(lib2)

    n_mol2_merged = lib.numMols

    assert n_mol2_merged == SDF_N_MOLS * 2, (
        "The number of molecules in the SmallMolLib is not as expected."
        "The two sdf were not correctly merged. "
    )


def _test_appendSmallMol():
    mol2file = os.path.join(curr_dir, "test_smallmol", "benzamidine.mol2")
    sdffile = os.path.join(curr_dir, "test_smallmol", "fda_drugs_light.sdf")

    lib = SmallMolLib(sdffile)
    sm = SmallMol(mol2file)
    lib.appendSmallMol(sm)

    n_mol2_append = lib.numMols

    assert n_mol2_append == SDF_N_MOLS + 1, (
        "The number of molecules in the SmallMolLib is not as expected."
        "The mol2 were not correctly append. "
    )


def _test_removeMols():
    sdffile = os.path.join(curr_dir, "test_smallmol", "fda_drugs_light.sdf")

    lib = SmallMolLib(sdffile)

    mols_ids = SDF_IDS_DELETE
    ref_mols_name = SDF_MOLNAME_DELETE

    mols_name = [s.ligname for s in lib.getMols(mols_ids)]

    assert (
        mols_name == ref_mols_name
    ), "The molecules at the given indexes do not match with the expected"
    lib.removeMols(mols_ids)

    mols_name_now = [s.ligname for s in lib.getMols(mols_ids)]

    assert mols_name_now != mols_name, "The molecules seem to not be deleted correctly"


def _test_convertToDataFrame():
    import pandas as pd

    sdffile = os.path.join(curr_dir, "test_smallmol", "fda_drugs_light.sdf")

    lib = SmallMolLib(sdffile)

    df = lib.toDataFrame()

    assert isinstance(
        df, pd.DataFrame
    ), "The SmallMolLib object was not correctly converted into pandas DataFrame"

    cols = df.columns.tolist()
    ref_cols = SDF_FIELDS

    assert (
        cols == ref_cols
    ), "The fields in the SmallMolLib object was not the expected one"

    ligname_99 = df.iloc[99][0]
    ref_ligname = SDF_LOC_0_99

    assert ligname_99 == ref_ligname, "The ligand name found is not the expected one"


def _test_writeSmiles(tmp_path):
    sdffile = os.path.join(curr_dir, "test_smallmol", "fda_drugs_light.sdf")

    lib = SmallMolLib(sdffile)
    tmpfile = os.path.join(tmp_path, "test.smi")
    lib.writeSmiles(tmpfile)

    with open(tmpfile, "r") as f:
        filelines = f.readlines()[1:]

    with open(os.path.join(curr_dir, "test_smallmol", "fda_drugs_light.smi"), "r") as f:
        reflines = f.readlines()[1:]

    assert filelines == reflines, "The smiles written were not the expected ones"


def _test_readSmiles():
    smifile = os.path.join(curr_dir, "test_smallmol", "fda_drugs_light.smi")
    lib = SmallMolLib(smifile)
    assert len(lib) == 100

    smifile = os.path.join(curr_dir, "test_smallmol", "fda_drugs_light.smi.gz")
    lib = SmallMolLib(smifile)
    assert len(lib) == 100
