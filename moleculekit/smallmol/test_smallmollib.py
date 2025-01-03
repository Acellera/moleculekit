# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import os
import unittest
from tempfile import NamedTemporaryFile
from moleculekit.home import home
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.smallmol.smallmollib import SmallMolLib
import rdkit
from pandas import core


SDF_N_MOLS = 100

SDF_IDS_DELETE = [1, 10, 15, 16]
SDF_MOLNAME_DELETE = ["ZINC02583363", "ZINC86860147", "ZINC04342657", "ZINC02023420"]
SDF_FIELDS = ["ligname", "_mol"]
SDF_LOC_0_99 = "ZINC02141008"


class _TestSmallMol(unittest.TestCase):
    def setUp(self):
        self.dataDir = home("test-smallmol")

    def test_loaSdffile(self):
        sdffile = os.path.join(self.dataDir, "fda_drugs_light.sdf")
        lib = SmallMolLib(sdffile)
        n_mols = lib.numMols
        self.assertEqual(
            n_mols,
            SDF_N_MOLS,
            f"Molecules not correctly loaded. Expected: {SDF_N_MOLS}; Now: {n_mols}",
        )

        sdffile = os.path.join(self.dataDir, "fda_drugs_light.sdf.gz")
        lib = SmallMolLib(sdffile)
        n_mols = lib.numMols
        self.assertEqual(
            n_mols,
            SDF_N_MOLS,
            f"Molecules not correctly loaded. Expected: {SDF_N_MOLS}; Now: {n_mols}",
        )

    def test_writeSdf(self):
        sdffile = os.path.join(self.dataDir, "fda_drugs_light.sdf")
        lib = SmallMolLib(sdffile)

        sdfname = NamedTemporaryFile().name + ".sdf"
        lib.writeSdf(sdfname)

        sdf_exists = os.path.isfile(sdfname)

        self.assertTrue(sdf_exists, msg="The sdf written was not found")

        sdf = SmallMolLib(sdfname)

        self.assertIsInstance(
            sdf,
            SmallMolLib,
            msg="The sdf written was not correctly loaded. Probably the previous"
            "writing went wrong",
        )

    def test_appendSmallMolLib(self):
        sdffile = os.path.join(self.dataDir, "fda_drugs_light.sdf")
        lib = SmallMolLib(sdffile)
        lib2 = SmallMolLib(sdffile)

        lib.appendSmallLib(lib2)

        n_mol2_merged = lib.numMols

        self.assertEqual(
            n_mol2_merged,
            SDF_N_MOLS * 2,
            msg="The number of molecules in the SmallMolLib is not as expected."
            "The two sdf were not correctly merged. ",
        )

    def test_appendSmallMol(self):
        mol2file = os.path.join(self.dataDir, "benzamidine.mol2")
        sdffile = os.path.join(self.dataDir, "fda_drugs_light.sdf")

        lib = SmallMolLib(sdffile)
        sm = SmallMol(mol2file)
        lib.appendSmallMol(sm)

        n_mol2_append = lib.numMols

        self.assertEqual(
            n_mol2_append,
            SDF_N_MOLS + 1,
            msg="The number of molecules in the SmallMolLib is not as expected."
            "The mol2 were not correctly append. ",
        )

    def test_removeMols(self):
        sdffile = os.path.join(self.dataDir, "fda_drugs_light.sdf")

        lib = SmallMolLib(sdffile)

        mols_ids = SDF_IDS_DELETE
        ref_mols_name = SDF_MOLNAME_DELETE

        mols_name = [s.ligname for s in lib.getMols(mols_ids)]

        self.assertListEqual(
            mols_name,
            ref_mols_name,
            msg="The molecules at the given indexes do not match with the" "expected",
        )
        lib.removeMols(mols_ids)

        mols_name_now = [s.ligname for s in lib.getMols(mols_ids)]

        self.assertFalse(
            mols_name_now == mols_name,
            msg="The molecules seem to not be deleted correctly",
        )

    def test_convertToDataFrame(self):
        sdffile = os.path.join(self.dataDir, "fda_drugs_light.sdf")

        lib = SmallMolLib(sdffile)

        df = lib.toDataFrame()

        self.assertIsInstance(
            df,
            core.frame.DataFrame,
            msg="The SmallMolLib object was not correctly converted into pandas"
            "DataFrame",
        )

        cols = df.columns.tolist()
        ref_cols = SDF_FIELDS

        self.assertEqual(
            cols,
            ref_cols,
            msg="The fields in the SmallMolLib object was not the expected one",
        )

        ligname_99 = df.iloc[99][0]
        ref_ligname = SDF_LOC_0_99

        self.assertEqual(
            ligname_99, ref_ligname, msg="The ligand name found is not the expected one"
        )

    def test_writeSmiles(self):
        sdffile = os.path.join(self.dataDir, "fda_drugs_light.sdf")

        lib = SmallMolLib(sdffile)
        tmpfile = NamedTemporaryFile().name + ".smi"
        lib.writeSmiles(tmpfile)

        with open(tmpfile, "r") as f:
            filelines = f.readlines()[1:]

        with open(os.path.join(self.dataDir, "fda_drugs_light.smi"), "r") as f:
            reflines = f.readlines()[1:]

        self.assertEqual(filelines, reflines)

    def test_readSmiles(self):
        smifile = os.path.join(self.dataDir, "fda_drugs_light.smi")
        lib = SmallMolLib(smifile)
        assert len(lib) == 100

        smifile = os.path.join(self.dataDir, "fda_drugs_light.smi.gz")
        lib = SmallMolLib(smifile)
        assert len(lib) == 100


if __name__ == "__main__":
    unittest.main(verbosity=2)
