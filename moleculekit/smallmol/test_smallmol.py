import os
import unittest
from tempfile import NamedTemporaryFile
from glob import glob
from moleculekit.home import home
from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
import rdkit
from rdkit.Chem import MolFromSmiles

BENZAMIDINE_N_ATOMS = 18
BENZAMIDINE_N_HEAVYATOMS = 9
BENZAMIDINE_BONDTYPES = ['ar', 'ar', '1', 'ar', '1', 'ar', '1', 'ar', '1', 'ar', '1', '1', '2', '1', '1', '1', '1', '1']
BENZAMIDINE_BOND_ATOMS = [[0, 1], [0, 5], [0, 6], [1, 2], [1, 7], [2, 3], [2, 8], [3, 4], [3, 9], [4, 5], [4, 10],
                          [5, 11], [6, 12], [6, 13], [12, 16], [12, 17], [13, 14], [13, 15]]

LIGAND_N_ATOMS = 64
LIGAND_N_HEAVYATOMS = 35

SMILE_SMI = 'c1ccccc1O'
SMILE_N_ATOMS = 13

SDF_N_MOLS = 100

PHENOL_ELEMENT_IDX_1 = 'C'
PHENOL_ELEMENT_NEIGHBORS_OX = [5, 12]
PHENOL_BTYPES_OX = [1, 1]

CHIRAL_SMI = 'C[C@H](Cl)F'
CHIRAL_DETAILS = [('C1', 'S')]

FOUNDBOND_SMI = 'C=CN'

class _TestSmallMol(unittest.TestCase):

    def setUp(self):
        self.dataDir= home('test-smallmol')
        self.benzamidine_mol2 = os.path.join(self.dataDir, 'benzamidine.mol2')

    def test_loadMol2file(self):
        sm = SmallMol(self.benzamidine_mol2)
        n_atoms = sm.numAtoms
        self.assertEqual(n_atoms, BENZAMIDINE_N_ATOMS, 'Atoms not correctly loaded. '
                                                        'Expected: {}; Now: {}'.format(BENZAMIDINE_N_ATOMS, n_atoms))

    def test_loadPdbfile(self):
        pdbfile = os.path.join(self.dataDir, 'ligand.pdb')
        sm = SmallMol(pdbfile)
        n_atoms = sm.numAtoms
        self.assertEqual(n_atoms, LIGAND_N_ATOMS, 'Atoms not correctly loaded. '
                                                        'Expected: {}; Now: {}'.format(LIGAND_N_ATOMS, n_atoms))

    def test_loadSmile(self):
        smi = SMILE_SMI
        sm = SmallMol(smi)
        n_atoms = sm.numAtoms
        self.assertEqual(n_atoms, SMILE_N_ATOMS, 'Atoms not correctly loaded. '
                                                  'Expected: {}; Now: {}'.format(SMILE_N_ATOMS, n_atoms))


    def test_getAtoms(self):
        smi = SMILE_SMI
        sm = SmallMol(smi)
        element_idx_1 = sm.get('element', 'idx 1')[0]
        neighbors_element_O = sm.get('neighbors', 'element O')[0]
        btypes_element_O = sm.get('bondtype', 'element O', convertType=False)[0]

        self.assertEqual(element_idx_1, PHENOL_ELEMENT_IDX_1, 'Element of the first atom does not correspond'
                                                              'Expect: {}; Now: {}'.format(element_idx_1, PHENOL_ELEMENT_IDX_1))
        self.assertListEqual(neighbors_element_O, PHENOL_ELEMENT_NEIGHBORS_OX,  'Neighbors atoms of the oxygen atom do not correspond'
                           'Expected: {}; Now: {}'.format(PHENOL_ELEMENT_NEIGHBORS_OX, neighbors_element_O))

        self.assertListEqual(btypes_element_O, PHENOL_BTYPES_OX, 'Bondtypes of the oxygen atom do not correspond:'
                                                                 'Expeected: {}; Now: {}'.format(btypes_element_O, PHENOL_BTYPES_OX))

    def test_isChiral(self):
        smi = CHIRAL_SMI
        sm = SmallMol(smi)
        ischiral, details = sm.isChiral(returnDetails=True)
        self.assertListEqual(details, CHIRAL_DETAILS, 'chiral atom does not match.'
                                                      'Expected: {}; Now: {}'.format(CHIRAL_DETAILS, details))

    def test_foundBond(self):
        smi = FOUNDBOND_SMI
        sm = SmallMol(smi)
        isbond_0_N = sm.foundBondBetween('idx 0', 'element N')
        isbond_0_1_single = sm.foundBondBetween('idx 0', 'idx 1', bondtype=1)
        isbond_0_1_double, _ = sm.foundBondBetween('idx 0', 'idx 1', bondtype=2)


        self.assertFalse(isbond_0_N, 'Bond between atom 0 and any nitrogens should not be present')
        self.assertFalse(isbond_0_1_single, 'Bond between atom 0 1 should not be single')
        self.assertTrue(isbond_0_1_double, 'Bond between atom 0 1 should  be double')


    def test_generateConformers(self):
        sm = SmallMol(self.benzamidine_mol2)
        current_conformer = sm.numFrames
        sm.generateConformers(num_confs=10, append=False)
        n_conformers = sm.numFrames

        self.assertGreater(n_conformers, current_conformer, 'The generation of conforemr should provide at least the '
                                                            'same amount of conformer')

    def test_writeGenerateAndWriteConformers(self):
        sm = SmallMol(self.benzamidine_mol2)
        sm.generateConformers(num_confs=10, append=False)
        tmpfname = os.path.join(NamedTemporaryFile().name, 'benzamidine.sdf')
        tmpdir = os.path.dirname(tmpfname)
        sm.write(tmpfname, merge=False)
        direxists = os.path.isdir(tmpdir)
        n_files = len(glob(os.path.join(tmpdir, '*.sdf')))
        self.assertTrue(direxists, 'The directory where to store the conformations where not created')
        self.assertGreater(n_files, 1, 'None conformations were written. At least one should be present')


    def test_removeGenerateConformer(self):
        molsmile = SMILE_SMI
        sm = SmallMol(molsmile)
        sm.generateConformers(num_confs=10, append=False)
        n_confs = sm.numFrames
        sm.dropFrames([0])
        n_confs_del = sm.numFrames
        sm.dropFrames()
        n_confs_zero = sm.numFrames

        self.assertEqual(n_confs_del, n_confs - 1, "The number of conformations after the deletion was not reduced of "
                                                   "exactly one unit")
        self.assertEqual(n_confs_zero, 0, "The number of conformations after the deletion was not reduced to 0")


    def test_convertToMolecule(self):
        from moleculekit.molecule import mol_equal 

        sm = SmallMol(self.benzamidine_mol2)
        mol = sm.toMolecule(formalcharges=False)

        assert mol_equal(sm, mol, exceptFields=['serial', 'name'], _logger=False)

    def test_convertFromMolecule(self):
        from moleculekit.molecule import mol_equal 

        mol = Molecule(self.benzamidine_mol2)
        sm = SmallMol(mol)

        assert mol_equal(sm, mol, exceptFields=['serial', 'name'], _logger=False)

    def test_getBonds(self):
        sm = SmallMol(self.benzamidine_mol2)

        self.assertListEqual(sm._bonds.tolist(), BENZAMIDINE_BOND_ATOMS, msg="The atoms in bonds are not the same of the reference")

        self.assertListEqual(sm._bondtype.tolist(), BENZAMIDINE_BONDTYPES, msg="The bonds type are not the same of the reference")

    def test_depict(self):
        import IPython
        refimg = os.path.join(self.dataDir, 'benzamidine.svg')

        sm = SmallMol(self.benzamidine_mol2)

        img_name = NamedTemporaryFile().name + '.svg'
        sm.depict(sketch=True, filename=img_name, atomlabels="%a%i%c")
        png_name = NamedTemporaryFile().name + '.png'
        sm.depict(sketch=True, filename=png_name, atomlabels="%a%i%c")
        noext_name = NamedTemporaryFile().name
        sm.depict(sketch=True, filename=noext_name, atomlabels="%a%i%c")
        sm.depict(sketch=False, optimize=True)
        _img = sm.depict(sketch=True, ipython=True)

        refimg_size = os.path.getsize(refimg)
        sm_img_size = os.path.getsize(img_name)

        self.assertIsInstance(_img, IPython.core.display.SVG, msg="The object is not an IPython image as expected")
        self.assertEqual(sm_img_size, refimg_size, msg="The svg image does not have the same size of the reference")

    def test_repr(self):
        sm = SmallMol(self.benzamidine_mol2)
        _ = str(sm)

    def test_toSMILES(self):
        sm = SmallMol(self.benzamidine_mol2)
        assert sm.toSMILES() == 'NC(=[NH2+])C1=CC=CC=C1', 'Failed with SMILES: {}'.format(sm.toSMILES())

    def test_toSMARTS(self):
        sm = SmallMol(self.benzamidine_mol2)
        assert sm.toSMARTS() == '[#6]1(:[#6H]:[#6H]:[#6H]:[#6H]:[#6H]:1)-[#6](=[#7H2+])-[#7H2]', 'Failed with SMARTS: {}'.format(sm.toSMARTS())

    def test_align(self):
        from moleculekit.util import rotationMatrix
        import numpy as np

        sm = SmallMol(self.benzamidine_mol2)
        mol = sm.toMolecule()
        mol.rotateBy(rotationMatrix([1, 0, 0], 3.14))
        sm.align(mol)
        
        assert (np.abs(sm._coords) - np.abs(mol.coords)).max()  # I need to do the abs of the coords since it's a symmetrical molecule

    # def test_getRCSBLigandByLigname(self):
    #     from moleculekit.smallmol.util import getRCSBLigandByLigname
    #     sm = getRCSBLigandByLigname('BEN')



if __name__ == '__main__':
    unittest.main(verbosity=2)