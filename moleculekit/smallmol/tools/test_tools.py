import os
import unittest
from tempfile import NamedTemporaryFile
from glob import glob
from moleculekit.home import home
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.smallmol.smallmollib import SmallMolLib
from pandas import core
from moleculekit.smallmol.tools.clustering import cluster, getMaximumCommonSubstructure
import numpy as np
import rdkit


CMS_N_ATOMS = 2
CMS_AIDX_MOL0 = [0,1]

# clusters
MACCS_N_CLUSTER = 75
MACCS_POPULATION_CLUSTER = [2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 1, 1, 1, 1, 2, 3, 
                            1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 
                            1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]

PATHFINGERPRINTS_N_CLUSTER = 82
PATHFINGERPRINTS_POPULATION_CLUSTER = [2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 1, 2, 2, 1, 1, 
                                       1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 
                                       1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]

ATOMSFINGERPRINTS_N_CLUSTER = 79
ATOMSFINGERPRINTS_POPULATION_CLUSTER = [2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 1, 2, 2, 1, 1, 1, 
                                        1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 
                                        1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]

TORSIONSFINGERPRINTS_N_CLUSTER = 79
TORSIONSFINGERPRINTS_POPULATION_CLUSTER = [2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 1, 2, 2, 1, 
                                           1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 1, 1, 1, 2, 1, 1, 2, 
                                           1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]


CIRCULARFINGERPRINTS_N_CLUSTER = 80
CIRCULARFINGERPRINTS_POPULATION_CLUSTER = [2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 
                                           1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 1, 1, 1, 2, 1, 1, 2, 
                                           1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]

MCS_N_CLUSTER = 78
MCS_POPULATION_CLUSTER = [2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 
                          1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 
                          1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]

class _TestSmallMol(unittest.TestCase):

    def setUp(self):
        self.dataDir= home('test-smallmol')

    def test_getCommonStructure(self):
        sdffile = os.path.join(self.dataDir, 'fda_drugs_light.sdf')
        lib = SmallMolLib(sdffile)
        rd_mols = lib._mols

        cms, cms_idxs, other_idxs = getMaximumCommonSubstructure(rd_mols, returnAtomIdxs=True)

        self.assertIsInstance(cms, rdkit.Chem.rdchem.Mol, msg="The object is not a rdkit Molecule object")

        ref_cms_natoms = CMS_N_ATOMS
        ref_atomidx_mol0 = CMS_AIDX_MOL0
        cms_atomidx_mol0 = cms_idxs[0]
        cms_natoms = len(cms_atomidx_mol0)

        self.assertEqual(cms_atomidx_mol0, ref_atomidx_mol0, msg="The atomidx of the CMS for the mol0 are not the"
                                                                 " expected ones")
        self.assertEqual(cms_natoms, ref_cms_natoms,  msg="The number of atom in the CMS for the mol0 are not the"
                                                                 " expected ones")

    _methods = ['maccs', 'pathFingerprints', 'atomsFingerprints', 'torsionsFingerprints',
                'circularFingerprints', 'shape', 'mcs']
    def test_cluster_maccs(self):
        sdffile = os.path.join(self.dataDir, 'fda_drugs_light.sdf')
        lib = SmallMolLib(sdffile)
        mols = lib._mols

        cl, det = cluster(mols, 'maccs', returnDetails=True)

        ref_ncluster = MACCS_N_CLUSTER
        ref_populations = MACCS_POPULATION_CLUSTER

        ncluster = det['numClusters']
        population = det['populations'].tolist()

        self.assertEqual(ncluster, ref_ncluster, msg="The number of cluster identified are not as expected")
        self.assertEqual(population, ref_populations, msg="The population fo the cluster are not the expected one")


    def test_cluster_pathFingerprints(self):
        sdffile = os.path.join(self.dataDir, 'fda_drugs_light.sdf')
        lib = SmallMolLib(sdffile)
        mols = lib._mols

        cl, det = cluster(mols, 'pathFingerprints', returnDetails=True)

        ref_ncluster = PATHFINGERPRINTS_N_CLUSTER
        ref_populations = PATHFINGERPRINTS_POPULATION_CLUSTER

        ncluster = det['numClusters']
        population = det['populations'].tolist()

        self.assertEqual(ncluster, ref_ncluster, msg="The number of cluster identified are not as expected")
        self.assertEqual(population, ref_populations, msg="The population fo the cluster are not the expected one")


    def test_cluster_atomsFingerprints(self):
        sdffile = os.path.join(self.dataDir, 'fda_drugs_light.sdf')
        lib = SmallMolLib(sdffile)
        mols = lib._mols

        cl, det = cluster(mols, 'atomsFingerprints', returnDetails=True)

        ref_ncluster = ATOMSFINGERPRINTS_N_CLUSTER
        ref_populations = ATOMSFINGERPRINTS_POPULATION_CLUSTER

        ncluster = det['numClusters']
        population = det['populations'].tolist()

        self.assertEqual(ncluster, ref_ncluster, msg="The number of cluster identified are not as expected")
        self.assertEqual(population, ref_populations, msg="The population fo the cluster are not the expected one")

    def test_cluster_torsionsFingerprints(self):
        sdffile = os.path.join(self.dataDir, 'fda_drugs_light.sdf')
        lib = SmallMolLib(sdffile)
        mols = lib._mols

        cl, det = cluster(mols, 'torsionsFingerprints', returnDetails=True)

        ref_ncluster = TORSIONSFINGERPRINTS_N_CLUSTER
        ref_populations = TORSIONSFINGERPRINTS_POPULATION_CLUSTER

        ncluster = det['numClusters']
        population = det['populations'].tolist()
        
        self.assertEqual(ncluster, ref_ncluster, msg="The number of cluster identified are not as expected")
        self.assertEqual(population, ref_populations, msg="The population fo the cluster are not the expected one")

    def test_cluster_circularFingerprints(self):
        sdffile = os.path.join(self.dataDir, 'fda_drugs_light.sdf')
        lib = SmallMolLib(sdffile)
        mols = lib._mols

        cl, det = cluster(mols, 'circularFingerprints', returnDetails=True)

        ref_ncluster = CIRCULARFINGERPRINTS_N_CLUSTER
        ref_populations = CIRCULARFINGERPRINTS_POPULATION_CLUSTER

        ncluster = det['numClusters']
        population = det['populations'].tolist()

        self.assertEqual(ncluster, ref_ncluster, msg="The number of cluster identified are not as expected")
        self.assertEqual(population, ref_populations, msg="The population fo the cluster are not the expected one")

    def test_cluster_shape(self):
        sdffile = os.path.join(self.dataDir, 'fda_drugs_light.sdf')
        lib = SmallMolLib(sdffile)
        mols = lib._mols

        cl, det = cluster(mols, 'shape', returnDetails=True)

        n_clusters = det['numClusters']

        self.assertIsInstance(n_clusters, np.int64, msg="None valid number of clusters")

    def test_cluster_mcs(self):
        sdffile = os.path.join(self.dataDir, 'fda_drugs_light.sdf')
        lib = SmallMolLib(sdffile)
        mols = lib._mols

        cl, det = cluster(mols, 'mcs', returnDetails=True)

        ref_ncluster = MCS_N_CLUSTER
        ref_populations = MCS_POPULATION_CLUSTER

        ncluster = det['numClusters']
        population = det['populations'].tolist()

        self.assertEqual(ncluster, ref_ncluster, msg="The number of cluster identified are not as expected")
        self.assertEqual(population, ref_populations, msg="The population fo the cluster are not the expected one")

    def test_restrained_embedding(self):
        from moleculekit.smallmol.tools.restrainedembed import restrainedEmbed
        import numpy as np

        atomPos = np.load(os.path.join(self.dataDir, 'restrain_positions.npy'), allow_pickle=True).item()
        refcoords = np.load(os.path.join(self.dataDir, 'restrain_final_coords.npy'), allow_pickle=True)
        sml = SmallMolLib(os.path.join(self.dataDir, 'restrain_molecule.sdf'))
        sm = sml[0]
        
        restrainedEmbed(sm._mol, atomPos)

        assert np.allclose(refcoords, sm._coords)

    # TODO: Add here tests for tautomers


if __name__ == '__main__':
    unittest.main(verbosity=2)