import argparse
import os
import sys
import numpy as np
from glob import glob

sys.setrecursionlimit(25000)

class SystemBuilder:
    def __init__(self, args):
        for arg, value in args.items():
            if value == '':
                args[arg] = None

        self.vars = argparse.Namespace(**args)

    def run(self):
        from moleculekit.molecule import Molecule
        from tqdm import trange

        protein, ligand, others = self._loadStructures()
        refprot = protein.copy()

        protein = self.checkCaps(protein)

        prep_protein = protein

        post_prep_protein, noncanonicalpatches = self._amberpatchPostTranslationalMutation(prep_protein)

        # Assemble system
        system = Molecule()
        system.append(post_prep_protein)

        # complete the system by adding 'others' molecules
        for mol in others:
            system.append(mol)

        self._doBuild(0, system, ligand, noncanonicalpatches, refprot)
        self._prepareOutputs()


    def checkCaps(self, protein):
        from numpy import sum as _sum
        aceAnameCorrect = np.array(['C', 'O', 'CH3'])
        nmeAnameCorrect = np.array(['N', 'CH3'])
        sel_base = 'resname ACE NME'
        sel = '{} and hydrogen'.format(sel_base)

        if _sum(protein.atomselect(sel_base)) == 0:
            print('No caps found. Skiping the caps check')
            return protein

        m = protein.copy()
        if _sum(m.atomselect(sel)) != 0:
            print('WARNING: Hydrogen found in caps. These atoms will be removed')
            m.remove(sel)

        # check if atomname are not correct
        aceAname = np.unique(m.get('name', 'resname ACE'))
        nmeAname = np.unique(m.get('name', 'resname NME'))
        if not np.array_equal(np.sort(aceAname), np.sort(aceAnameCorrect)) or \
                not np.array_equal(np.sort(nmeAname), np.sort(nmeAnameCorrect)):
            print('Not valid atom name for caps found. Will be modified')
            m.bonds = m._getBonds()

        # Check ACE caps
        aceResIds = np.unique(m.resid[np.where(m.resname == 'ACE')])
        nextResIds = np.array([ m.resid[np.where(m.resid == a)[0][-1] + 1] for a in aceResIds  ])
        nextResIdsAsString = " ".join(nextResIds.astype(str).tolist())
        nextNidx = np.where(m.atomselect('resid {} and name  N'.format(nextResIdsAsString)))[0]


        for resace, nnext in list(zip(aceResIds, nextNidx)):

            aceIdxs = np.where(m.atomselect('resname ACE and resid {} and element C'.format(resace)))[0]
            aceIdx_O = np.where(m.atomselect('resname ACE and resid {} and element O'.format(resace)))[0]
            combs = [[a, nnext] for a in aceIdxs]

            bonds = m.bonds.tolist()
            for n, c in enumerate(combs):
                c = list(c)
                if c in bonds:
                    m.name[combs[n][0]] = aceAnameCorrect[0]
                else:
                    m.name[combs[n][0]] = aceAnameCorrect[2]
            m.name[aceIdx_O] = aceAnameCorrect[1]

        # Check NME caps

        nmeResIds = np.unique(m.resid[np.where(m.resname == 'NME')])
        prevResIds = np.array([ m.resid[np.where(m.resid == n)[0][0] - 1] for n in nmeResIds  ])
        prevResIdsAsString = " ".join(prevResIds.astype(str).tolist())
        prevCidx = np.where(m.atomselect('resid {} and name  C'.format(prevResIdsAsString)))[0]

        for resnme, cprev in list(zip(nmeResIds, prevCidx)):
            nmeIdxs = np.where(m.atomselect('resname NME and resid {} and element N'.format(resnme)))[0]
            nmeIdx_C = np.where(m.atomselect('resname NME and resid {} and element C'.format(resnme)))[0]
            m.name[nmeIdxs] = nmeAnameCorrect[0]
            m.name[nmeIdx_C] = nmeAnameCorrect[1]
        return m


    def _prepareOutputs(self):
        builds = glob(self.vars.outdir + '/build_*')
        self._prepareMoleculeStatistics(builds)

    def getBestAutosegment(self, protein):
        from moleculekit.tools.autosegment import autoSegment as a1
        from moleculekit.tools.autosegment import autoSegment2 as a2
        methods = [a1, a2]

        bestmol = None
        minNsegid = None
        for n, meth in enumerate(methods):
            m = meth(protein, sel='protein or resname ACE NME', basename='P')
            nsegids = len(np.unique(m.segid))
            if minNsegid is None:
                minNsegid = nsegids
                bestmol = m
            if nsegids < minNsegid:
                minNsegid = nsegids
                bestmol = m

        return bestmol


    def _loadStructures(self):
        from moleculekit.molecule import Molecule
        from numpy import sum as _sum
        from moleculekit.tools.autosegment import autoSegment
        class NotValidMembrane(Exception):
            pass
        class NotValidProtein(Exception):
            pass

        prot, lig, memb, others = None, None, None, []

        prot = Molecule(self.vars.protein)
        prot.filter('not water')
        if _sum(prot.atomselect('protein')) == 0:
            raise NotValidProtein('The protein argument expect at least a protein structure.')

        prot = self.getBestAutosegment(prot)

        #prot = autoSegment(prot, sel='protein or resname ACE NME', basename='P')
        if _sum(prot.atomselect('protein or resname ACE NME')) != prot.numAtoms:
            prot = autoSegment(prot, sel='not (protein or resname ACE NME)', basename='C')

        # ligand0
        if self.vars.ligand is not None:
            lig = Molecule(self.vars.ligand)
            if _sum(lig.atomselect('not segid \'\'')) != lig.numAtoms:
                print(
                    "Ligand file did not contain complete unique segments. Performing full automatic segmentation . . .")
                lig.set('segid', 'L')

        # others
        if hasattr(self.vars, 'others'):
            for i, o in enumerate(self.vars.others):
                oMol = Molecule(o)
                if _sum(oMol.atomselect('not segid \'\'')) != oMol.numAtoms:
                    print("Molecule file {} did not contain complete unique segments. Performing full automatic "
                          "segmentation . . .".format(o))
                    oMol.set('segid', 'X{}'.format(i), sel='all')

        return prot, lig, others


    def _prepareMoleculeStatistics(self, builds_folder):
        from moleculekit.molecule import Molecule
        import pandas as pd

        df = pd.DataFrame(
            columns=['Build #', 'FF', '# Atoms', '# Waters', '# Cations', '# Anions', '# Lipids', 'celldimension'])
        for n, bdir in enumerate(builds_folder):
            pdb = os.path.join(bdir, 'structure.pdb')
            m = Molecule(pdb)

            coords = m.get('coords', sel='water')
            dim = np.max(coords, axis=0) - np.min(coords, axis=0)

            numAtoms = m.numAtoms
            numWaters = len(m.get('name', 'waters and element O'))
            numLipids = len(m.get('name', 'lipids and element P'))
            numCations = len(np.where(m.element == 'NA')[0])
            numAnions = len(np.where(m.element == 'CL')[0])
            ff = 'amber'
            celldimension = " x ".join([str(d) for d in dim])

            df.loc[n] = [n + 1, ff, numAtoms, numWaters, numCations, numAnions, numLipids, celldimension]
        df.to_csv(os.path.join(self.vars.outdir, 'systemBuilder.csv'))


    def _detectStructFiles(self):
        # checks for '.pdb' and '.mol2' files. At least one needs to be present
        class StructFilesDetectionError(Exception):
            pass
        inputdir = self.vars.inputdir
        instructs = glob(os.path.join(inputdir, '*.pdb')) + glob(os.path.join(inputdir, '*.mol2'))
        if len(instructs) != 0:
            print('{} PDB/MOL2 file{} found: {}\n'.format(len(instructs),
                                                          ' was' if len(instructs) == 1 else 's were', instructs))
        else:
            raise StructFilesDetectionError('No PDB/MOL2 files found in directory \'{}\'. At least one file should be '
                                            'provided'.format(inputdir))
        self._checkStructures(instructs)


    def _checkStructures(self, instructs):
        from htmd.molecule.molecule import Molecule
        from numpy import sum as _sum
        class StructFilesLoadingError(Exception):
            pass

        protein = None
        ligand = None
        others = []

        for struct in instructs:
            tmp = Molecule(struct)
            if _sum(tmp.atomselect('protein')) == tmp.numAtoms:
                if protein is None:
                    protein = tmp
                    print('Protein detected: {}\n'.format(struct))
                    protein = struct
                    self.vars.protein = struct
                else:
                    all_protStruct = ",".join([struct, protein])
                    raise StructFilesLoadingError(
                        'More than one protein file detected ({}). Currently unsupported'.format(all_protStruct))
            elif _sum(tmp.atomselect('not protein')) == tmp.numAtoms:
                if self.vars.ligand is not None:
                    ligand = tmp
                    self.vars.ligand = struct
                else:
                    others.append(struct)
        if not protein:
            raise StructFilesLoadingError('No protein detected within the loaded files: {}. At least one protein '
                                          'structure is required'.format(instructs))
        if len(others) == 1:
            self.vars.ligand = others[0]


    def _amberpatchPostTranslationalMutation(self, protein):
        class PtmPatchingError(Exception):
            pass
        from moleculekit.molecule import _residueNameTable
        from htmd.home import home
        patches = None
        _derived_residues = ['CYM']
        _protein_residues = list(_residueNameTable.keys())
        _protein_residues.extend(_derived_residues)
        _protein_residues.extend(['ACE', 'NME'])  # Don't report on caps as non-canonical aas if they are passed

        list_aa = np.unique(protein.get('resname', sel='protein'))
        list_aa_noncanonical = np.unique([a for a in list_aa if a not in _protein_residues])
        print('Non canonical amino acid residues detected in '
              'protein: {}\n'.format(list_aa_noncanonical if len(list_aa_noncanonical) != 0 else 'None'))

        if len(list_aa_noncanonical) == 0:
            return protein, patches

        htmdamberdir = os.path.abspath(os.path.join(home(), 'builder', 'amberfiles'))
        list_ptm_aa_available = [p for fdir in os.listdir(htmdamberdir)
                                 for p in glob(os.path.join(htmdamberdir, fdir, '*.frcmod'))]
        detect_param = ["/".join(pa.split("/")[-2:]) for p in list_aa_noncanonical
                        for pa in list_ptm_aa_available if p in pa]

        if len(list_aa_noncanonical) != len(detect_param):
            notFound = list_aa_noncanonical
            if len(detect_param) != 0:
                notFound = [a for a in list_aa_noncanonical for pa in detect_param if a not in pa]

            raise PtmPatchingError(
                'Non-canonical residues detected ({}), but the parameters \'{}\' are not available in the standard '
                'Amber location'.format(",".join(list_aa_noncanonical), ','.join(notFound)))
        detect_topo = [glob(os.path.join("/".join(a_param.split("/")[:-1]), "*.in"))[0]
                       for fdir in os.listdir(htmdamberdir)
                       for d_param in detect_param
                       for a_param in glob(os.path.join(htmdamberdir, fdir, "*.frcmod"))
                       if d_param in a_param]
        setattr(self.vars, 'resfrcmod', detect_param)
        setattr(self.vars, 'resprepi', detect_topo)

        # check for atomName conversion
        for aa in list_aa_noncanonical:
            protein = self.convertAtomName(protein, aa)

        return protein, patches


    def _defineAmberParameters(self):
        from htmd.builder import amber
        ff = amber.defaultFf()
        topos = amber.defaultTopo()
        list_topos = ['ligprepi', 'resprepi']
        for topo in list_topos:
            if hasattr(self.vars, topo):
                newtopos = getattr(self.vars, topo)
                if newtopos is not None:
                    topos.extend(newtopos)
        params = amber.defaultParam()
        list_params = ['ligfrcmod', 'resfrcmod']
        for param in list_params:
            if hasattr(self.vars, param):
                newparams = getattr(self.vars, param)
                if newparams is not None and newparams != '':
                    params.extend(newparams)
        params.extend(["frcmod.ions234lm_126_tip3p"])

        return ff, topos, params


    def _getNewCoords(self, ligand, system, refsystem):
        from moleculekit.align import _pp_measure_fit

        r_sel = refsystem.atomselect('name CA')
        s_sel = system.atomselect('name CA')

        P = refsystem.coords[r_sel, :, 0]
        Q = system.coords[s_sel, :, 0]

        center_P = np.zeros(3, dtype=P.dtype)
        center_Q = np.zeros(3, dtype=Q.dtype)

        for i in range(3):
            center_P[i] = np.mean(P[:, i])
            center_Q[i] = np.mean(Q[:, i])

        (rot, tmp) = _pp_measure_fit(P - center_P, Q - center_Q)

        all1 = ligand.coords[:, :, 0]
        all1 = all1 - center_P
        all1 = np.dot(all1, rot.T)
        all1 = all1 + center_Q

        return all1


    def _doBuild(self, index, system, ligand, patches, refsystem):
        class LeapError(Exception):
            pass

        from htmd.builder.solvate import solvate
        from htmd.builder import charmm, amber
        from htmd.molecule.util import maxDistance, uniformRandomRotation

        outer_shell = 5
        builddir = self.vars.outdir
        saltconc = self.vars.saltconc
        bmol = None

        print('Build #{} starting...\n{}'.format(index + 1, ''.join('=' for _ in range(19 + len(str(index))))))

        mol = system.copy()
        # sysradius = maxDistance(mol)
        ligdistance = 0
        ligradius = 0
        # if ligand is not None

        if ligand is not None:
            ligand.coords[:, :, 0] = self._getNewCoords(ligand, mol, refsystem)
            mol.append(ligand)

        mol.center()
        sysradius = maxDistance(mol)

        # print(sysradius, ligradius, ligdistance, outer_shell)
        halfside = sysradius + ligradius + ligdistance + outer_shell

        if ligand is None:
            minmax = self._computeBox(mol, ligand)
        else:
            minmax = self._computeBox(mol, ligand)
        smol = solvate(mol, prefix="W", minmax=minmax)
        bdir = os.path.join(builddir, 'build_{}'.format(index + 1))

        ff, topo, param = self._defineAmberParameters()
        param = [p for p in param if p != '']
        topo = [t for t in topo if t != '']
        ff = [f for f in ff if f != '']

        try:
            bmol = amber.build(smol, ff=ff, topo=topo, param=param,
                               outdir=bdir, saltconc=saltconc)
        except:
            raise LeapError('The building failed due to a LEaP (AmberTools package) error. Most common explanations are: i) wrong atom name;'
                            ' ii) missing atom type; iii) missing parameters. Check the complete log. ')
        return bmol


    def _computeBox(self, system, ligand):
        from htmd.molecule.util import maxDistance
        class InvalidMinDistance(Exception):
            pass

        pad = 10
        pad_withLigand = 3

        protcoords = system.coords#get('coords', 'protein')
        mc = np.min(protcoords)
        Mc = np.max(protcoords)
        center = np.mean(protcoords)
        dm = np.linalg.norm((center-mc))
        dM = np.linalg.norm((Mc-center))
        D = dm if dm > dM else dM

        if self.vars.mindist != 0.0:

            tmpD = D
            D = self.vars.mindist

            if D < tmpD:
                raise InvalidMinDistance("The distance chosen ({}). with the -mindist argument is smaller than the "
                                         "size of the protein ({:.2f}). You need to increase the value or use the "
                                         "automatic procedure".format(D, tmpD))
            minmax = [[-D]*3, [D]*3]

        elif ligand is None:
            D = D + pad
            minmax = [[-D]*3, [D]*3]
        else:
            D = D + pad
            minmax = [[-D] * 3, [D] * 3]

        return minmax


    def _restoreResName(self, index, patches):
        from htmd.molecule.molecule import Molecule
        bdir = os.path.join(self.vars.outdir, "build_{}".format(index + 1))
        inv_dict_patchs_charmm = {v: k for k, v in self._dict_patchs_charmm.items()}
        for patch in patches:
            resname = patch.split()[1]
            print('Restoring patched residue name {} to its '
                  'original {}\n'.format(resname, inv_dict_patchs_charmm[resname]))
            segid = patch.split()[-1].split(":")[0]
            resid = patch.split()[-1].split(":")[1]
            ref_mol = Molecule(os.path.join(bdir, "structure.psf"))
            ref_mol.read(os.path.join(bdir, "structure.pdb"))
            ref_mol.set("resname", inv_dict_patchs_charmm[resname], "resid {} and segid {}".format(resid, segid))
            ref_mol.write(os.path.join(bdir, "structure.psf"))
            ref_mol.write(os.path.join(bdir, "structure.pdb"))


    def convertAtomName(self, prot, resName):
        import numpy as np
        atomnameList = np.unique(prot.get('name', 'resname {}'.format(resName)))
        atom_names = {}
        if resName == 'SEP':
            if 'HB1' in atomnameList:
                atom_names = {'HB1': 'HB2', 'HB2': 'HB3'}
        elif resName == 'PTR':
            atom_names = {'HE1': 'HE11', 'HE2': 'HE21', 'HD1': 'HD11', 'HD2': 'HD21'}
        elif resName == 'CSO':
            atom_names = {'HD': 'HD1'}

        else:
            return prot

        for key, value in atom_names.items():
            prot.set('name', value, 'resname {} and name {}'.format(resName, key))
        return prot
