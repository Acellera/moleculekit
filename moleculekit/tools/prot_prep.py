import sys
import numpy as np
import pickle
import re
import argparse
from moleculekit.molecule import Molecule
from moleculekit.tools.preparation import proteinPrepare

sys.setrecursionlimit(25000)

class ProteinPreparator:
    def __init__(self, args):

        for arg, value in args.items():
            if value == '':
                args[arg] = None

        self.vars = argparse.Namespace(**args)
        print(self.vars)
        self.vars.pH = float(self.vars.pH)
        self.vars.chain = "all" if not self.vars.chain else self.vars.chain


    def _loadPDB(self, pdb):
        m = Molecule(pdb)
        if self.vars.remove_water:
            if not self.vars.include_heteroatoms:
                prot_sel = "protein"
                if self.vars.chain != "all":
                    prot_sel = "protein and chain " + self.vars.chain
            else:
                prot_sel = "protein or not water"
                if self.vars.chain != "all":
                    prot_sel = "(protein or not water) and chain " + self.vars.chain
        else:
            if not self.vars.include_heteroatoms:
                prot_sel = "protein or water"
                if self.vars.chain != "all":
                    prot_sel = "(protein or water) and chain " + self.vars.chain
            else:
                prot_sel = "protein or water or not water"
                if self.vars.chain != "all":
                    prot_sel = "(protein or water or not water) and chain " + self.vars.chain
        m.filter(prot_sel)
        return m

    def _runFirstProteinPrepare(self):
        protprep, prepdetails = proteinPrepare(self.mol, pH=self.vars.pH, returnDetails=True)

        # return details
        prepdetails.data.to_csv( "details.csv")

        heteroatoms = []
        if self.vars.include_heteroatoms:
            heteroatoms = list(np.unique(protprep.get("resname", "not protein and not water")))

        # produce svg diagram
        svg_plot = prepdetails._get_pka_plot(pH=self.vars.pH, font_size=8)
        with open('protonation_diagram.svg', 'w') as f:
            f.write(svg_plot)

        # generate ans object
        ans = {
            'svg_plot': svg_plot,
            'csv': 'details.csv',
            'protein': 'output.pdb',
            'prepData': prepdetails,
            'pH': self.vars.pH,
            'heteroatoms': heteroatoms
        }
        pickle.dump(ans, open("job_content.pickle", "wb"))

        # write result
        protprep.write("output.pdb")


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
        if len(aceAname) > 0:
            aceResIds = np.unique(m.resid[np.where(m.resname == 'ACE')])
            nextResIds = np.array([m.resid[np.where(m.resid == a)[0][-1] + 1] for a in aceResIds])
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
        if len(nmeAname) > 0:
            nmeResIds = np.unique(m.resid[np.where(m.resname == 'NME')])
            prevResIds = np.array([m.resid[np.where(m.resid == n)[0][0] - 1] for n in nmeResIds])
            prevResIdsAsString = " ".join(prevResIds.astype(str).tolist())
            prevCidx = np.where(m.atomselect('resid {} and name  C'.format(prevResIdsAsString)))[0]

            for resnme, cprev in list(zip(nmeResIds, prevCidx)):
                nmeIdxs = np.where(m.atomselect('resname NME and resid {} and element N'.format(resnme)))[0]
                nmeIdx_C = np.where(m.atomselect('resname NME and resid {} and element C'.format(resnme)))[0]
                m.name[nmeIdxs] = nmeAnameCorrect[0]
                m.name[nmeIdx_C] = nmeAnameCorrect[1]
        return m


    def run(self):

        self.mol = self._loadPDB(self.vars.pdb)

        # check caps
        self.mol = self.checkCaps(self.mol)

        self._runFirstProteinPrepare()

