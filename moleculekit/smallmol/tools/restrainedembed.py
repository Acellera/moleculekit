from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField


def restrainedEmbed(mol, atomPos={}, restraintStrengths=(100, 25), forceField=UFFGetMoleculeForceField, 
                    numMinim=40, **kwargs):
    """ Restrained embedding of a molecule
    
    Generates an embedding of a molecule where part of the molecule is restrained to specific coordinates.
    It works in an iterative manner minimizing with different restraint strength in each iteration.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule to embed. This function will modify the molecule in place.
    atomPos : dict
        A dictionary with key the index of the atom as an integer and value a list of the three xyz coordinates
        where the atom should be restrained to. For the atoms for which no keys are in the dictionary, they will
        be left unrestrained.
    restraintStrengths : list
        A list of restraints strengths which will be applied in order.
    forceField : rdkit ForceField
        The rdkit forcefield to use for the minimization.
    numMinim : int
        The number of minimizations to perform at each iteration.
    """

    for restr in restraintStrengths:
        ff = forceField(mol, confId=0)
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            if i in atomPos:
                p.x, p.y, p.z = float(atomPos[i][0]), float(atomPos[i][1]), float(atomPos[i][2])
                pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
                ff.AddDistanceConstraint(pIdx, i, 0, 0, float(restr))

        ff.Initialize()
        n = numMinim
        more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        while more and n:
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            n -= 1
