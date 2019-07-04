try:
    import rdkit
except ImportError as e:
    raise ImportError('{}. You are probably missing the rdkit package. Please install it to support rdkit integration '
                      'features'.format(e))
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Geometry import Point3D
import os
import numpy as np


def _convertMoleculeToRDKitMol(mol):
    """
    Convert Molecule to RDKit Mol
    Note: sometimes RDKit fails just writing a warning message to stderr, rather than raising an exception.
    Parameters
    ----------
    mol: Molecule
        Molecule to convert
    Return
    ------
    results: RDKit Mol
        RDKit Mol object
    Examples
    --------
    >>> from htmd.home import home
    >>> from moleculekit.molecule import Molecule
    >>> molFile = os.path.join(home('test-charge'), 'H2O.mol2')
    >>> mol = Molecule(molFile)
    >>> type(_convertMoleculeToRDKitMol(mol))
    <class 'rdkit.Chem.rdchem.Mol'>
    """

    from rdkit.Chem import MolFromMol2File
    import tempfile
    import sys

    rdkit_mol = None
    with tempfile.TemporaryFile(mode='w+') as stderr:

        # Redirect stderr to a file
        temp_fileno = os.dup(sys.stderr.fileno())
        os.dup2(stderr.fileno(), sys.stderr.fileno()) # Change process stderr
        sys.stderr = stderr # Change Python stderr

        # Convert Molecule to RDKit Mol
        with tempfile.TemporaryDirectory() as tmpDir:
            filename = os.path.join(tmpDir, 'mol.mol2')
            mol.write(filename)
            rdkit_mol = MolFromMol2File(filename, removeHs=False)

        # Reset stderr
        os.dup2(temp_fileno, sys.__stderr__.fileno())
        os.close(temp_fileno)
        sys.stderr = sys.__stderr__

        # Read RDKit warnings
        stderr.flush()
        stderr.seek(0)
        message = stderr.read()

    if rdkit_mol is None:
        raise RuntimeError(f'RDKit error\n{message}\n')

    if rdkit_mol.GetNumAtoms() != mol.numAtoms:
        raise RuntimeError(f'RDKit error: incorrect number of atoms.\n{message}\n')

    return rdkit_mol


def StandardPDBResidueChirality(rdmol):
    for a in rdmol.GetAtoms():
        if a.GetChiralTag() != rdkit.Chem.rdchem.CHI_UNSPECIFIED:
            info = a.GetMonomerInfo()
            if info and (info.GetMonomerType() == rdkit.Chem.rdchem.AtomMonomerType.PDBRESIDUE) \
                and not info.GetIsHeteroAtom() and not StandardPDBChiralAtom(info.GetResidueName(), info.GetName()):
                a.SetChiralTag(rdkit.Chem.rdchem.CHI_UNSPECIFIED)
                # TODO: Missing CIPCode unsetting


def StandardPDBChiralAtom(resname, atomname):
    if resname == 'GLY':
        return False
    if resname == 'ILE' or resname == 'THR':
        return atomname == 'CA' or atomname == 'CB'
    if resname in ('ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'TRP', 'TYR', 'VAL'):
        return atomname == 'CA'
    return False


def toRDKITmol(mol, protidx, sanitize=True, removeHs=False):
    # Taken from rdkit/Code/GraphMol/FileParsers/PDBParser.cpp
    conformer = Chem.Conformer(len(protidx))
    conformer.Set3D(True)
    conformer.SetId(0)
    rdmol = Chem.RWMol()
    atomlist = []
    for ii, i in enumerate(protidx):
        a = Chem.Atom(mol.element[i])
        a.SetFormalCharge(int(mol.charge[i]))
        info = Chem.AtomPDBResidueInfo(atomName=mol.name[i],
                                    serialNumber=int(mol.serial[i]),
                                    altLoc=mol.altloc[i],
                                    residueName=mol.resname[i],
                                    residueNumber=int(mol.resid[i]),
                                    chainId=mol.chain[i],
                                    insertionCode=mol.insertion[i],
                                    occupancy=float(mol.occupancy[i]),
                                    tempFactor=float(mol.beta[i]),
                                    isHeteroAtom=mol.record[i]=='HETATM')
        a.SetMonomerInfo(info)

        rdmol.AddAtom(a)
        atomlist.append(a)
        coor = [float(c) for c in mol.coords[i, :, mol.frame]]
        conformer.SetAtomPosition(ii, Point3D(coor[0], coor[1], coor[2]))  # Correct the atom idx
    rdmol.AddConformer(conformer)

    # Here I diverge from the C++ parser because you cannot instantiate Chem.Bond objects in python
    # I also don't take into account double/triple bonds etc since I don't think we actually store them in Molecule
    for b in mol._getBonds():
        if b[0] in protidx and b[1] in protidx:
            bond = rdmol.GetBondBetweenAtoms(int(b[0]), int(b[1]))
            if bond is None:
                rdmol.AddBond(int(np.where(protidx == b[0])[0]), int(np.where(protidx == b[1])[0]), Chem.BondType.SINGLE)

    # Proximitybonds I already did by using _getBonds which calls _guessBonds
    # TODO: Set PDB double bonds

    # Calculate explicit valence of atoms
    for a in atomlist:
        pass

    if sanitize:
        if removeHs:
            Chem.RemoveHs(rdmol)
        else:
            Chem.SanitizeMol(rdmol)
    else:
        rdmol.UpdatePropertyCache()

    # Set tetrahedral chirality from 3D co-ordinates
    Chem.AssignAtomChiralTagsFromStructure(rdmol)
    StandardPDBResidueChirality(rdmol)

    return rdmol


if __name__ == '__main__':
    from moleculekit.molecule import Molecule

    mol = Molecule('3PTB')
    mol.filter('resname BEN')
    res = _convertMoleculeToRDKitMol(mol)
    assert res is not None

    # # We don't use this yet. It's experimental. So no point in wasting time testing it.
    # protidx = mol.atomselect('protein', indexes=True)

    # rdmol = toRDKITmol(mol, protidx)

    # fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    # factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # feats = factory.GetFeaturesForMol(rdmol)
