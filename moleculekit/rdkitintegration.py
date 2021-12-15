# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
try:
    import rdkit
except ImportError as e:
    raise ImportError(
        f"{e}. You are probably missing the rdkit package. Please install it to support rdkit integration features"
    )
from rdkit import Chem
from rdkit.Geometry import Point3D
import numpy as np


def StandardPDBResidueChirality(rdmol):
    for a in rdmol.GetAtoms():
        if a.GetChiralTag() != rdkit.Chem.rdchem.CHI_UNSPECIFIED:
            info = a.GetMonomerInfo()
            if (
                info
                and (
                    info.GetMonomerType()
                    == rdkit.Chem.rdchem.AtomMonomerType.PDBRESIDUE
                )
                and not info.GetIsHeteroAtom()
                and not StandardPDBChiralAtom(info.GetResidueName(), info.GetName())
            ):
                a.SetChiralTag(rdkit.Chem.rdchem.CHI_UNSPECIFIED)
                # TODO: Missing CIPCode unsetting


def StandardPDBChiralAtom(resname, atomname):
    if resname == "GLY":
        return False
    if resname == "ILE" or resname == "THR":
        return atomname == "CA" or atomname == "CB"
    if resname in (
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "HIS",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "TRP",
        "TYR",
        "VAL",
    ):
        return atomname == "CA"
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
        a.SetFormalCharge(int(mol.formalcharge[i]))
        info = Chem.AtomPDBResidueInfo(
            atomName=mol.name[i],
            serialNumber=int(mol.serial[i]),
            altLoc=mol.altloc[i],
            residueName=mol.resname[i],
            residueNumber=int(mol.resid[i]),
            chainId=mol.chain[i],
            insertionCode=mol.insertion[i],
            occupancy=float(mol.occupancy[i]),
            tempFactor=float(mol.beta[i]),
            isHeteroAtom=mol.record[i] == "HETATM",
        )
        a.SetMonomerInfo(info)

        rdmol.AddAtom(a)
        atomlist.append(a)
        coor = [float(c) for c in mol.coords[i, :, mol.frame]]
        conformer.SetAtomPosition(
            ii, Point3D(coor[0], coor[1], coor[2])
        )  # Correct the atom idx
    rdmol.AddConformer(conformer)

    # Here I diverge from the C++ parser because you cannot instantiate Chem.Bond objects in python
    # I also don't take into account double/triple bonds etc since I don't think we actually store them in Molecule
    for b in mol._getBonds():
        if b[0] in protidx and b[1] in protidx:
            bond = rdmol.GetBondBetweenAtoms(int(b[0]), int(b[1]))
            if bond is None:
                rdmol.AddBond(
                    int(np.where(protidx == b[0])[0]),
                    int(np.where(protidx == b[1])[0]),
                    Chem.BondType.SINGLE,
                )

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
