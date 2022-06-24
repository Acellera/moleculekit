from moleculekit.molecule import Molecule, getBondedGroups
from moleculekit.home import home
import numpy as np
import json
import os

from moleculekit.util import sequenceID

_sel = os.path.join(home(shareDir="atomselect"), "atomselect.json")
with open(_sel, "r") as f:
    _sel = json.load(f)


def analyze(mol: Molecule, bonds):
    from moleculekit.util import sequenceID

    analysis = {}
    analysis["waters"] = np.isin(mol.resname, _sel["water_resnames"])
    analysis["lipids"] = np.isin(mol.resname, _sel["lipid_resnames"])
    analysis["ions"] = np.isin(mol.resname, _sel["ion_resnames"])
    analysis["residues"] = sequenceID(
        (mol.resid, mol.insertion, mol.chain, mol.segid)
    ).astype(np.uint32)
    analysis["protein_bb"] = find_backbone(mol, "protein")
    analysis["nucleic_bb"] = find_backbone(mol, "nucleic")
    analysis["protein"], analysis["nucleic"] = find_protein_nucleic(
        mol,
        analysis["protein_bb"],
        analysis["nucleic_bb"],
        analysis["residues"],
        bonds,
    )
    # Fix BB atoms by unmarking them if they are not polymers
    # This is necessary since we use just N CA C O names and other
    # molecules such as waters or ligands might have them
    analysis["protein_bb"] &= analysis["protein"]
    analysis["nucleic_bb"] &= analysis["nucleic"]
    _, analysis["fragments"] = getBondedGroups(mol, bonds)

    return analysis


def find_backbone(mol: Molecule, mode):
    if mode == "protein":
        backb = np.isin(mol.name, _sel["protein_backbone_names"])
        terms = np.isin(mol.name, _sel["protein_terminal_names"])
    elif mode == "nucleic":
        backb = np.isin(mol.name, _sel["nucleic_backbone_names"])
        terms = np.isin(mol.name, _sel["nucleic_terminal_names"])
    else:
        raise RuntimeError(f"Invalid backbone mode {mode}")

    for tt in np.where(terms)[0]:
        # Check if atoms bonded to terminal Os are backbone
        nn = mol.getNeighbors(tt)
        for n in nn:
            if backb[n]:  # If bonded atom is backbone break
                break
        else:
            # Could not find any backbone atom bonded to the term
            terms[tt] = False
    return backb | terms


def find_protein_nucleic(mol, pbb, nbb, uqres, bonds):
    from moleculekit.atomselect_utils import find_protein_nucleic_ext

    protein = np.zeros(mol.numAtoms, dtype=bool)
    nucleic = np.zeros(mol.numAtoms, dtype=bool)
    segcrossers = np.zeros(mol.numAtoms, dtype=bool)

    uqseg = sequenceID((mol.chain,))  # mol.segid  had disagreements with VMD
    bondsegs = uqseg[bonds]

    # Find atoms with bonds which cross segments
    segcrossers[bonds[bondsegs[:, 0] != bondsegs[:, 1]].flatten()] = True

    find_protein_nucleic_ext(protein, nucleic, pbb, nbb, segcrossers, uqres)

    return protein, nucleic


def test_analysis():
    from moleculekit.atomselect_utils import analyze_molecule
    from moleculekit.molecule import Molecule, calculateUniqueBonds
    from moleculekit.atomselect.analyze import find_backbone
    import numpy as np

    mol = Molecule("5vbl")
    insertion = np.unique(mol.insertion, return_inverse=True)[1].astype(np.uint32)
    chain_id = np.unique(mol.chain, return_inverse=True)[1].astype(np.uint32)
    seg_id = np.unique(mol.segid, return_inverse=True)[1].astype(np.uint32)
    bonds, _ = calculateUniqueBonds(mol.bonds.astype(np.uint32), [])
    analysis = {}
    analysis["waters"] = np.isin(mol.resname, _sel["water_resnames"])
    analysis["lipids"] = np.isin(mol.resname, _sel["lipid_resnames"])
    analysis["ions"] = np.isin(mol.resname, _sel["ion_resnames"])
    analysis["residues"] = np.zeros(mol.numAtoms, dtype=np.uint32)
    analysis["protein_bb"] = find_backbone(mol, "protein")
    analysis["nucleic_bb"] = find_backbone(mol, "nucleic")
    analysis["protein"] = np.zeros(mol.numAtoms, dtype=bool)
    analysis["nucleic"] = np.zeros(mol.numAtoms, dtype=bool)
    atom_bonds, residue_atoms = analyze_molecule(
        mol.numAtoms,
        bonds,
        mol.resid.astype(np.uint32),
        insertion,
        chain_id,
        seg_id,
        analysis["protein"],
        analysis["nucleic"],
        analysis["protein_bb"],
        analysis["nucleic_bb"],
        analysis["residues"],
    )
    # Fix BB atoms by unmarking them if they are not polymers
    # This is necessary since we use just N CA C O names and other
    # molecules such as waters or ligands might have them
    analysis["protein_bb"] &= analysis["protein"]
    analysis["nucleic_bb"] &= analysis["nucleic"]
    return analysis, atom_bonds, residue_atoms


def timerun():
    atom_bonds, residue_atoms = analyze_molecule(
        mol.numAtoms,
        bonds,
        mol.resid.astype(np.uint32),
        insertion,
        chain_id,
        seg_id,
        analysis["protein"],
        analysis["nucleic"],
        analysis["protein_bb"],
        analysis["nucleic_bb"],
        analysis["residues"],
    )
