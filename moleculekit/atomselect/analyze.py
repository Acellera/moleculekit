from moleculekit.molecule import Molecule, getBondedGroups
from moleculekit.home import home
import numpy as np
import json
import os

_sel = os.path.join(home(shareDir="atomselect"), "atomselect.json")
with open(_sel, "r") as f:
    _sel = json.load(f)


def analyze(mol: Molecule, bonds):
    from moleculekit.util import sequenceID

    analysis = {}
    analysis["waters"] = np.isin(mol.resname, _sel["water_resnames"])
    analysis["lipids"] = np.isin(mol.resname, _sel["lipid_resnames"])
    analysis["ions"] = np.isin(mol.resname, _sel["ion_resnames"])
    analysis["residues"] = sequenceID((mol.resid, mol.insertion, mol.chain, mol.segid))
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
    analysis["fragments"] = find_fragments(mol, bonds)

    return analysis


def find_fragments(mol, bonds):
    fragments = np.zeros(mol.numAtoms, dtype=bool)
    groups = getBondedGroups(mol, bonds)
    for i in range(len(groups) - 1):
        fragments[groups[i] : groups[i + 1]] = i
    return fragments


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
    protein = np.zeros(mol.numAtoms, dtype=bool)
    nucleic = np.zeros(mol.numAtoms, dtype=bool)

    for ri in np.unique(uqres):
        resmask = uqres == ri
        chainmask = mol.chain == mol.chain[resmask][0]
        segmask = mol.segid == mol.segid[resmask][0]
        chainsegidx = np.where(chainmask | segmask)[0]

        # If there are 4 backbone atoms in this residue it's protein/nucleic
        isprot = pbb[resmask].sum() >= 4
        isnucl = nbb[resmask].sum() >= 4

        if not isprot and not isnucl:
            continue

        array = protein if isprot else nucleic
        bb = pbb if isprot else nbb

        resbb = resmask & bb
        array[resbb] = True  # BB is protein

        # Check which atoms in residue are only bonded to this chain/segment
        resnotbb = resmask & ~bb
        resnotbb_idx = np.where(resnotbb)[0]
        badbondrows = np.sum(np.isin(bonds, chainsegidx), axis=1) == 1
        resnotbb_idx = np.setdiff1d(resnotbb_idx, bonds[badbondrows].flatten())
        array[resnotbb_idx] = True

    return protein, nucleic
