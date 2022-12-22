from moleculekit.molecule import Molecule
from moleculekit.home import home
import numpy as np
import json
import os

_sel = os.path.join(home(shareDir="atomselect"), "atomselect.json")
with open(_sel, "r") as f:
    _sel = json.load(f)


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


def _guessMass(name):
    val = 12  # Default mass if element cannot be found
    while len(name) and name[0].isdigit():
        name = name[1:]

    if len(name):
        nn = name[0].upper()
        if nn == "H":
            val = 1.008
        elif nn == "C":
            val = 12.011
        elif nn == "N":
            val = 14.007
        elif nn == "O":
            val = 15.999
        elif nn == "F":
            val = 55.847
        elif nn == "P":
            val = 30.97376
        elif nn == "S":
            val = 32.06

        if len(name) > 1:
            nn2 = name[1].upper()
            if nn == "C" and nn2 == "L":
                val = 35.453  # Chlorine
            elif nn == "N" and nn2 == "A":
                val = 22.98977  # Natrium
            elif nn == "M" and nn2 == "G":
                val = 24.3050  # Magnesium
    return val


def analyze(mol: Molecule, bonds, _profile=False):
    from moleculekit.atomselect_utils import analyze_molecule
    from moleculekit.molecule import calculateUniqueBonds
    from moleculekit.periodictable import periodictable
    import numpy as np

    insertion = np.unique(mol.insertion, return_inverse=True)[1].astype(np.uint32)
    chain_id = np.unique(mol.chain, return_inverse=True)[1].astype(np.uint32)
    seg_id = np.unique(mol.segid, return_inverse=True)[1].astype(np.uint32)
    bonds = bonds.astype(np.uint32)
    if bonds.size != 0:
        bonds, _ = calculateUniqueBonds(bonds, [])
    analysis = {}
    analysis["waters"] = np.isin(mol.resname, _sel["water_resnames"])
    analysis["lipids"] = np.isin(mol.resname, _sel["lipid_resnames"])
    analysis["ions"] = np.isin(mol.resname, _sel["ion_resnames"])
    analysis["residues"] = np.zeros(mol.numAtoms, dtype=np.uint32)
    analysis["protein_bb"] = find_backbone(mol, "protein")
    analysis["nucleic_bb"] = find_backbone(mol, "nucleic")
    analysis["protein"] = np.zeros(mol.numAtoms, dtype=bool)
    analysis["nucleic"] = np.zeros(mol.numAtoms, dtype=bool)
    analysis["fragments"] = np.full(mol.numAtoms, mol.numAtoms + 1, dtype=np.uint32)
    analysis["sidechain"] = np.zeros(mol.numAtoms, dtype=np.uint32)
    masses = []
    for i, el in enumerate(mol.element):
        if el not in periodictable:
            masses.append(_guessMass(mol.name[i]))
            continue
        masses.append(periodictable[el].mass)
    masses = np.array(masses, dtype=np.float32)

    if not _profile:
        analyze_molecule(
            mol.numAtoms,
            bonds,
            mol.resid.astype(np.int64),
            insertion,
            chain_id,
            seg_id,
            analysis["protein"],
            analysis["nucleic"],
            analysis["protein_bb"],
            analysis["nucleic_bb"],
            analysis["residues"],
            analysis["fragments"],
            masses,
            analysis["sidechain"],
            [x.encode("utf-8") for x in mol.name],
        )
    else:
        import pstats, cProfile

        cProfile.runctx(
            'analyze_molecule(mol.numAtoms,bonds,mol.resid,insertion,chain_id,seg_id,analysis["protein"],analysis["nucleic"],analysis["protein_bb"],analysis["nucleic_bb"],analysis["residues"],mol.name == "SG",analysis["fragments"])',
            globals(),
            locals(),
            "Profile.prof",
        )
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()

    analysis["sidechain"] = analysis["sidechain"] > 0
    # assert not np.any(analysis["fragments"] == (mol.numAtoms + 1))
    return analysis  # , atom_bonds, residue_atoms
