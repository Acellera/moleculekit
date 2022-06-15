from moleculekit.molecule import Molecule
from moleculekit.home import home
import numpy as np
import yaml
import os

selections = os.path.join(home(shareDir="atomselect"), "atomselect.yaml")
with open(selections, "r") as f:
    selections = yaml.load(f, Loader=yaml.BaseLoader)


def find_backbone(mol: Molecule, mode):
    if mode == "protein":
        backb = np.isin(mol.name, selections["protein_backbone_names"])
        terms = np.isin(mol.name, selections["protein_terminal_names"])
    elif mode == "nucleic":
        backb = np.isin(mol.name, selections["nucleic_backbone_names"])
        terms = np.isin(mol.name, selections["nucleic_terminal_names"])
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
