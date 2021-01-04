from moleculekit.tools.voxeldescriptors import getChannels, _order
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.smallmol.util import _highlight_colors
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
import os

legend_patches = []
for i, prop in enumerate(_order[:-1]):
    legend_patches.append(mpatches.Patch(color=_highlight_colors[i], label=prop))


def getAtomBonds(mol, idx):
    bond_partners = np.setdiff1d(
        mol.bonds[np.where(mol.bonds == idx)[0], :].flatten(), idx
    )
    partner_elem = mol.element[bond_partners]
    heavy_bonds = (partner_elem != "H").sum()
    h_bonds = len(partner_elem) - heavy_bonds
    return heavy_bonds, h_bonds


def getAtomLabel(atomname, heavy_bonds, h_bonds):
    return f"{atomname}_{heavy_bonds}_{h_bonds}"


def depictResidue(mol, resname, protein_atomtypes, imgpath):
    os.makedirs(os.path.dirname(imgpath), exist_ok=True)

    res = mol.copy()
    residx = np.where(res.resname == resname)[0]
    firstgap = np.where(np.diff(residx) != 1)[0]
    raise RuntimeError("WHAT IF THEY ARE CONSECUTIVE! FIND OTHER SOLUTION")
    if len(firstgap):
        residx = residx[: firstgap[0] + 1]

    res.filter(f"index {' '.join(map(str, residx))}")
    if res.numAtoms == 0:
        return

    atom_groups = [[], [], [], [], [], [], []]
    for i, (rn, an, el) in enumerate(zip(res.resname, res.name, res.element)):
        if el == "H":
            continue

        heavy_bonds, h_bonds = getAtomBonds(res, i)
        key = getAtomLabel(an, heavy_bonds, h_bonds)
        # Fixing the backbone since the bonds are severed
        if an == "N":
            key = "N_2_1" if not rn == "PRO" else "N_3_0"
        if an == "O" and not rn == "HOH":
            key = "O_1_0"
        if an == "C":
            key = "C_3_0"
        if an == "CA" and rn != "CA":
            key = "CA_3_1" if rn != "GLY" else "CA_2_2"
        if rn == "CYX" and an == "SG":  # Fix the bonds of S in CYX for viz
            key = "SG_2_0"
        props = protein_atomtypes[rn][key]["properties"]
        for prop in props:
            idx = _order.index(prop)
            atom_groups[idx].append(i)

    # res.name = res.element
    res.atomtype[:] = ""
    sm = SmallMol(res, force_reading=True, removeHs=False, fixHs=False)
    sm.depict(
        filename=imgpath,
        highlightAtoms=atom_groups,
        removeHs=False,
        resolution=(800, 400),
        atomlabels="%a",
    )

    img = mpimg.imread(imgpath)
    plt.figure(figsize=(13, 5))
    plt.imshow(img)
    plt.title(resname)
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.2, 1))
    plt.savefig(imgpath)
    plt.close()


def getOpenBabelResidueDictionary(mol):
    channels, newmol = getChannels(mol, True, 2, False)

    resdict = {res: {} for res in np.unique(newmol.resname)}
    for i, (rn, an, el, ch) in enumerate(
        zip(newmol.resname, newmol.name, newmol.element, channels)
    ):
        if el == "H":
            continue

        props = np.array(_order[:-1])[ch[:-1] != 0].tolist()
        heavy_bonds, h_bonds = getAtomBonds(mol, i)
        key = getAtomLabel(an, heavy_bonds, h_bonds)

        if key not in resdict[rn]:
            resdict[rn][key] = {}

        if an in ["C", "CA"]:
            props = []
        if an == "N" and rn == "PRO":
            props = []
        resdict[rn][key] = {
            "element": str(el),
            "heavy_bonds": int(heavy_bonds),
            "h_bonds": int(h_bonds),
            "properties": props,
        }
    return resdict


if __name__ == "__main__":
    from moleculekit.molecule import Molecule
    from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
    import yaml

    mol = Molecule("3ptb")
    mol.filter("protein or water")
    pmol = prepareProteinForAtomtyping(mol)

    # Either compute OB atom types
    protein_atomtypes = getOpenBabelResidueDictionary(pmol)

    # Or read them from our file
    with open("protein_atomtypes.yaml", "r") as f:
        protein_atomtypes = yaml.load(f, Loader=yaml.BaseLoader)

    for rn in protein_atomtypes:
        depictResidue(pmol, rn, protein_atomtypes, os.path.join("images", f"{rn}.png"))
