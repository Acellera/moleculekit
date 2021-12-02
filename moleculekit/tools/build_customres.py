from parameterize.parameterization.cli import main_parameterize
from moleculekit.tools.graphalignment import makeMolGraph, compareGraphs
from moleculekit.molecule import Molecule
from moleculekit.home import home
import shutil
import parmed
import numpy as np
import tempfile
import os
import types


def get_calculator():
    from energyforcecalculators.torchanicalculator import TorchAniCalculator

    return TorchAniCalculator(model="ANI-2x")


mymod = types.ModuleType("wrappers.torchani2x")
mymod.get_calculator = get_calculator

ace = Molecule(os.path.join(home(shareDir="builder"), "ACE.cif"))
nme = Molecule(os.path.join(home(shareDir="builder"), "NME.cif"))
backbone_prm = parmed.amber.AmberParameterSet(
    os.path.join(home(shareDir="builder"), "backbone.frcmod")
)


def _cap_residue(mol):
    mol = mol.copy()
    resn = mol.resname[0]

    acec = ace.copy()
    acec.align("name N H CA", refmol=mol)
    acec.remove("name N H CA", _logger=False)

    nmec = nme.copy()
    nmec.align("name CA C O", refmol=mol)
    nmec.remove("name CA C O", _logger=False)

    mol.insert(acec, index=0)
    mol.append(nmec)
    # Bond ACE/NME to the residue
    mol.bonds = np.vstack(
        (
            mol.bonds,
            [
                np.where((mol.resname == "NME") & (mol.name == "N"))[0][0],
                np.where((mol.resname == resn) & (mol.name == "C"))[0][0],
            ],
            [
                np.where((mol.resname == "ACE") & (mol.name == "C"))[0][0],
                np.where((mol.resname == resn) & (mol.name == "N"))[0][0],
            ],
        )
    )
    mol.bondtype = np.hstack((mol.bondtype, ["1"], ["1"]))
    return mol


def _parameterize_custom_residue(mol, outdir):
    mol = mol.copy()
    # Remove backbone formal charge from templated molecule
    mol.formalcharge[mol.name == "N"] = 0
    resn = mol.resname[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        cmol = _cap_residue(mol)

        sdffile = os.path.join(tmpdir, "mol.sdf")
        cmol.write(sdffile)
        # TODO: Improve it to not parameterize the cap dihedrals!
        main_parameterize(
            [
                sdffile,
                "--charge",
                str(int(cmol.formalcharge.sum())),
                "--nnp",
                "wrappers.torchani2x",
                "--charge-type",
                "AM1-BCC",
                "--forcefield",
                "GAFF2",
                "--min-type",
                "mm",
                "--scan-type",
                "mm",
                "--dihed-fit-type",
                "iterative",
                "--outdir",
                tmpdir,
            ]
        )
        shutil.copy(
            os.path.join(tmpdir, "parameters", "GAFF2", "mol-orig.mol2"),
            os.path.join(outdir, "mol.mol2"),
        )
        shutil.copy(
            os.path.join(tmpdir, "parameters", "GAFF2", "mol.frcmod"),
            os.path.join(outdir, "mol.frcmod"),
        )


def _post_process_parameterize(cmol, outdir):
    # TODO: Move this to parameterize (?)
    mol = Molecule(os.path.join(outdir, "mol.mol2"))
    fields = ("element",)
    g1 = makeMolGraph(cmol, "all", fields)
    g2 = makeMolGraph(mol, "all", fields)
    _, _, matching = compareGraphs(
        g1, g2, fields=fields, tolerance=0.5, returnmatching=True
    )
    for pp in matching:  # Rename atoms in reference molecule
        mol.name[pp[0]] = cmol.name[pp[1]]

    # Remove the caps
    mask = np.ones(mol.numAtoms, dtype=bool)
    mask[:6] = False
    mask[-6:] = False

    # Rename backbone atom types
    backbone_at = {"N": "N", "H": "H", "CA": "CT", "HA": "H1", "C": "C", "O": "O"}
    original = {}
    for key, val in backbone_at.items():
        original[key] = mol.atomtype[mol.name == key][0]
        mol.atomtype[mol.name == key] = val

    mol.filter(mask, _logger=False)
    mol.write(os.path.join(outdir, "mol_mod.mol2"))

    prm = parmed.amber.AmberParameterSet(os.path.join(outdir, "mol.frcmod"))
    # Remove unused parameters

    # Add backbone parameters

    # Duplicate parameters for renamed atoms which don't exist in the backbone
