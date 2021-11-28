from moleculekit.molecule import Molecule
from pdb2pqr.aa import Amino
import numpy as np
import os


# PDB2PQR is quite finicky about the backbone coordinates so I copy them from ALA
backbone = Molecule().empty(6)
backbone.name[:] = ["N", "CA", "C", "O", "H", "HA"]
backbone.element[:] = ["N", "C", "C", "O", "H", "H"]
backbone.coords = (
    np.array(
        [
            [1.2010, 0.8470, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [-1.250, 0.8810, 0.0000],
            [-2.185, 0.6600, -0.784],
            [1.2010, 1.8470, 0.0000],
            [0.0000, -0.557, -0.831],
        ]
    )
    .reshape((-1, 3, 1))
    .astype(np.float32)
    .copy()
)
backbone.bonds = np.array([[0, 1], [0, 4], [1, 5], [1, 2], [2, 3]])
backbone.bondtype = np.array(["1", "1", "1", "1", "2"], dtype=object)


def _get_residue_mol(mol: Molecule, nsres: str):
    from rdkit import Chem
    from aceprep.detector import getLigandFromDict
    from aceprep.prepare import RDKprepare
    import tempfile
    import logging

    acepreplog = logging.getLogger("aceprep")
    oldlevel = acepreplog.getEffectiveLevel()
    acepreplog.setLevel("CRITICAL")

    try:
        assert np.all(np.isin(["N", "CA", "C", "O"], mol.name))

        with tempfile.TemporaryDirectory() as outdir:
            resfile = os.path.join(outdir, f"residue_{nsres}.pdb")
            mol.write(resfile)

            outsdf = os.path.join(outdir, f"residue_{nsres}_templated.sdf")
            new_mol = getLigandFromDict(resfile, nsres)
            w = Chem.SDWriter(outsdf)
            w.write(new_mol)
            w.close()

            outsdfh = outsdf.replace(".sdf", "_h.sdf")
            RDKprepare(
                outsdf, outsdfh, os.path.join(outdir, "aceprep.log"), gen3d=False
            )

            mol = Molecule(outsdfh)
    except Exception:
        acepreplog.setLevel(oldlevel)
        raise

    acepreplog.setLevel(oldlevel)
    return mol


def _generate_custom_residue(inmol: Molecule, resname: str):
    from moleculekit.tools.graphalignment import makeMolGraph, compareGraphs
    import networkx as nx

    def get_idx(mol, name):
        res = np.where(mol.name == name)
        if len(res) == 0 or len(res[0]) == 0:
            return None
        return res[0][0]

    mol = _get_residue_mol(inmol, resname)
    # Adding an X_ to all atom names to separate them from the matched names
    for i, nn in enumerate(mol.name):
        mol.name[i] = f"X_{nn}"

    fields = ("element",)
    g1 = makeMolGraph(mol, "all", fields)
    g2 = makeMolGraph(inmol, "all", fields)
    _, _, matching = compareGraphs(
        g1, g2, fields=fields, tolerance=0.5, returnmatching=True
    )
    for pp in matching:  # Rename atoms in reference molecule
        mol.name[pp[0]] = inmol.name[pp[1]]

    # Align and replace the backbone with the standard backbone of pdb2pqr
    mol.align("name N CA C", refmol=backbone)

    # Remove everything connected to CA
    gg = mol.toGraph()
    gg.remove_edge(get_idx(mol, "CA"), get_idx(mol, "CB"))
    cd_idx = get_idx(mol, "CD")
    n_idx = get_idx(mol, "N")
    has_n_cd_bond = False
    if cd_idx is not None and gg.has_edge(n_idx, cd_idx):
        gg.remove_edge(n_idx, cd_idx)  # Prolines
        has_n_cd_bond = True
    to_remove = np.array(list(nx.node_connected_component(gg, get_idx(mol, "CA"))))
    mol.remove(to_remove, _logger=False)

    # Insert the standard backbone
    mol.insert(backbone, index=0)

    # Add back CA CB bond
    mol.bonds = np.vstack(([get_idx(mol, "CA"), get_idx(mol, "CB")], mol.bonds))
    mol.bondtype = np.hstack(("1", mol.bondtype))
    if has_n_cd_bond:
        mol.bonds = np.vstack(([get_idx(mol, "N"), get_idx(mol, "CD")], mol.bonds))
        mol.bondtype = np.hstack(("1", mol.bondtype))

    # Rename all added hydrogens
    hydr = mol.name == "X_H"
    mol.name[hydr] = [f"H{i}" for i in range(1, sum(hydr) + 1)]

    # Rename residues
    mol.resname[:] = resname
    # AMBER format is: N H CA HA [sidechain] C O
    mol.reorderAtoms([0, 4, 1, 5] + list(range(6, mol.numAtoms)) + [2, 3])

    return mol


class CustomResidue(Amino):
    """Custom residue class. Hack for pdb2pqr which requires one class per residue in pdb2pqr.aa module"""

    def __init__(self, atoms, ref):
        Amino.__init__(self, atoms, ref)
        self.reference = ref

    def letter_code(self):
        return "X"


def _mol_to_xml_def(mol: Molecule, outfile: str):
    with open(outfile, "w") as f:
        f.write("<?xml version='1.0'?>\n")
        f.write("<aminoacids>\n")
        f.write("  <residue>\n")
        f.write(f"    <name>{mol.resname[0]}</name>\n")

        for i in range(mol.numAtoms):
            f.write("    <atom>")
            f.write(
                f"""
      <name>{mol.name[i]}</name>
      <x>{mol.coords[i, 0, 0]:.3f}</x>
      <y>{mol.coords[i, 1, 0]:.3f}</y>
      <z>{mol.coords[i, 2, 0]:.3f}</z>\n"""
            )
            atombonds = mol.bonds[np.any(mol.bonds == i, axis=1), :].flatten()
            for b in sorted(atombonds):
                if b == i:
                    continue
                f.write(f"      <bond>{mol.name[b]}</bond>\n")
            f.write("    </atom>\n")
        f.write("  </residue>\n")
        f.write("</aminoacids>\n")


def _mol_to_dat_def(mol: Molecule, outfile: str):
    with open(outfile, "w") as f:
        for i in range(mol.numAtoms):
            radius = 0
            f.write(
                f"{mol.resname[i]}\t{mol.name[i]}\t{mol.charge[i]:.3f}\t{radius:.1f}\n"
            )


def _generate_mol2(outsdfh):
    from subprocess import call

    tmpmol = Molecule(outsdfh)
    nc = int(tmpmol.charge.sum())

    nshmol2 = outsdfh.replace(".sdf", ".mol2")
    call(
        [
            "antechamber",
            "-i",
            outsdfh,
            "-fi",
            "sdf",
            "-o",
            nshmol2,
            "-fo",
            "mol2",
            "-c",  # charge guessing method
            "gas",
            "-at",  # atom types
            "gaff2",
            "-s",  # verbosity
            "0",
            "-nc",  # net charge
            str(nc),
            "-j",  # atom type and bond type prediction index. 1: atom type (only)
            "1",
            "-pf",  # remove intermediate files
            "y",
        ]
    )
    frcmodfile = nshmol2.replace(".mol2", ".frcmod")
    call(
        [
            "parmchk2",
            "-i",
            nshmol2,
            "-o",
            frcmodfile,
            "-f",
            "mol2",
        ]
    )
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     leapin = os.path.join(tmpdir, "tleap.in")
    #     leaptempl = os.path.join(
    #         home(), "newprep/build_templates/non-standard-aa/tleap.in"
    #     )
    #     outlib = os.path.join(outdir, f"{ns}.lib")
    #     with open(leaptempl) as fin, open(leapin, "w") as fout:
    #         template = Template(fin.read())
    #         fout.write(
    #             template.render(
    #                 mol2file=nshmol2,
    #                 frcmodfile=frcmodfile,
    #                 molname=ns,
    #                 outlib=outlib,
    #             )
    #         )
    #     call([teleap, *teleapimportflags, "-f", leapin])


def _get_custom_ff(user_ff=None, molkit_ff=True):
    import xml.etree.ElementTree as ET
    import pdb2pqr
    import pdb2pqr.aa
    import tempfile
    from moleculekit.home import home
    import shutil
    from glob import glob

    original = os.path.join(os.path.dirname(pdb2pqr.__file__), "dat")
    molkitcustom = home(shareDir=os.path.join("pdb2pqr", "residues"))

    custom_xml = []
    if molkit_ff:
        custom_xml += glob(os.path.join(molkitcustom, "*.xml"))
    if user_ff is not None:
        custom_xml += glob(os.path.join(user_ff, "*.xml"))

    custom_dat = []
    if molkit_ff:
        custom_dat += glob(os.path.join(molkitcustom, "*.dat"))
    if user_ff is not None:
        custom_dat += glob(os.path.join(user_ff, "*.dat"))

    custom_resnames = np.unique(
        [os.path.splitext(os.path.basename(x))[0] for x in custom_xml]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copytree(original, os.path.join(tmpdir, "originals"))

        # Merge moleculekit custom residues into AA.xml
        residues = {}
        with open(os.path.join(tmpdir, "originals", "AA.xml")) as fh:
            tree = ET.parse(fh)
            root = tree.getroot()
            for res in root.iter("residue"):
                residues[res.find("name").text] = res

        for cxml in custom_xml:
            with open(cxml, "r") as fh:
                tree = ET.parse(fh)
                root = tree.getroot()
                for res in root.iter("residue"):
                    residues[res.find("name").text] = res

        root = ET.Element("aminoacids")
        for name in sorted(residues):
            root.append(residues[name])

        out_aa = os.path.join(tmpdir, "AA.xml")
        with open(out_aa, "w") as fout:
            fout.write(ET.tostring(root, encoding="unicode"))

        # Merge moleculekit custom residues into PARSE.DAT
        with open(os.path.join(tmpdir, "originals", "PARSE.DAT"), "r") as fin:
            lines = fin.readlines()
            lines[-1] = lines[-1].strip() + "\n"

        for cdat in custom_dat:
            with open(cdat, "r") as fc:
                custom_lines = fc.readlines()
            lines += custom_lines

        # Write out the new PARSE.DAT
        out_dat = os.path.join(tmpdir, "PARSE.DAT")
        with open(out_dat, "w") as fout:
            for line in lines:
                fout.write(line)

        # HACK: pdb2pqr currently requires each residue to have a unique class in pdb2pqr.aa module
        for resn in custom_resnames:
            pdb2pqr.aa.__dict__[resn] = CustomResidue

        from pdb2pqr.io import get_definitions
        from pdb2pqr import forcefield

        definition = get_definitions(aa_path=out_aa)
        ff = forcefield.Forcefield("parse", definition, out_dat, None)
    return definition, ff
