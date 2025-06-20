# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from moleculekit.molecule import Molecule
from moleculekit import __share_dir
from pdb2pqr.aa import Amino
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


# PDB2PQR is quite finicky about the backbone coordinates so I copy them from ALA
backbone = Molecule(os.path.join(__share_dir, "backbone.cif"), zerowarning=False)
alanine = Molecule(os.path.join(__share_dir, "ALA.cif"), zerowarning=False)


def _reorder_residue_atoms(mol, resid):
    # Reorder atoms. AMBER order is: N H CA HA [sidechain] C O
    # the H atom will get added later
    first_bbatoms = [_get_idx(mol, x, resid) for x in ["N", "CA", "HA"]]
    first_bbatoms = [x for x in first_bbatoms if x is not None]
    last_bbatoms = [_get_idx(mol, x, resid) for x in ["C", "O"]]
    last_bbatoms = [x for x in last_bbatoms if x is not None]
    other_idx = np.setdiff1d(
        np.where(mol.resid == resid)[0], first_bbatoms + last_bbatoms
    ).tolist()
    prev_res = np.where(mol.resid == resid)[0][0]
    if prev_res > 0:
        prev_res = list(range(prev_res))
    else:
        prev_res = []
    next_res = np.where(mol.resid == resid)[0][-1] + 1
    if next_res < mol.numAtoms:
        next_res = list(range(next_res, mol.numAtoms))
    else:
        next_res = []
    mol.reorderAtoms(prev_res + first_bbatoms + other_idx + last_bbatoms + next_res)


def _get_idx(mol, name, resid=None):
    sel = mol.name == name
    if resid is not None:
        sel &= mol.resid == resid
    res = np.where(sel)
    if len(res) == 0 or len(res[0]) == 0:
        return None
    assert len(res[0]) == 1
    return res[0][0]


def _process_custom_residue(mol: Molecule, resid: int = None, align: bool = True):
    import networkx as nx

    if resid is None:
        resid = mol.resid[0]
    resname = mol.resname[mol.resid == resid][0]

    gg = mol.toGraph()
    n_idx = _get_idx(mol, "N", resid)
    c_idx = _get_idx(mol, "C", resid)
    if n_idx is None or c_idx is None:
        raise RuntimeError(
            f"Residue {resname} does not contain N or C atoms. List of atoms: {mol.name}"
        )

    sp = nx.shortest_path(gg, n_idx, c_idx)
    if len(sp) != 3:
        raise RuntimeError(
            f"Cannot prepare residues with elongated backbones. This backbone consists of atoms {' '.join(mol.name[sp])}"
        )

    # Fix hydrogen names for CA / N
    ca_idx = _get_idx(mol, "CA", resid)
    ca_hs = [nn for nn in gg.neighbors(ca_idx) if gg.nodes[nn]["element"] == "H"]
    if len(ca_hs) > 1:
        raise RuntimeError("Found more than 1 hydrogen on CA atom!")
    if len(ca_hs) == 1:
        mol.name[ca_hs[0]] = "HA"

    # Remove all N terminal hydrogens
    gg = mol.toGraph()
    n_idx = _get_idx(mol, "N", resid)
    n_neighbours = list(gg.neighbors(n_idx))
    n_hs = [nn for nn in n_neighbours if gg.nodes[nn]["element"] == "H"]
    n_heavy = len(n_neighbours) - len(n_hs)
    if len(n_hs):
        mol.remove(f"index {' '.join(map(str, n_hs))}", _logger=False)

    # Remove all hydrogens attached to terminal C
    gg = mol.toGraph()
    idx = _get_idx(mol, "C", resid)
    neighbours = list(gg.neighbors(idx))
    hs = [nn for nn in neighbours if gg.nodes[nn]["element"] == "H"]
    if len(hs):
        mol.remove(f"index {' '.join(map(str, hs))}", _logger=False)

    # Remove all hydrogens attached to C-terminal O
    gg = mol.toGraph()
    idx = _get_idx(mol, "O", resid)
    neighbours = list(gg.neighbors(idx))
    hs = [nn for nn in neighbours if gg.nodes[nn]["element"] == "H"]
    if len(hs):
        mol.remove(f"index {' '.join(map(str, hs))}", _logger=False)

    # Rename all non-matched hydrogens
    hydr = mol.name == "X_H"
    mol.name[hydr] = [f"H{i}" for i in range(10, sum(hydr) + 10)]

    _reorder_residue_atoms(mol, resid)

    if align:
        # Align to reference BB for pdb2pqr
        mol.align("name N CA C", refmol=backbone)

    if n_heavy == 1 and "N" in mol.name:
        # Add the H atom if N is only bonded to CA.
        # This is necessary to add it in the right position for pdb2pqr
        nmol = backbone.copy()
        if not align and resid is not None:
            nmol.align(
                "name N CA C", refsel=f"name N CA C and resid {resid}", refmol=mol
            )
        nmol.filter("name H", _logger=False)
        nmol.resname[:] = resname
        nmol.resid[:] = resid
        insert_idx = np.where(mol.resid == resid)[0][0] + 1
        mol.insert(nmol, insert_idx)
        mol.addBond(insert_idx - 1, insert_idx, "1")

    return mol


def _prepare_for_parameterize(mol, resid=None):
    # Add OXT HXT HN2 atoms to convert it to RCSB-like structures and pass it to parameterize
    import networkx as nx

    mol = mol.copy()
    if resid is None:
        resid = mol.resid[0]
    resname = mol.resname[mol.resid == resid][0]

    gg = mol.toGraph()
    bb = nx.shortest_path(gg, _get_idx(mol, "N", resid), _get_idx(mol, "C", resid))

    n_idx = _get_idx(mol, "N", resid)
    mol.formalcharge[n_idx] = 0
    n_neighbours = list(gg.neighbors(n_idx))
    if len(n_neighbours) == 2:
        # Add HN2 atom
        non_bb_idx = [nn for nn in n_neighbours if nn not in bb]
        align_idx = [n_idx, bb[1], non_bb_idx[0]]
        nterm = alanine.copy()
        nterm.align(
            [_get_idx(nterm, n) for n in ("N", "CA", "H")],
            refmol=mol,
            refsel=align_idx,
        )
        nterm.filter("name H2", _logger=False)
        nterm.name[0] = "HN2"
        nterm.resname[:] = resname
        nterm.resid[:] = resid
        insert_idx = np.where(mol.resid == resid)[0][0] + 1  # Second position
        mol.insert(nterm, insert_idx)
        mol.addBond(n_idx, insert_idx, "1")

    gg = mol.toGraph()
    bb = nx.shortest_path(gg, _get_idx(mol, "N", resid), _get_idx(mol, "C", resid))

    c_idx = _get_idx(mol, "C", resid)
    mol.formalcharge[c_idx] = 0
    c_neighbours = list(gg.neighbors(c_idx))
    if len(c_neighbours) == 2:
        # Add OXT HXT atoms
        non_bb_idx = [cc for cc in c_neighbours if cc not in bb]
        align_idx = [bb[-2], c_idx, non_bb_idx[0]]
        cterm = alanine.copy()
        cterm.align(
            [_get_idx(cterm, n) for n in ("CA", "C", "O")],
            refmol=mol,
            refsel=align_idx,
        )
        cterm.filter("name OXT HXT", _logger=False)
        cterm.resname[:] = resname
        cterm.resid[:] = resid
        insert_idx = np.where(mol.resid == resid)[0][-1] + 1  # End
        mol.insert(cterm, insert_idx)
        mol.addBond(c_idx, insert_idx, "1")

    # Fix C=O bond type
    _, _, idx = mol.hasBond(_get_idx(mol, "C", resid), _get_idx(mol, "O", resid))
    mol.bondtype[idx] = "2"

    # Reorder atoms. AMBER order is: N H CA HA [sidechain] C O
    _reorder_residue_atoms(mol, resid)

    return mol


def _convert_amber_prepi_to_pdb2pqr_residue(prepi, outdir, name=None):
    """
    Used as follows:

    prepis = glob("./htmd/share/builder/amberfiles/ff-ncaa/*.prepi") + glob("./htmd/share/builder/amberfiles/ff-ptm/*.prepi")
    for pp in prepis:
        try:
            _convert_amber_prepi_to_pdb2pqr_residue(pp, "/tmp/pdb2pqr_res/")
        except Exception as e:
            print(f"ERROR {e}")
    """
    import tempfile

    os.makedirs(outdir, exist_ok=True)

    if name is None:
        name = os.path.splitext(os.path.basename(prepi))[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        outf = os.path.join(tmpdir, f"{name}.mol2")
        outsdf = os.path.join(tmpdir, f"{name}.sdf")
        os.popen(f"antechamber -fi prepi -fo mol2 -i {prepi} -o {outf} -pf y").read()
        os.popen(f"antechamber -fi prepi -fo sdf -i {prepi} -o {outsdf} -pf y").read()
        try:
            mol = Molecule(outf, validateElements=False)  # Read everything
            sdf = Molecule(outsdf)  # Read elements
        except Exception as e:
            raise RuntimeError(f"ERROR: {e} {prepi}")

        diff = mol.element != sdf.element
        if any(diff):
            print(
                f"Different elements read: {mol.element[diff]} and {sdf.element[diff]}"
            )
        mol.element[:] = sdf.element[:]

        pmol = _process_custom_residue(mol)
        # Rename to correct resname
        pmol.resname[:] = name

        _mol_to_xml_def(pmol, os.path.join(outdir, f"{name}.xml"))
        _mol_to_dat_def(pmol, os.path.join(outdir, f"{name}.dat"))


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
                f"{mol.resname[i]}\t{mol.name[i]}\t{mol.charge[i]:6.3f}\t{radius:.1f}\n"
            )


def _get_custom_ff(user_ff=None, molkit_ff=True):
    import xml.etree.ElementTree as ET
    import pdb2pqr
    import pdb2pqr.aa
    import tempfile
    import shutil
    from glob import glob

    original = os.path.join(os.path.dirname(pdb2pqr.__file__), "dat")
    try:
        molkitcustom = os.path.join(__share_dir, "pdb2pqr", "residues")
    except Exception:
        molkitcustom = None

    custom_xml = []
    if molkit_ff and molkitcustom is not None:
        custom_xml += glob(os.path.join(molkitcustom, "*.xml"))
    if user_ff is not None:
        custom_xml += glob(os.path.join(user_ff, "*.xml"))

    custom_dat = []
    if molkit_ff and molkitcustom is not None:
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
