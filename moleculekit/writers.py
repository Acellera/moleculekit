# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np
import ctypes as ct
import os
from moleculekit.support import xtc_lib
from moleculekit.util import ensurelist
import networkx as nx
import collections
import logging
import numbers

logger = logging.getLogger(__name__)

# fmt: off

_Pair = collections.namedtuple('Atom', 'resname name')
# The following is taken from MDAnalysis to align atom names based on the strict PDB formatting.
# Some programs like AutoDock require this strict formatting to guess atom types.

# These attributes are used to deduce how to format the atom name.
_ions_metals = ('FE', 'AS', 'ZN', 'MG', 'MN', 'CO', 'BR',
                'CU', 'TA', 'MO', 'AL', 'BE', 'SE', 'PT',
                'EU', 'NI', 'IR', 'RH', 'AU', 'GD', 'RU', 
                'XE', 'RB', 'LU', 'GA', 'BA', 'CS', 'Y',
                'PB', 'SM', 'SR', 'YB')

# Mercurial can be confused for hydrogen gamma. Yet, mercurial is
# rather rare in the PDB. Here are all the residues that contain
# mercurial.
_special_hg = ('CMH', 'EMC', 'MBO', 'MMC', 'HGB', 'BE7', 'PMB')
# Chloride can be confused for a carbon. Here are the residues that
# contain chloride.
_special_cl = ('0QE', 'CPT', 'DCE', 'EAA', 'IMN', 'OCZ', 'OMY', 'OMZ',
               'UN9', '1N1', '2T8', '393', '3MY', 'BMU', 'CLM', 'CP6',
               'DB8', 'DIF', 'EFZ', 'LUR', 'RDC', 'UCL', 'XMM', 'HLT',
               'IRE', 'LCP', 'PCI', 'VGH')
# In these pairs, the atom name is aligned on the first column
# (column 13).
_include_pairs = (_Pair('OEC', 'CA1'),
                  _Pair('PLL', 'PD'),
                  _Pair('OEX', 'CA1'))
# In these pairs, the atom name is aligned on the second column
# (column 14), but other rules would align them on the first column.
_exclude_pairs = (_Pair('C14', 'C14'), _Pair('C15', 'C15'),
                  _Pair('F9F', 'F9F'), _Pair('OAN', 'OAN'),
                  _Pair('BLM', 'NI'), _Pair('BZG', 'CO'),
                  _Pair('BZG', 'NI'), _Pair('VNL', 'CO1'),
                  _Pair('VNL', 'CO2'), _Pair('PF5', 'FE1'),
                  _Pair('PF5', 'FE2'), _Pair('UNL', 'UNL'))

_rename_elements = {('SOD', 'SO'): 'Na'}

# After scanning most of the PDB database these are the most common elements which start on position 12
_position_12_elements = ['Se', 'Sr', 'Gd', 'Mn', 'Pt', 'Cs', 'Mg', 'Be', 'Sb', 'Bi', 'Th', 'Yb', 
                         'Zn', 'Kr', 'Rb', 'Ar', 'Pa', 'Fe', 'Ag', 'In', 'Eu', 'Hg', 
                         'Pr', 'Al', 'Ru', 'Sm', 'Ba', 'Lu', 'Re', 'Xe', 'Dy', 'Ti', 'As', 'Pu', 'Ir', 
                         'Br', 'Li', 'Pd', 'Ce', 'Cr', 'Ho', 'La', 'Sn', 'Te', 'Cl', 'Hf', 'Ga', 
                         'Co', 'Os', 'Tl', 'Au', 'Sc', 'Tb', 'Zr', 'Cu', 'Rh', 'Er', 'Mo', 'Si', 
                         'Ca', 'Ta', 'Cm', 'Am', 'Na', 'Pb', 'Ni', 'Ne', 'Cd']

# fmt: on


def _deduce_PDB_atom_name(name, resname, element=None):
    """Deduce how the atom name should be aligned.
    Atom name format can be deduced from the atom type, yet atom type is
    not always available. This function uses the atom name and residue name
    to deduce how the atom name should be formatted. The rules in use got
    inferred from an analysis of the PDB. See gist at
    <https://gist.github.com/jbarnoud/37a524330f29b5b7b096> for more
    details.
    """
    if (
        element is not None
        and (element.strip().capitalize() in _position_12_elements)
        and (name[0:2].upper() == element[0:2].upper())
    ):
        return "{:<4}".format(name)
    if len(name) >= 4:
        return name[:4]
    elif len(name) == 1:
        return " {}  ".format(name)
    elif (
        resname == name
        or name[:2] in _ions_metals
        or name == "UNK"
        or (resname in _special_hg and name[:2] == "HG")
        or (resname in _special_cl and name[:2] == "CL")
        or _Pair(resname, name) in _include_pairs
    ) and _Pair(resname, name) not in _exclude_pairs:
        return "{:<4}".format(name)
    return " {:<3}".format(name)


def _getPDBElement(name, element, lowersecond=True):
    """
    Given a PDB atom name of 4 characters (including spaces), get the element
    """
    import re

    regH_old = re.compile("H.[123][123]")  # Matches i.e. HE13
    regH_inv = re.compile("[123]H")  # Matches i.e. 2H
    element_backup = element.strip()
    if not element.isalpha():
        element = name[0:2].strip()
        if element and element[0].isdigit():
            if element_backup:
                element = element_backup
            else:
                element = name[1]
        if element and len(element) > 1 and element[1].isdigit():
            if element_backup:
                element = element_backup
            else:
                element = name[0]
    if element:
        element = element.strip()
    if regH_old.match(name.strip()) or regH_inv.match(name.strip()):
        element = "H"
    if len(element) == 2:
        if lowersecond:
            element = element[0] + element[1].lower()
        else:
            element = element[0] + element[1]

    # Treat special cases like sodium being called SOD but element being Na
    nameelement = (name.strip(), element.upper().strip())
    if nameelement in _rename_elements:
        element = _rename_elements[nameelement]
    return element.capitalize()


def checkTruncations(mol):
    fieldsizes = {
        "record": 6,
        "serial": 5,
        "name": 4,
        "altloc": 1,
        "resname": 4,
        "chain": 1,
        "resid": 4,
        "insertion": 1,
        "segid": 4,
        "element": 2,
    }
    for f in fieldsizes:
        if np.any(
            [
                True if len(x) > fieldsizes[f] else False
                for x in mol.__getattribute__(f).astype("str")
            ]
        ):
            if fieldsizes[f] == 1:
                logger.warning(
                    f'Field "{f}" of PDB overflows. Your data will be truncated to 1 character.'
                )
            else:
                logger.warning(
                    f'Field "{f}" of PDB overflows. Your data will be truncated to {fieldsizes[f]} characters.'
                )


def PDBQTwrite(mol, filename, frames=None, writebonds=True):
    PDBwrite(mol, filename, frames, writebonds, mode="pdbqt")


def PDBwrite(mol, filename, frames=None, writebonds=True, mode="pdb"):
    if frames is None and mol.numFrames != 0:
        frames = mol.frame
    else:
        frames = 0
    frames = ensurelist(frames)

    checkTruncations(mol)
    box = None
    if mol.numFrames != 0:
        coords = np.atleast_3d(mol.coords[:, :, frames])
        if hasattr(mol, "box") and mol.box.shape[1] != 0:
            box = mol.box[:, frames[0]]
    else:  # If Molecule only contains topology, PDB requires some coordinates so give it zeros
        coords = np.zeros((mol.numAtoms, 3, 1), dtype=np.float32)

    numFrames = coords.shape[2]
    nAtoms = coords.shape[0]

    serial = np.arange(1, np.size(coords, 0) + 1).astype(object)
    serial[serial > 99999] = "*****"
    serial = serial.astype("U5")

    if nAtoms > 0:
        if coords.max() >= 1e8 or coords.min() <= -1e7:
            raise RuntimeError(
                "Cannot write PDB coordinates with values smaller than -1E7 or larger than 1E8"
            )
        if mol.occupancy.max() >= 1e6 or mol.occupancy.min() <= -1e5:
            raise RuntimeError(
                "Cannot write PDB occupancy with values smaller than -1E5 or larger than 1E6"
            )
        if mol.beta.max() >= 1e6 or mol.beta.min() <= -1e5:
            raise RuntimeError(
                "Cannot write PDB beta/temperature with values smaller than -1E5 or larger than 1E6"
            )

    fh = open(filename, "w", encoding="ascii")

    if box is not None and not np.all(mol.box == 0):
        fh.write(
            "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1 \n"
            % (box[0], box[1], box[2], 90, 90, 90)
        )

    for f in range(numFrames):
        fh.write("MODEL    %5d\n" % (frames[f] + 1))
        for i in range(0, len(mol.record)):
            name = _deduce_PDB_atom_name(mol.name[i], mol.resname[i], mol.element[i])

            if mode == "pdb":
                linefmt = "{!s:6.6}{!s:>5.5} {}{!s:>1.1}{!s:4.4}{!s:>1.1}{!s:>4.4}{!s:>1.1}   {}{}{}{}{}      {!s:4.4}{!s:>2.2}  \n"
                prelast = mol.segid[i]
                last = mol.element[i].upper()
            elif mode == "pdbqt":
                linefmt = "{!s:6.6}{!s:>5.5} {}{!s:>1.1}{!s:4.4}{!s:>1.1}{!s:>4.4}{!s:>1.1}   {}{}{}{}{}    {:>6.3f} {!s:<2.2}  \n"
                prelast = mol.charge[i]
                last = mol.atomtype[i]
            else:
                raise AssertionError("Invalid mode for PDB writer")

            fh.write(
                linefmt.format(
                    mol.record[i],
                    serial[i],
                    name,
                    mol.altloc[i],
                    mol.resname[i],
                    mol.chain[i],
                    mol.resid[i],
                    mol.insertion[i],
                    "{:8.3f}".format(coords[i, 0, f])[:8],
                    "{:8.3f}".format(coords[i, 1, f])[:8],
                    "{:8.3f}".format(coords[i, 2, f])[:8],
                    "{:6.2f}".format(mol.occupancy[i])[:6],
                    "{:6.2f}".format(mol.beta[i])[:6],
                    prelast,
                    last,
                )
            )
            # TODO : convert charges to ints if we ever write them
            if i < len(mol.record) - 1 and mol.segid[i] != mol.segid[i + 1]:
                fh.write("TER\n")

        if writebonds and mol.bonds is not None and len(mol.bonds) != 0:
            goodbonds = mol.bonds[
                np.all(mol.bonds < 99998, axis=1), :
            ]  # Bonds over 99999 cause issues with PDB fixedwidth format
            bondgraph = nx.Graph()
            bondgraph.add_edges_from(goodbonds + 1)  # Add 1 for PDB 1-based indexing
            for atom, neighbours in sorted(bondgraph.adj.items()):
                neighbours = sorted(list(neighbours))
                for ni in range(0, len(neighbours), 4):
                    subneighs = neighbours[ni : min(ni + 4, len(neighbours))]
                    neighstring = "".join("%5d" % sn for sn in subneighs)
                    fh.write("CONECT{:5d}{}\n".format(atom, neighstring))

        fh.write("ENDMDL\n")
    fh.write("END\n")

    fh.close()


def XTCwrite(mol, filename):
    coords = mol.coords

    nframes = mol.numFrames

    box = np.zeros((3, nframes), dtype=np.float32)
    if hasattr(mol, "box"):
        box = mol.box
    time = np.zeros(nframes)
    if hasattr(mol, "time"):
        time = mol.time
    step = np.zeros(nframes)
    if hasattr(mol, "step"):
        step = mol.step

    if np.size(box, 1) != nframes:  # Box should have as many frames as trajectory
        box = np.tile(box, (1, nframes))
    if np.size(time) != nframes:
        time = np.zeros(nframes)
    if np.size(step) != nframes:
        step = np.zeros(nframes, dtype=int)

    if os.path.isfile(filename):
        os.unlink(filename)

    box = box.astype(np.float32) * 0.1
    step = step.astype(np.int32)
    time = time.astype(np.float32) / 1e3  # Convert from fs to ps
    coords = coords.astype(np.float32) * 0.1  # Convert from A to nm
    if not box.flags["C_CONTIGUOUS"]:
        box = np.ascontiguousarray(box)
    if not step.flags["C_CONTIGUOUS"]:
        step = np.ascontiguousarray(step)
    if not time.flags["C_CONTIGUOUS"]:
        time = np.ascontiguousarray(time)
    if not coords.flags["C_CONTIGUOUS"]:
        coords = np.ascontiguousarray(coords)

    lib = xtc_lib()
    natoms = ct.c_int(coords.shape[0])
    nframes = ct.c_int(coords.shape[2])

    cstep = step.ctypes.data_as(ct.POINTER(ct.c_int))
    ctime = time.ctypes.data_as(ct.POINTER(ct.c_float))
    cbox = box.ctypes.data_as(ct.POINTER(ct.c_float))
    ccoords = coords.ctypes.data_as(ct.POINTER(ct.c_float))
    lib["libxtc"].xtc_write(
        ct.c_char_p(filename.encode("ascii")),
        natoms,
        nframes,
        cstep,
        ctime,
        ccoords,
        cbox,
    )


def BINCOORwrite(mol, filename):
    import struct

    natoms = np.array([mol.numAtoms])
    f = open(filename, "wb")

    dat = mol.coords[:, :, mol.frame].copy()
    dat = dat.reshape(dat.shape[0] * 3).astype(np.float64)

    fmt1 = "i" * natoms.shape[0]
    bin1 = struct.pack(fmt1, *natoms)
    fmt2 = "d" * dat.shape[0]
    bin2 = struct.pack(fmt2, *dat)
    f.write(bin1)
    f.write(bin2)
    f.close()


def PSFwrite(m, filename, explicitbonds=None):
    import string

    segments = np.array(
        [segid if segid != "" else chain for segid, chain in zip(m.segid, m.chain)]
    )
    used_segids = set(segments)
    # Letters to be used for default segids, if free: 0123456789abcd...ABCD..., minus chain symbols already used
    segid_alphabet = list(string.digits + string.ascii_letters)
    available_segids = np.setdiff1d(segid_alphabet, used_segids)
    segments[segments == ""] = available_segids[0]

    fs = {
        "serial": max(len(str(np.max(m.serial))), 10),
        "segid": max(len(max(segments, key=len)), 8),
        "resid": max(len(str(np.max(m.resid))) + len(max(m.insertion, key=len)), 8),
        "resname": max(len(max(m.resname, key=len)), 8),
        "name": max(len(max(m.name, key=len)), 8),
        "atomtype": max(len(max(m.atomtype, key=len)), 7),
    }

    f = open(filename, "w", encoding="ascii")
    print(
        "PSF NAMD\n", file=f
    )  # Write NAMD in the header so that VMD will read it as space delimited format instead of FORTRAN
    print("%8d !NTITLE\n" % (0), file=f)
    print("%8d !NATOM" % (len(m.serial)), file=f)
    for i in range(len(m.serial)):
        atomtype = (
            m.atomtype[i] if m.atomtype[i] != "" else "NULL"
        )  # Write default atomtype otherwise the PSF reader will fail to read correctly the PSF

        segid = available_segids[0]
        if m.segid[i] != "":
            segid = m.segid[i]
        elif m.segid[i] == "" and m.chain[i] != "":
            segid = m.chain[i]

        string_format = (
            f"{m.serial[i]:>{fs['serial']}} "
            f"{segments[i]:<{fs['segid']}} "
            f"{str(m.resid[i]) + str(m.insertion[i]):<{fs['resid']}} "
            f"{m.resname[i]:<{fs['resname']}} "
            f"{m.name[i]:<{fs['name']}} "
            f"{atomtype:<{fs['atomtype']}} "
            f"{m.charge[i]:>9.6f} "
            f"{m.masses[i]:>13.4f} "
            f"{0:>11} "
        )
        print(string_format, file=f)

    if explicitbonds is not None:
        bonds = explicitbonds
    else:
        bonds = m.bonds

    fieldlen = max(len(str(np.max(bonds))), 10) if bonds.shape[0] != 0 else 10
    print("\n\n", file=f)
    print(f"{bonds.shape[0]:{fieldlen}d} !NBOND: bonds", file=f)
    for i in range(bonds.shape[0]):
        if i and not (i % 4):
            print("", file=f)
        vals = bonds[i] + 1
        print(f"{vals[0]:{fieldlen}d}{vals[1]:{fieldlen}d}", file=f, end="")

    if hasattr(m, "angles"):
        fieldlen = max(len(str(np.max(m.angles))), 10) if m.angles.shape[0] != 0 else 10
        print("\n\n", file=f)
        print(f"{m.angles.shape[0]:{fieldlen}d} !NTHETA: angles", file=f)
        for i in range(m.angles.shape[0]):
            if i and not (i % 3):
                print("", file=f)
            vals = m.angles[i] + 1
            print(
                f"{vals[0]:{fieldlen}d}{vals[1]:{fieldlen}d}{vals[2]:{fieldlen}d}",
                file=f,
                end="",
            )

    if hasattr(m, "dihedrals"):
        fieldlen = (
            max(len(str(np.max(m.dihedrals))), 10) if m.dihedrals.shape[0] != 0 else 10
        )
        print("\n\n", file=f)
        print(f"{m.dihedrals.shape[0]:{fieldlen}d} !NPHI: dihedrals", file=f)
        for i in range(m.dihedrals.shape[0]):
            if i and not (i % 2):
                print("", file=f)
            vals = m.dihedrals[i] + 1
            print(
                f"{vals[0]:{fieldlen}d}{vals[1]:{fieldlen}d}{vals[2]:{fieldlen}d}{vals[3]:{fieldlen}d}",
                file=f,
                end="",
            )

    if hasattr(m, "impropers"):
        fieldlen = (
            max(len(str(np.max(m.impropers))), 10) if m.impropers.shape[0] != 0 else 10
        )
        print("\n\n", file=f)
        print(f"{m.impropers.shape[0]:{fieldlen}d} !NIMPHI: impropers", file=f)
        for i in range(m.impropers.shape[0]):
            if i and not (i % 2):
                print("", file=f)
            vals = m.impropers[i] + 1
            print(
                f"{vals[0]:{fieldlen}d}{vals[1]:{fieldlen}d}{vals[2]:{fieldlen}d}{vals[3]:{fieldlen}d}",
                file=f,
                end="",
            )

    print("\n\n", file=f)
    print("%10d !NDON: donors\n" % (0), file=f)
    print("%10d !NACC: acceptors\n" % (0), file=f)
    # According ParmEd, CHARMM PSF has to have an extra blank line after NNB
    # https: // github.com / ParmEd / ParmEd / blob / master / parmed / charmm / psf.py#L151
    print("%10d !NNB\n\n" % (0), file=f)
    print("%10d %10d !NGRP\n" % (0, 0), file=f)
    f.close()


def XYZwrite(src, filename):
    import re

    fh = open(filename, "w", encoding="ascii")
    natoms = len(src.record)
    print("%d\n" % (natoms), file=fh)
    for i in range(natoms):
        e = src.element[i].strip()
        if not len(e):
            e = re.sub("[1234567890]*", "", src.name[i])
        print(
            "%s   %f   %f    %f"
            % (
                e,
                src.coords[i, 0, src.frame],
                src.coords[i, 1, src.frame],
                src.coords[i, 2, src.frame],
            ),
            file=fh,
        )
    fh.close()


def XSCwrite(mol, filename, frames=None):
    if frames is None and mol.numFrames != 0:
        frames = mol.frame
    else:
        frames = 0
    box = mol.box[:, frames]
    step = mol.step[frames]
    with open(filename, "w", encoding="ascii") as f:
        f.write(
            "# NAMD extended system configuration restart file generated by MoleculeKit\n"
        )
        f.write("#$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z\n")
        f.write(
            f"{step} {box[0]:e} {0:e} {0:e} {0:e} {box[1]:e} {0:e} {0:e} {0:e} {box[2]:e} {0:e} {0:e} {0:e} {0:e} {0:e} {0:e} {0:e} {0:e} {0:e}"
        )


def MOL2write(mol, filename):
    uqresname = np.unique(mol.resname)
    if len(uqresname) > 1:
        raise RuntimeError(
            f"MOL2 file can only be written for a single residue. We detected {len(uqresname)} resnames in the Molecule."
        )
    if len(uqresname[0]) == 0:
        raise RuntimeError(
            "MOL2 file can only be written if a resname is defined for the Molecule. Currently the "
            "resname is empty."
        )

    with open(filename, "w", encoding="ascii") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"    {uqresname[0]}\n")
        unique_bonds = [
            list(t) for t in set(map(tuple, [sorted(x) for x in mol.bonds]))
        ]
        unique_bonds = np.array(sorted(unique_bonds, key=lambda x: (x[0], x[1])))
        f.write(
            "%5d %5d %5d %5d %5d\n" % (mol.numAtoms, unique_bonds.shape[0], 0, 0, 0)
        )
        f.write("SMALL\nUSER_CHARGES\n\n\n")
        """
        @<TRIPOS>ATOM
        Each data record associated with this RTI consists of a single data line. This
        data line contains all the information necessary to reconstruct one atom
        contained within the molecule. The atom ID numbers associated with the atoms
        in the molecule will be assigned sequentially when the .mol2 file is read into
        SYBYL.
        Format:
        atom_id atom_name x y z atom_type [subst_id [subst_name [charge [status_bit]]]]

        • atom_id (integer) = the ID number of the atom at the time the file was
        created. This is provided for reference only and is not used when the
        .mol2 file is read into SYBYL.
        • atom_name (string) = the name of the atom.
        • x (real) = the x coordinate of the atom.
        • y (real) = the y coordinate of the atom.
        • z (real) = the z coordinate of the atom.
        • atom_type (string) = the SYBYL atom type for the atom.
        • subst_id (integer) = the ID number of the substructure containing the
        atom.
        • subst_name (string) = the name of the substructure containing the atom.
        • charge (real) = the charge associated with the atom.
        • status_bit (string) = the internal SYBYL status bits associated with the
        atom. These should never be set by the user. Valid status bits are
        DSPMOD, TYPECOL, CAP, BACKBONE, DICT, ESSENTIAL, WATER and
        DIRECT.
        """
        f.write("@<TRIPOS>ATOM\n")
        for i in range(mol.coords.shape[0]):
            f.write(
                "{:7d} {:8s} {:9.4f} {:9.4f} {:9.4f} {:8s} ".format(
                    i + 1,
                    mol.name[i],
                    mol.coords[i, 0, mol.frame],
                    mol.coords[i, 1, mol.frame],
                    mol.coords[i, 2, mol.frame],
                    mol.atomtype[i] if mol.atomtype[i] != "" else mol.element[i],
                )
            )
            if isinstance(mol.resid[i], numbers.Integral):
                f.write("{:3d} ".format(mol.resid[i]))
                if mol.resname[i] != "":
                    f.write("{:4s} ".format(mol.resname[i]))
                    if isinstance(mol.charge[i], numbers.Real):
                        f.write("{:12.6f}".format(mol.charge[i]))
            f.write("\n")
        f.write("@<TRIPOS>BOND\n")
        for i in range(unique_bonds.shape[0]):
            bt = "un"
            if len(mol.bondtype) > 0:
                idx = (mol.bonds[:, 0] == unique_bonds[i, 0]) & (
                    mol.bonds[:, 1] == unique_bonds[i, 1]
                )
                idx |= (mol.bonds[:, 0] == unique_bonds[i, 1]) & (
                    mol.bonds[:, 1] == unique_bonds[i, 0]
                )
                tmp = np.unique(mol.bondtype[idx])
                assert (
                    len(tmp) == 1
                ), f"There should only exist one bond type for atoms {unique_bonds[i, 0]} {unique_bonds[i, 1]}"
                bt = tmp[0]
            f.write(
                "{:6d} {:4d} {:4d} {}\n".format(
                    i + 1, unique_bonds[i, 0] + 1, unique_bonds[i, 1] + 1, bt
                )
            )


def SDFwrite(mol, filename):
    import datetime

    mol2bonds = {"1": 1, "2": 2, "3": 3, "ar": 4, "4": 4}
    chargemap = {-3: 7, -2: 6, -1: 5, 0: 0, 1: 3, 2: 2, 3: 1}
    with open(filename, "w", encoding="ascii") as fh:
        fh.write(f"{mol.viewname}\n")
        currtime = datetime.datetime.now().strftime("%m%d%y%H%M")
        fh.write(f" -MoleculeKit-{currtime}3D\n")
        fh.write(" Structure written by MoleculeKit.\n")
        fh.write(
            f"{mol.numAtoms:>3}{mol.bonds.shape[0]:>3}  0  0  0  0  0  0  0  0999 V2000\n"
        )

        coor = mol.coords[:, :, mol.frame]
        charges = []
        for i in range(mol.numAtoms):
            element = mol.element[i]
            if element == "":
                element = mol.name[i]

            charge = 0
            if mol.charge[i] != 0 and mol.charge[i] in chargemap:
                charge = chargemap[mol.charge[i]]
                charges.append([i + 1, int(mol.charge[i])])

            fh.write(
                f"{coor[i, 0]:>10.4f}{coor[i, 1]:>10.4f}{coor[i, 2]:>10.4f} {element:<2}  0  0  0  0  0  0  0  0  0  0  0  0\n"
            )

        for i in range(mol.bonds.shape[0]):
            fh.write(
                f"{mol.bonds[i, 0]+1:>3}{mol.bonds[i, 1]+1:>3}  {mol2bonds[mol.bondtype[i]]}  0  0  0  0\n"
            )

        if len(charges):
            for idx in range(0, len(charges), 8):
                curr_charges = range(idx, min(idx + 8, len(charges)))
                fh.write(f"M  CHG{len(curr_charges):>3}")
                for cc in curr_charges:
                    fh.write(f"{charges[cc][0]:>4}{charges[cc][1]:>4}")
                fh.write("\n")
        fh.write("M  END\n")
        fh.write("$$$$")


def GROwrite(mol, filename):
    import pandas as pd
    from collections import OrderedDict

    coor = mol.coords[:, :, mol.frame] / 10  # Convert to nm
    box = mol.box[:, mol.frame] / 10  # Convert to nm
    datadict = OrderedDict(
        [
            ("resid", mol.resid),
            ("resname", mol.resname),
            ("name", mol.name),
            ("serial", mol.serial),
            ("posx", coor[:, 0]),
            ("posy", coor[:, 1]),
            ("posz", coor[:, 2]),
        ]
    )
    a = pd.DataFrame(data=datadict)
    with open(filename, "wb") as fh:
        if mol.fstep is not None:
            fh.write(b"Generated with moleculekit, t= %f\n" % (mol.fstep * 1000))
        else:
            fh.write(b"Generated with moleculekit\n")
        fh.write(b"%5d\n" % mol.numAtoms)
        np.savetxt(fh, a.values, "%5d%-5s%5s%5d%8.3f%8.3f%8.3f")
        fh.write(b"%f %f %f 0 0 0 0 0 0" % (box[0], box[1], box[2]))


# Taken from trajectory.py Trajectory()._savers() method of MDtraj
_MDTRAJ_TOPOLOGY_SAVERS = ("pdb", "pdb.gz", "xyz", "xyz.gz")

_MDTRAJ_TRAJECTORY_SAVERS = (
    "xtc",
    "trr",
    "dcd",
    "h5",
    "binpos",
    "nc",
    "netcdf",
    "ncrst",
    "crd",
    "mdcrd",
    "ncdf",
    "lammpstrj",
    "gro",
    "rst7",
    "tng",
)

_MDTRAJ_SAVERS = _MDTRAJ_TRAJECTORY_SAVERS + _MDTRAJ_TOPOLOGY_SAVERS


def MDTRAJwrite(mol, filename):
    try:
        import mdtraj as md
    except ImportError:
        raise ImportError(
            f"To support extension {os.path.splitext(filename)[1]} please install the `mdtraj` package"
        )

    try:
        from moleculekit.util import tempname

        ext = os.path.splitext(filename)[1][1:]
        if ext == "gz":
            pieces = filename.split(".")
            ext = f"{pieces[-2]}.{pieces[-1]}"

        if ext in _MDTRAJ_TOPOLOGY_SAVERS:
            tmppdb = tempname(suffix=".pdb")
            mol.write(tmppdb)
            traj = md.load(tmppdb)
            os.remove(tmppdb)
        elif ext in _MDTRAJ_TRAJECTORY_SAVERS:
            mol = mol.copy()
            mol.time = mol.time / 1000  # convert fs to ps
            tmppdb = tempname(suffix=".pdb")
            tmpxtc = tempname(suffix=".xtc")
            mol.write(tmppdb)
            mol.write(tmpxtc)
            traj = md.load(tmpxtc, top=tmppdb)
            os.remove(tmppdb)
            os.remove(tmpxtc)
        else:
            raise ValueError(f"Unknown file type for file {filename}")
        # traj.xyz = np.swapaxes(np.swapaxes(self.coords, 1, 2), 0, 1) / 10
        # traj.time = self.time
        # traj.unitcell_lengths = self.box.T / 10
        traj.save(filename)
    except Exception as e:
        raise ValueError(f'MDtraj reader failed for file {filename} with error "{e}"')


_WRITERS = {
    "psf": PSFwrite,
    "pdb": PDBwrite,
    "pdbqt": PDBQTwrite,
    "mol2": MOL2write,
    "sdf": SDFwrite,
    "xyz": XYZwrite,
    "gro": GROwrite,
    "coor": BINCOORwrite,
    "xtc": XTCwrite,
    "xsc": XSCwrite,
}


for ext in _MDTRAJ_SAVERS:
    if ext not in _WRITERS:
        _WRITERS[ext] = MDTRAJwrite


import unittest


class _TestWriters(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule, calculateUniqueBonds
        import numpy as np
        import os

        self.testfolder = home(dataDir="molecule-writers")
        mol = Molecule(os.path.join(self.testfolder, "filtered.psf"))
        mol.read(os.path.join(self.testfolder, "filtered.pdb"))
        mol.coords = np.tile(mol.coords, (1, 1, 2))
        mol.filter("protein and resid 1 to 20")
        mol.boxangles = np.ones((3, 2), dtype=np.float32) * 90
        mol.box = np.ones((3, 2), dtype=np.float32) * 15
        mol.step = np.arange(2)
        mol.time = np.arange(2) * 1e5
        mol.fileloc = [mol.fileloc[0], mol.fileloc[0]]
        mol.bondtype[:] = "1"
        mol.bonds, mol.bondtype = calculateUniqueBonds(mol.bonds, mol.bondtype)
        self.mol = mol

    def test_writers(self):
        from moleculekit.util import tempname
        from moleculekit.home import home
        import filecmp

        # Skip file-comparing binary filetypes
        # TODO: Remove SDF. Currently skipping it due to date in second line
        skipcomparison = (
            "ncrst",
            "rst7",
            "dcd",
            "h5",
            "nc",
            "netcdf",
            "ncdf",
            "tng",
            "pdb.gz",
            "xyz.gz",
        )

        for ext in _WRITERS:
            with self.subTest(extension=ext):
                tmpfile = tempname(suffix="." + ext)
                if ext == "pdbqt":
                    mol = self.mol.copy()
                    mol.atomtype[:] = "NA"
                    mol.write(tmpfile)
                elif ext == "mol2":
                    self.mol.write(tmpfile, sel="resid 1")
                else:
                    self.mol.write(tmpfile)
                if ext in skipcomparison:
                    continue

                reffile = os.path.join(home(dataDir="molecule-writers"), "mol." + ext)

                try:
                    with open(tmpfile, "r") as f:
                        filelines = f.readlines()
                        if ext == "sdf":
                            filelines = filelines[2:]
                except UnicodeDecodeError:
                    print(f"Could not compare file {reffile} due to not being unicode")
                    continue
                with open(reffile, "r") as f:
                    reflines = f.readlines()
                    if ext == "sdf":
                        reflines = reflines[2:]
                print("Testing file", reffile, tmpfile)
                self.assertEqual(
                    filelines, reflines, msg=f"Failed comparison of {reffile} {tmpfile}"
                )

    def test_sdf_writer(self):
        from moleculekit.molecule import Molecule
        from moleculekit.util import tempname
        from moleculekit.home import home
        import filecmp

        reffile = os.path.join(self.testfolder, "mol_bromium_out.sdf")
        mol = Molecule(os.path.join(self.testfolder, "mol_bromium.sdf"))
        tmpfile = tempname(suffix=".sdf")
        mol.write(tmpfile)

        with open(tmpfile, "r") as f:
            filelines = f.readlines()[2:]
        with open(reffile, "r") as f:
            reflines = f.readlines()[2:]

        self.assertEqual(
            filelines, reflines, msg=f"Failed comparison of {reffile} {tmpfile}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
