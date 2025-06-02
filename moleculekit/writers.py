# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import numpy as np
import os
from moleculekit.util import ensurelist, sequenceID
import logging
import numbers

logger = logging.getLogger(__name__)

# fmt: off
first_col_elements = {
    'Se', 'Sr', 'Gd', 'Mn', 'Pt', 'Cs', 'Mg', 'Be', 'Sb', 'Bi', 'Th', 'Yb',
    'Zn', 'Kr', 'Rb', 'Ar', 'Pa', 'Fe', 'Ag', 'In', 'Eu', 'Hg', 'Pr', 'Al',
    'Ru', 'Sm', 'Ba', 'Lu', 'Re', 'Xe', 'Dy', 'Ti', 'As', 'Pu', 'Ir', 'Br',
    'Li', 'Pd', 'Ce', 'Cr', 'Ho', 'La', 'Sn', 'Te', 'Cl', 'Hf', 'Ga', 'Co',
    'Os', 'Tl', 'Au', 'Sc', 'Tb', 'Zr', 'Cu', 'Rh', 'Er', 'Mo', 'Si', 'Ca',
    'Ta', 'Cm', 'Am', 'Na', 'Pb', 'Ni', 'Ne', 'Cd'
}

metals_ions = {
    'FE', 'AS', 'ZN', 'MG', 'MN', 'CO', 'BR', 'CU', 'TA', 'MO', 'AL', 'BE',
    'SE', 'PT', 'EU', 'NI', 'IR', 'RH', 'AU', 'GD', 'RU', 'XE', 'RB', 'LU',
    'GA', 'BA', 'CS', 'PB', 'SM', 'SR', 'YB', 'Y'
}

_rename_elements = {('SOD', 'SO'): 'Na'}
# fmt: on


def _format_large_time(mol, ext):
    step = mol.step
    time = mol.time
    nframes = mol.numFrames

    if len(step) == 1:
        trajfreq = step[0]
    else:
        trajfreq = step[1] - step[0]

    if step[-1] != step[-1].astype(np.uint32):
        logger.warning(
            f"Molecule.step contains values too large to be written to a {ext} file. They will be renumbered starting from 1."
        )
        step = np.arange(1, nframes + 1, dtype=np.uint32)
    else:
        step = step.astype(np.uint32)

    time = time / 1e3  # convert from fs to ps
    if time[-1] != time[-1].astype(np.float32):
        logger.warning(
            f"Molecule.time contains values too large to be written to a {ext} file. They will be renumbered starting from 0."
        )
        if trajfreq == 0:
            raise AssertionError("The trajectory step should not be 0")
        timestep = (mol.fstep / trajfreq) / 1e-6
        time = (mol.time - ((mol.step[0] - trajfreq) * timestep)) / 1e3

    return step, time


def _format_pdb_name(name, resname, element=None):
    name = name[:4]
    first_col = f"{name:<4}"
    second_col = f" {name:<3}"

    element_matched = False
    if element is not None:
        in_elements = element.strip().capitalize() in first_col_elements
        same_name_elem = name[0:2].upper() == element[0:2].upper()
        element_matched = in_elements and same_name_elem

    if name == resname or len(name) == 4 or element_matched or name[:2] in metals_ions:
        return first_col
    return second_col


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
    import networkx as nx

    if frames is None and mol.numFrames != 0:
        frames = mol.frame
    else:
        frames = 0
    frames = ensurelist(frames)

    checkTruncations(mol)
    box = None
    boxangles = None
    if mol.numFrames != 0:
        coords = np.atleast_3d(mol.coords[:, :, frames])
        if hasattr(mol, "box") and mol.box.shape[1] != 0:
            box = mol.box[:, frames[0]]
            boxangles = [90, 90, 90]
        if hasattr(mol, "boxangles") and mol.boxangles.shape[1] != 0:
            boxangles = mol.boxangles[:, frames[0]]
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

    if filename.endswith(".gz"):
        import gzip

        fh = gzip.open(filename, "wt", encoding="ascii")
    else:
        fh = open(filename, "w", encoding="ascii")

    if box is not None and not np.all(box == 0):
        fh.write(
            "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1 \n"
            % (box[0], box[1], box[2], boxangles[0], boxangles[1], boxangles[2])
        )

    if mode == "pdb":
        linefmt = "{!s:6.6}{!s:>5.5} {}{!s:>1.1}{!s:4.4}{!s:>1.1}{!s:>4.4}{!s:>1.1}   {}{}{}{}{}      {!s:4.4}{!s:>2.2}{!s:>2.2}\n"
    elif mode == "pdbqt":
        linefmt = "{!s:6.6}{!s:>5.5} {}{!s:>1.1}{!s:4.4}{!s:>1.1}{!s:>4.4}{!s:>1.1}   {}{}{}{}{}    {:>6.3f} {!s:<2.2}  \n"
    else:
        raise AssertionError("Invalid mode for PDB writer")

    for f in range(numFrames):
        fh.write("MODEL    %5d\n" % (frames[f] + 1))
        for i in range(0, len(mol.record)):
            name = _format_pdb_name(mol.name[i], mol.resname[i], mol.element[i])

            data = [
                mol.record[i],
                serial[i],
                name,
                mol.altloc[i],
                mol.resname[i],
                mol.chain[i],
                str(mol.resid[i])[-4:],
                mol.insertion[i],
                f"{coords[i, 0, f]:8.3f}"[:8],
                f"{coords[i, 1, f]:8.3f}"[:8],
                f"{coords[i, 2, f]:8.3f}"[:8],
                f"{mol.occupancy[i]:6.2f}"[:6],
                f"{mol.beta[i]:6.2f}"[:6],
            ]

            if mode == "pdb":
                formalcharge = ""
                if mol.formalcharge[i] > 0:
                    formalcharge = f"{int(mol.formalcharge[i])}+"
                elif mol.formalcharge[i] < 0:
                    formalcharge = f"{abs(int(mol.formalcharge[i]))}-"
                data += [mol.segid[i], mol.element[i].upper(), formalcharge]
            elif mode == "pdbqt":
                data += [mol.charge[i], mol.atomtype[i]]

            fh.write(linefmt.format(*data))

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
                    fh.write(f"CONECT{atom:5d}{neighstring}\n")

        fh.write("ENDMDL\n")
    fh.write("END\n")

    fh.close()


def XTCwrite(mol, filename):
    from moleculekit.xtc import write_xtc

    coords = mol.coords
    nframes = mol.numFrames

    box = np.zeros((3, 3, nframes), dtype=np.float32)
    if hasattr(mol, "box"):
        box = mol.boxvectors
    time = np.zeros(nframes)
    if hasattr(mol, "time"):
        time = mol.time
    step = np.zeros(nframes)
    if hasattr(mol, "step"):
        step = mol.step

    if np.size(box, 2) != nframes:  # Box should have as many frames as trajectory
        box = np.tile(box, (1, 1, nframes))
    if np.size(time) != nframes:
        time = np.zeros(nframes)
    if np.size(step) != nframes:
        step = np.zeros(nframes, dtype=int)

    if os.path.isfile(filename):
        os.unlink(filename)

    step, time = _format_large_time(mol, "XTC")

    box = box.astype(np.float32) * 0.1
    time = time.astype(np.float32)
    coords = coords.astype(np.float32) * 0.1  # Convert from A to nm
    if not box.flags["C_CONTIGUOUS"]:
        box = np.ascontiguousarray(box)
    if not step.flags["C_CONTIGUOUS"]:
        step = np.ascontiguousarray(step)
    if not time.flags["C_CONTIGUOUS"]:
        time = np.ascontiguousarray(time)
    if not coords.flags["C_CONTIGUOUS"]:
        coords = np.ascontiguousarray(coords)

    write_xtc(filename.encode("UTF-8"), coords, box, time, step)


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


def PSFwrite(mol, filename, explicitbonds=None):
    import string
    from moleculekit.periodictable import periodictable

    segments = np.array(
        [segid if segid != "" else chain for segid, chain in zip(mol.segid, mol.chain)]
    )
    used_segids = set(segments)
    # Letters to be used for default segids, if free: 0123456789abcd...ABCD..., minus chain symbols already used
    segid_alphabet = list(string.digits + string.ascii_letters)
    available_segids = np.setdiff1d(segid_alphabet, used_segids)
    segments[segments == ""] = available_segids[0]

    fs = {
        "serial": max(len(str(np.max(mol.serial))), 10),
        "segid": max(len(max(segments, key=len)), 8),
        "resid": max(len(str(np.max(mol.resid))) + len(max(mol.insertion, key=len)), 8),
        "resname": max(len(max(mol.resname, key=len)), 8),
        "name": max(len(max(mol.name, key=len)), 8),
        "atomtype": max(len(max(mol.atomtype, key=len)), 7),
    }

    f = open(filename, "w", encoding="ascii")
    print(
        "PSF NAMD\n", file=f
    )  # Write NAMD in the header so that VMD will read it as space delimited format instead of FORTRAN
    print("%8d !NTITLE\n" % (0), file=f)
    print("%8d !NATOM" % (len(mol.serial)), file=f)
    for i in range(len(mol.serial)):
        # Defaults. PSF readers will fail if any are empty since it's a space delimited format
        resname = mol.resname[i] if mol.resname[i] != "" else "MOL"
        atomtype = mol.atomtype[i] if mol.atomtype[i] != "" else "NULL"
        # If mass is not defined take it from the element
        mass = mol.masses[i]
        if mass == 0 and len(mol.element[i]) and not mol.virtualsite[i]:
            mass = periodictable[mol.element[i]].mass

        string_format = (
            f"{mol.serial[i]:>{fs['serial']}} "
            f"{segments[i]:<{fs['segid']}} "
            f"{str(mol.resid[i]) + str(mol.insertion[i]):<{fs['resid']}} "
            f"{resname:<{fs['resname']}} "
            f"{mol.name[i]:<{fs['name']}} "
            f"{atomtype:<{fs['atomtype']}} "
            f"{mol.charge[i]:>9.6f} "
            f"{mass:>13.4f} "
            f"{0:>11} "
        )
        print(string_format, file=f)

    if explicitbonds is not None:
        bonds, _ = explicitbonds
    else:
        bonds = mol.bonds

    fieldlen = max(len(str(np.max(bonds))), 10) if bonds.shape[0] != 0 else 10
    print("\n\n", file=f)
    print(f"{bonds.shape[0]:{fieldlen}d} !NBOND: bonds", file=f)
    for i in range(bonds.shape[0]):
        if i and not (i % 4):
            print("", file=f)
        vals = bonds[i] + 1
        print(f"{vals[0]:{fieldlen}d}{vals[1]:{fieldlen}d}", file=f, end="")

    if hasattr(mol, "angles"):
        fieldlen = (
            max(len(str(np.max(mol.angles))), 10) if mol.angles.shape[0] != 0 else 10
        )
        print("\n\n", file=f)
        print(f"{mol.angles.shape[0]:{fieldlen}d} !NTHETA: angles", file=f)
        for i in range(mol.angles.shape[0]):
            if i and not (i % 3):
                print("", file=f)
            vals = mol.angles[i] + 1
            print(
                f"{vals[0]:{fieldlen}d}{vals[1]:{fieldlen}d}{vals[2]:{fieldlen}d}",
                file=f,
                end="",
            )

    if hasattr(mol, "dihedrals"):
        fieldlen = (
            max(len(str(np.max(mol.dihedrals))), 10)
            if mol.dihedrals.shape[0] != 0
            else 10
        )
        print("\n\n", file=f)
        print(f"{mol.dihedrals.shape[0]:{fieldlen}d} !NPHI: dihedrals", file=f)
        for i in range(mol.dihedrals.shape[0]):
            if i and not (i % 2):
                print("", file=f)
            vals = mol.dihedrals[i] + 1
            print(
                f"{vals[0]:{fieldlen}d}{vals[1]:{fieldlen}d}{vals[2]:{fieldlen}d}{vals[3]:{fieldlen}d}",
                file=f,
                end="",
            )

    if hasattr(mol, "impropers"):
        fieldlen = (
            max(len(str(np.max(mol.impropers))), 10)
            if mol.impropers.shape[0] != 0
            else 10
        )
        print("\n\n", file=f)
        print(f"{mol.impropers.shape[0]:{fieldlen}d} !NIMPHI: impropers", file=f)
        for i in range(mol.impropers.shape[0]):
            if i and not (i % 2):
                print("", file=f)
            vals = mol.impropers[i] + 1
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


def XSCwrite(mol, filename, frames=None):
    if frames is None and mol.numFrames != 0:
        frame = mol.frame
    else:
        frame = 0
    boxvectors = mol.boxvectors[:, :, frame]
    step = mol.step[frame]
    with open(filename, "w", encoding="ascii") as f:
        f.write(
            "# NAMD extended system configuration restart file generated by MoleculeKit\n"
        )
        f.write("#$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z\n")
        bvx = " ".join([f"{x:e}" for x in boxvectors[0]])
        bvy = " ".join([f"{x:e}" for x in boxvectors[1]])
        bvz = " ".join([f"{x:e}" for x in boxvectors[2]])
        f.write(
            f"{step} {bvx} {bvy} {bvz} {0:e} {0:e} {0:e} {0:e} {0:e} {0:e} {0:e} {0:e} {0:e}"
        )


def _uniquify_atomnames(orig_names, resnames):
    # Guarantee unique atom names for each unique residue
    import re

    new_names = orig_names.copy()
    for resn in np.unique(resnames):
        idx = np.where(resnames == resn)[0]
        for i in idx:
            name = new_names[i]
            while np.sum(name == new_names[idx]) > 1:  # Check for identical names
                # Get the second identical name index
                j = np.flatnonzero(name == new_names[idx])[1]
                prefix, sufix = re.match(r"(.*?\D*)(\d*)$", new_names[idx[j]]).groups()
                sufix = 1 if sufix == "" else int(sufix)
                while prefix + str(sufix) in new_names[idx]:  # Search for a unique name
                    sufix += 1
                new_names[idx[j]] = prefix + str(sufix)
    return new_names


def MOL2write(mol, filename, explicitbonds=None):
    uqresname = np.unique(mol.resname)
    if len(uqresname) == 1 and uqresname[0] != "":
        molname = uqresname[0]
    elif len(uqresname[0]) == 0:
        molname = "MOL"
    else:
        molname = mol.viewname

    bonds = mol.bonds
    if explicitbonds is not None:
        bonds = explicitbonds

    with open(filename, "w", encoding="ascii") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"    {molname}\n")
        unique_bonds = [list(t) for t in set(map(tuple, [sorted(x) for x in bonds]))]
        unique_bonds = np.array(sorted(unique_bonds, key=lambda x: (x[0], x[1])))
        f.write(
            "%5d %5d %5d %5d %5d\n" % (mol.numAtoms, unique_bonds.shape[0], 0, 0, 0)
        )
        f.write("SMALL\nUSER_CHARGES\n\n")
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
        # Guarantee unique residues
        if mol.numResidues == 1:
            resn = mol.resname[0] if mol.resname[0] != "" else "MOL"
            resnames = [resn] * mol.numAtoms
        else:
            resnames = []
            for i in range(mol.numAtoms):
                resn = mol.resname[i] if mol.resname[i] != "" else "MOL"
                resnames.append(
                    f"{resn}{mol.resid[i]:<d}{mol.insertion[i]}{mol.chain[i]}"
                )

        # Guarantee unique atom names for each unique residue
        atomnames = _uniquify_atomnames(mol.name, mol.resname)

        f.write("@<TRIPOS>ATOM\n")
        for i in range(mol.coords.shape[0]):
            f.write(
                "{:7d} {:8s} {:9.4f} {:9.4f} {:9.4f} {:8s} ".format(
                    i + 1,
                    atomnames[i],
                    mol.coords[i, 0, mol.frame],
                    mol.coords[i, 1, mol.frame],
                    mol.coords[i, 2, mol.frame],
                    mol.atomtype[i] if mol.atomtype[i] != "" else mol.element[i],
                )
            )
            if isinstance(mol.resid[i], numbers.Integral):
                f.write(f"{mol.resid[i]} {resnames[i]} ")
                if isinstance(mol.charge[i], numbers.Real):
                    f.write(f"{mol.charge[i]:12.4f}")
            f.write("\n")

        # # Disabled because RDKit has issues with this section
        # if np.any(mol.formalcharge != 0):
        #     f.write("@<TRIPOS>UNITY_ATOM_ATTR\n")
        #     idx = np.where(mol.formalcharge != 0)[0]
        #     for i in idx:
        #         f.write(f"{i+1} 1\ncharge {mol.formalcharge[i]}\n")

        f.write("@<TRIPOS>BOND\n")
        for i in range(unique_bonds.shape[0]):
            bt = "un"
            if len(mol.bondtype) == bonds.shape[0]:
                idx = (bonds[:, 0] == unique_bonds[i, 0]) & (
                    bonds[:, 1] == unique_bonds[i, 1]
                )
                idx |= (bonds[:, 0] == unique_bonds[i, 1]) & (
                    bonds[:, 1] == unique_bonds[i, 0]
                )
                tmp = np.unique(mol.bondtype[idx])
                assert (
                    len(tmp) == 1
                ), f"There should only exist one bond type for atoms {unique_bonds[i, 0]} {unique_bonds[i, 1]}"
                bt = tmp[0]
            f.write(
                "{:6d} {:5d} {:5d} {:>4s}\n".format(
                    i + 1, unique_bonds[i, 0] + 1, unique_bonds[i, 1] + 1, bt
                )
            )


def SDFwrite(mol, filename):
    import datetime

    if any(mol.bondtype == "un"):
        raise RuntimeError(
            "Cannot write 'un' bond types to SDF. Please specify the molecule bond orders."
        )

    bondmap = {"1": 1, "2": 2, "3": 3, "ar": 4, "4": "un"}
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

            if mol.formalcharge[i] != 0:
                charges.append([i + 1, int(mol.formalcharge[i])])

            fh.write(
                f"{coor[i, 0]:>10.4f}{coor[i, 1]:>10.4f}{coor[i, 2]:>10.4f} {element:<2}  0  0  0  0  0  0  0  0  0  0  0  0\n"
            )

        for i in range(mol.bonds.shape[0]):
            fh.write(
                f"{mol.bonds[i, 0] + 1:>3}{mol.bonds[i, 1] + 1:>3}  {bondmap[mol.bondtype[i]]}  0  0  0  0\n"
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


def DCDwrite(mol, filename):
    from moleculekit.dcd import DCDTrajectoryFile

    xyz = np.transpose(mol.coords, (2, 0, 1))
    n_frames = xyz.shape[0]

    if not np.all(mol.box == 0):
        box = mol.box.T
    else:
        box = np.zeros((n_frames, 3), dtype=np.float32)

    if not np.all(mol.boxangles == 0):
        boxangles = mol.boxangles.T
    else:
        boxangles = np.zeros((n_frames, 3), dtype=np.float32)

    try:
        istart = int(mol.step[0])
        nsavc = int(mol.step[1] - mol.step[0])
        fstep = mol.fstep * 1000  # ns to ps
        delta = fstep / nsavc / 0.04888821  # Conversion factor found in OpenMM
        if mol.step[0] != mol.step[0].astype(np.int32):
            logger.warning(
                "Molecule.step contains values too large to be written to DCD file. They will be renumbered starting from 1."
            )
            istart //= nsavc
            delta *= nsavc
            nsavc = 1
        # If it's still too large just start from 1
        if istart != np.array(istart).astype(np.int32):
            istart = 1
    except Exception:
        istart = 0
        nsavc = 1
        delta = 1.0

    with DCDTrajectoryFile(filename, "w") as fh:
        fh.write(
            xyz,
            cell_lengths=box,
            cell_angles=boxangles,
            istart=istart,
            nsavc=nsavc,
            delta=delta,
        )


def BINPOSwrite(mol, filename):
    from moleculekit.binpos import BINPOSTrajectoryFile

    xyz = np.transpose(mol.coords, (2, 0, 1))
    with BINPOSTrajectoryFile(filename, "w") as fh:
        fh.write(xyz)


def TRRwrite(mol, filename):
    from moleculekit.trr import TRRTrajectoryFile

    xyz = np.transpose(mol.coords, (2, 0, 1)) / 10  # Convert Angstrom to nm
    step, time = _format_large_time(mol, "TRR")
    boxvectors = np.transpose(mol.boxvectors, (2, 0, 1)) / 10  # Angstrom to nm
    with TRRTrajectoryFile(filename, "w") as fh:
        fh.write(xyz, time=time, step=step, box=boxvectors, lambd=None)


def NETCDFwrite(mol, filename):
    # Patched together from MDTraj
    from moleculekit.fileformats.netcdf import netcdf_file
    from moleculekit.fileformats.utils import ensure_type
    from moleculekit import __version__
    from datetime import datetime
    import socket

    # typecheck all of the input arguments rigorously
    coordinates = ensure_type(
        np.transpose(mol.coords, (2, 0, 1)),
        np.float32,
        3,
        "coordinates",
        length=None,
        can_be_none=False,
        shape=(None, None, 3),
        warn_on_cast=False,
        add_newaxis_on_deficient_ndim=True,
    )
    n_frames, n_atoms = coordinates.shape[0], coordinates.shape[1]

    step, time = _format_large_time(mol, "NETCDF")

    time = ensure_type(
        time,  # In ps
        np.float32,
        1,
        "time",
        length=n_frames,
        can_be_none=True,
        warn_on_cast=False,
        add_newaxis_on_deficient_ndim=True,
    )
    step = ensure_type(
        step,
        np.int32,
        1,
        "step",
        length=n_frames,
        can_be_none=True,
        warn_on_cast=False,
        add_newaxis_on_deficient_ndim=True,
    )
    cell_lengths = ensure_type(
        mol.box.T,
        np.float64,
        2,
        "cell_lengths",
        length=n_frames,
        can_be_none=True,
        shape=(n_frames, 3),
        warn_on_cast=False,
        add_newaxis_on_deficient_ndim=True,
    )
    cell_angles = ensure_type(
        mol.boxangles.T,
        np.float64,
        2,
        "cell_angles",
        length=n_frames,
        can_be_none=True,
        shape=(n_frames, 3),
        warn_on_cast=False,
        add_newaxis_on_deficient_ndim=True,
    )

    # are we dealing with a periodic system?
    if (cell_lengths is None and cell_angles is not None) or (
        cell_lengths is not None and cell_angles is None
    ):
        provided, neglected = "cell_lengths", "cell_angles"
        if cell_lengths is None:
            provided, neglected = neglected, provided
        raise ValueError(
            'You provided the variable "%s", but neglected to '
            'provide "%s". They either BOTH must be provided, or '
            "neither. Having one without the other is meaningless"
            % (provided, neglected)
        )

    set_cell = mol.box is not None and not np.all(mol.box == 0)
    set_time = mol.time is not None and not np.all(mol.time == 0)

    ncfile = netcdf_file(filename, mode="w", version=2)

    # Initialize the headers
    # Set attributes.
    setattr(
        ncfile, "title", "CREATED at %s on %s" % (datetime.now(), socket.gethostname())
    )
    setattr(ncfile, "application", "MoleculeKit")
    setattr(ncfile, "program", "MoleculeKit")
    setattr(ncfile, "programVersion", __version__)
    setattr(ncfile, "Conventions", "AMBER")
    setattr(ncfile, "ConventionVersion", "1.0")

    # set the dimensions
    # unlimited number of frames in trajectory
    ncfile.createDimension("frame", 0)
    # number of spatial coordinates
    ncfile.createDimension("spatial", 3)
    # number of atoms
    ncfile.createDimension("atom", n_atoms)

    if set_cell:
        # three spatial coordinates for the length of the unit cell
        ncfile.createDimension("cell_spatial", 3)
        # three spatial coordinates for the angles that define the shape
        # of the unit cell
        ncfile.createDimension("cell_angular", 3)
        # length of the longest string used for a label
        ncfile.createDimension("label", 5)

        # Define variables to store unit cell data
        _cell_lengths = ncfile.createVariable(
            "cell_lengths", "d", ("frame", "cell_spatial")
        )
        setattr(_cell_lengths, "units", "angstrom")
        _cell_angles = ncfile.createVariable(
            "cell_angles", "d", ("frame", "cell_angular")
        )
        setattr(_cell_angles, "units", "degree")

        ncfile.createVariable("cell_spatial", "c", ("cell_spatial",))
        ncfile.variables["cell_spatial"][0] = "a"
        ncfile.variables["cell_spatial"][1] = "b"
        ncfile.variables["cell_spatial"][2] = "c"

        ncfile.createVariable("cell_angular", "c", ("cell_spatial", "label"))
        ncfile.variables["cell_angular"][0] = "alpha"
        ncfile.variables["cell_angular"][1] = "beta "
        ncfile.variables["cell_angular"][2] = "gamma"

    if set_time:
        # Define coordinates and snapshot times.
        frame_times = ncfile.createVariable("time", "f", ("frame",))
        setattr(frame_times, "units", "picosecond")
        frame_steps = ncfile.createVariable("step", "i", ("frame",))
        setattr(frame_steps, "units", "step")

    frame_coordinates = ncfile.createVariable(
        "coordinates", "f", ("frame", "atom", "spatial")
    )
    setattr(frame_coordinates, "units", "angstrom")

    ncfile.createVariable("spatial", "c", ("spatial",))
    ncfile.variables["spatial"][0] = "x"
    ncfile.variables["spatial"][1] = "y"
    ncfile.variables["spatial"][2] = "z"

    # this slice object says where we're going to put the data in the
    # arrays
    frame_slice = slice(0, mol.numFrames)

    # deposit the data
    try:
        ncfile.variables["coordinates"][frame_slice, :, :] = coordinates
        if time is not None and set_time:
            ncfile.variables["time"][frame_slice] = time
        if step is not None and set_time:
            ncfile.variables["step"][frame_slice] = step
        if cell_lengths is not None:
            ncfile.variables["cell_lengths"][frame_slice, :] = cell_lengths
        if cell_angles is not None:
            ncfile.variables["cell_angles"][frame_slice, :] = cell_angles
    except KeyError as e:
        raise ValueError(
            "The file that you're trying to save to doesn't "
            "contain the field %s." % str(e)
        )

    # check for missing attributes
    missing = None
    if time is None and "time" in ncfile.variables:
        missing = "time"
    elif step is None and "step" in ncfile.variables:
        missing = "step"
    elif cell_angles is None and "cell_angles" in ncfile.variables:
        missing = "cell_angles"
    elif cell_lengths is None and "cell_lengths" in ncfile.variables:
        missing = "cell_lengths"
    if missing is not None:
        raise ValueError(
            "The file that you're saving to expects each frame "
            "to contain %s information, but you did not supply it."
            "I don't allow 'ragged' arrays." % missing
        )

    ncfile.sync()
    ncfile.close()


def XYZwrite(mol, filename):
    from moleculekit import __version__
    from datetime import date
    import gzip

    gz = filename.endswith(".gz")

    with gzip.open(filename, "wt") if gz else open(filename, "w") as f:
        for i in range(mol.numFrames):
            f.write(f"{mol.numAtoms}\n")
            f.write(f"Created with MoleculeKit {__version__} {date.today()}\n")
            for j in range(mol.numAtoms):
                coord = mol.coords[j, :, i]
                f.write(
                    f"{mol.element[j]} {coord[0]:8.5f} {coord[1]:8.5f} {coord[2]:8.5f}\n"
                )


def INPCRDwrite(mol, filename):
    with open(filename, "w", encoding="ascii") as f:
        f.write("Created with MoleculeKit\n")
        f.write(f"  {mol.numAtoms}\n")
        for i in range(mol.numAtoms):
            f.write(
                f"{mol.coords[i, 0, mol.frame]:12.7f}{mol.coords[i, 1, mol.frame]:12.7f}{mol.coords[i, 2, mol.frame]:12.7f}"
            )
            if i % 2 == 1:  # Add a newline every two atoms
                f.write("\n")
        if mol.numAtoms % 2 == 1:  # Put a final newline if the number of atoms is odd
            f.write("\n")
        box = mol.box[:, mol.frame]
        angles = mol.boxangles[:, mol.frame]
        f.write(
            f"{box[0]:12.7f}{box[1]:12.7f}{box[2]:12.7f}{angles[0]:12.7f}{angles[1]:12.7f}{angles[2]:12.7f}"
        )
        f.write("\n")


# Taken from trajectory.py Trajectory()._savers() method of MDtraj

_MDTRAJ_SAVERS = ("h5", "ncrst", "mdcrd", "lammpstrj", "gro", "rst7")


def MDTRAJwrite(mol, filename):
    try:
        import mdtraj as md
    except ImportError:
        raise ImportError(
            f"To support extension {os.path.splitext(filename)[1]} please install the `mdtraj` package"
        )

    from mdtraj.core.trajectory import Trajectory

    try:
        from moleculekit.util import tempname

        ext = os.path.splitext(filename)[1][1:]
        if ext == "gz":
            pieces = filename.split(".")
            ext = f"{pieces[-2]}.{pieces[-1]}"

        tmppdb = tempname(suffix=".pdb")
        mol.write(tmppdb)
        traj = md.load(tmppdb)
        os.remove(tmppdb)

        if ext in _MDTRAJ_SAVERS:
            mol = mol.copy()

            time = np.array([x / 1000 for x in mol.time])  # convert fs to ps
            if time.shape[0] != mol.numFrames and np.all(time == 0):
                # Assign 0 time if not present
                time = np.zeros(mol.numFrames, dtype=np.float32)

            box = mol.box.T / 10  # Ang to nm
            if box.shape[0] != mol.numFrames and np.all(box == 0):
                # Assign 0 box if not present
                box = np.zeros((mol.numFrames, 3), dtype=np.float32)

            boxangles = mol.boxangles.T
            if boxangles.shape[0] != mol.numFrames and np.all(boxangles == 0):
                # Assign 90 degree box angles if not present
                boxangles = np.full_like(box, 90.0, dtype=np.float32)

            traj = Trajectory(
                xyz=np.transpose(mol.coords, (2, 0, 1)) / 10,  # Ang to nm
                topology=traj.topology,
                time=time.astype(np.float32),
                unitcell_lengths=box,
                unitcell_angles=boxangles,
            )
            traj.save(filename)
            return

        raise ValueError(f"Unknown file type for file {filename}")
    except Exception as e:
        raise ValueError(f'MDtraj writer failed for file {filename} with error "{e}"')


def CIFwrite(
    mol,
    filename,
    explicitbonds=None,
    chemcomp=None,
    return_data=False,
    writebonds=True,
    fp_precision=3,
):
    from moleculekit.pdbx.writer.PdbxWriter import PdbxWriter
    from moleculekit.molecule import _originalResname
    import re

    if not return_data:
        from moleculekit.pdbx.reader.PdbxContainers import DataContainer, DataCategory
    else:
        from mmcif.api.DataCategory import DataCategory
        from mmcif.api.PdbxContainers import DataContainer

    if chemcomp is not None:
        single_mol = chemcomp
    else:
        single_mol = len(np.unique(mol.resname)) == 1

    if not len(mol.resname[0]):
        raise RuntimeError("Please specify a resname for your molecule.")

    atom_site_mapping = {
        "group_PDB": "record",
        "id": "serial",
        "type_symbol": "element",
        "label_atom_id": "name",
        "label_alt_id": "altloc",
        "label_comp_id": "resname",
        "label_asym_id": "chain",
        "label_entity_id": "segid",
        "label_seq_id": "resid",
        "pdbx_PDB_ins_code": "insertion",
        "Cartn_x": "coords",
        "Cartn_y": "coords",
        "Cartn_z": "coords",
        "occupancy": "occupancy",
        "B_iso_or_equiv": "beta",
        "pdbx_formal_charge": "formalcharge",
        "auth_seq_id": "resid",
        "auth_comp_id": "resname",
        "auth_asym_id": "chain",
        "auth_atom_id": "name",
        "pdbx_PDB_model_num": "frame",
    }
    chem_comp_mapping = {
        "comp_id": "resname",
        "atom_id": "name",
        "alt_atom_id": "atomtype",
        "type_symbol": "element",
        "charge": "formalcharge",
        "partial_charge": "charge",
        "model_Cartn_x": "coords",
        "model_Cartn_y": "coords",
        "model_Cartn_z": "coords",
        "pdbx_model_Cartn_x_ideal": "coords",
        "pdbx_model_Cartn_y_ideal": "coords",
        "pdbx_model_Cartn_z_ideal": "coords",
    }
    bondtype_map = {
        "1": "SING",
        "2": "DOUB",
        "3": "TRIP",
        "4": "QUAD",
        "ar": "AROM",
        "am": "SING",
        "un": "SING",  # Default to single for unknown bond type
    }
    xyz_map = {
        "Cartn_x": 0,
        "Cartn_y": 1,
        "Cartn_z": 2,
        "model_Cartn_x": 0,
        "model_Cartn_y": 1,
        "model_Cartn_z": 2,
        "pdbx_model_Cartn_x_ideal": 0,
        "pdbx_model_Cartn_y_ideal": 1,
        "pdbx_model_Cartn_z_ideal": 2,
    }

    mapping = atom_site_mapping
    atom_block = "atom_site"

    atomnames = mol.name
    if single_mol:
        if len(np.unique(mol.resid)) != 1:
            raise RuntimeError(
                "CIF files don't support multiple residues with different resid and same resname"
            )
        # Guarantee unique atom names for each unique residue
        atomnames = _uniquify_atomnames(mol.name, mol.resname)
        if len(np.unique(atomnames)) != mol.numAtoms:
            raise RuntimeError(
                "Atom names need to be unique to write small molecule CIF file"
            )
        mapping = chem_comp_mapping
        atom_block = "chem_comp_atom"

    viewname = mol.viewname
    if viewname is None or len(viewname) == 0:
        viewname = "MOL"
    viewname = re.sub(r"\W+", "_", viewname)

    myDataList = []
    with open(filename, "w") as ofh:
        curContainer = DataContainer(mol.resname[0] if single_mol else viewname)
        if atom_block == "chem_comp_atom":
            aCat = DataCategory("chem_comp")
            aCat.appendAttribute("id")
            aCat.appendAttribute("type")
            aCat.appendAttribute("pdbx_formal_charge")
            aCat.append([mol.resname[0], "NON-POLYMER", int(mol.formalcharge.sum())])
            curContainer.append(aCat)

        aCat = DataCategory(atom_block)
        for at in mapping:
            aCat.appendAttribute(at)

        for i in range(mol.numAtoms):
            data = []
            for at in mapping:
                if mapping[at] == "coords":
                    data.append(
                        f"{mol.coords[i, xyz_map[at], mol.frame]:.{fp_precision}f}"
                    )
                elif mapping[at] == "frame":
                    data.append(1)
                elif mapping[at] == "name":
                    data.append(atomnames[i])
                elif mapping[at] == "serial":
                    data.append(i + 1)
                elif at == "label_comp_id":
                    if mol.resname[i] not in _originalResname:
                        data.append(mol.resname[i])
                    else:
                        data.append(_originalResname[mol.resname[i]])
                else:
                    data.append(mol.__dict__[mapping[at]][i])
            aCat.append(data)
        curContainer.append(aCat)

        if len(mol.bonds) and writebonds:
            bonds = mol.bonds
            bondtype = mol.bondtype
            if explicitbonds is not None:
                bonds = explicitbonds[0]
                bondtype = explicitbonds[1]

            uqresid = sequenceID((mol.resid, mol.insertion, mol.chain))

            bCat = DataCategory("chem_comp_bond")
            for at in ["comp_id", "atom_id_1", "atom_id_2", "value_order"]:
                bCat.appendAttribute(at)

            written_bonds = set()
            for i in range(bonds.shape[0]):
                bond = bonds[i]
                if uqresid[bond[0]] != uqresid[bond[1]]:
                    continue
                key = (mol.resname[bond[0]], atomnames[bond[0]], atomnames[bond[1]])
                if key in written_bonds:
                    continue
                written_bonds.add(key)
                bCat.append([*key, bondtype_map[bondtype[i]]])
            curContainer.append(bCat)

            if not single_mol:
                bCat = DataCategory("struct_conn")
                for at in [
                    "conn_type_id",
                    "ptnr1_auth_asym_id",
                    "ptnr1_auth_seq_id",
                    "ptnr1_label_atom_id",
                    "pdbx_ptnr1_PDB_ins_code",
                    "ptnr2_auth_asym_id",
                    "ptnr2_auth_seq_id",
                    "ptnr2_label_atom_id",
                    "pdbx_ptnr2_PDB_ins_code",
                    "pdbx_value_order",
                ]:
                    bCat.appendAttribute(at)
                for i in range(bonds.shape[0]):
                    bond = bonds[i]
                    if uqresid[bond[0]] == uqresid[bond[1]]:
                        continue
                    bCat.append(
                        [
                            "covale",
                            mol.chain[bond[0]],
                            mol.resid[bond[0]],
                            mol.name[bond[0]],
                            mol.insertion[bond[0]],
                            mol.chain[bond[1]],
                            mol.resid[bond[1]],
                            mol.name[bond[1]],
                            mol.insertion[bond[1]],
                            bondtype_map[bondtype[i]],
                        ]
                    )
                curContainer.append(bCat)

        myDataList.append(curContainer)
        if return_data:
            return myDataList
        pdbxW = PdbxWriter(ofh)
        pdbxW.write(myDataList)


mmcif_api = None


def BCIFwrite(mol, filename, explicitbonds=None, chemcomp=None):
    from mmcif.io.BinaryCifWriter import BinaryCifWriter
    from mmcif.api.DictionaryApi import DictionaryApi
    from mmcif.api.PdbxContainers import DataContainer
    from mmcif.api.DataCategoryTyped import DataCategoryTyped

    raise NotImplementedError("BinaryCIF writing is not yet fully implemented")

    global mmcif_api

    if mmcif_api is None:
        from mmcif.io.IoAdapterPy import IoAdapterPy as IoAdapter
        from moleculekit import __share_dir

        myIo = IoAdapter(raiseExceptions=True)
        dict_path = os.path.join(__share_dir, "mmcif", "mmcif_pdbx_v5_next.dic")
        cl = myIo.readFile(inputFilePath=dict_path)
        mmcif_api = DictionaryApi(cl, consolidate=True)

    containerList = CIFwrite(
        mol,
        filename=filename,
        explicitbonds=explicitbonds,
        chemcomp=chemcomp,
        return_data=True,
    )

    bcw = BinaryCifWriter(
        dictionaryApi=mmcif_api,
        storeStringsAsBytes=False,
        defaultStringEncoding="utf-8",
        applyTypes=True,
        useFloat64=False,
    )

    # Convert to typed container
    container = containerList[0]
    cName = container.getName()
    tc = DataContainer(cName)
    for catName in container.getObjNameList():
        dObj = container.getObj(catName)
        tObj = DataCategoryTyped(dObj, dictionaryApi=mmcif_api, copyInputData=True)
        tc.append(tObj)

    # Write to file
    bcw.serialize(filename, [tc])


def MMTFwrite(mol, filename):
    from mmtf import write_mmtf, MMTFDecoder
    from string import ascii_uppercase
    from moleculekit.molecule import _residueNameTable

    bondmap = {"1": 1, "2": 2, "3": 3, "4": 4, "un": 1, "ar": 1}

    class MolToMMTF(MMTFDecoder):
        def __init__(self, mol):
            uqres = sequenceID((mol.resid, mol.insertion, mol.chain))
            protein = mol.atomselect("protein")
            nucleic = mol.atomselect("nucleic")
            water = mol.atomselect("water")
            sequences = mol.sequence()
            insertions = []
            self.group_id_list = []
            chain_count = 0
            chain_id_list = []
            chain_name_list = []
            previous_chain = mol.chain[0]
            previous_segid = mol.segid[0]
            previous_rid = 0
            groups_per_chain = []
            group_list = {}
            residue_type = []
            sequence_index_list = []
            sec_struct_list = []
            entity_list = []
            accounted_bonds = np.zeros(mol.bonds.shape[0], dtype=bool)
            chain_res_count = 0
            for rr in range(uqres.max() + 1):
                mask = uqres == rr
                mask_idx = np.where(mask)[0]
                firstidx = mask_idx[0]
                resname = mol.resname[firstidx]
                insertions.append(
                    mol.insertion[firstidx] if mol.insertion[firstidx] != "" else "\x00"
                )
                self.group_id_list.append(mol.resid[firstidx])
                names = ",".join(mol.name[mask])
                key = (resname, names)
                residue_type.append(key)
                sequence_index_list.append(chain_res_count)
                sec_struct_list.append(-1)
                chain_res_count += 1

                if key not in group_list:
                    mask_bonds = np.isin(mol.bonds, mask_idx).sum(axis=1) == 2
                    bond_orders = [bondmap[x] for x in mol.bondtype[mask_bonds]]
                    group_bonds = mol.bonds[mask_bonds] - mask_idx[0]  # relative idx
                    accounted_bonds[mask_bonds] = True

                    group_list[key] = {
                        "groupName": resname,
                        "atomNameList": mol.name[mask].tolist(),
                        "elementList": mol.element[mask].tolist(),
                        "bondOrderList": bond_orders,
                        "bondAtomList": group_bonds.flatten().tolist(),
                        "formalChargeList": mol.formalcharge[mask].tolist(),
                        "singleLetterCode": _residueNameTable.get(resname, "?"),
                        "chemCompType": "OTHER",
                    }

                # If a chain changes, count residues in chain
                curr_segid = mol.segid[firstidx]
                curr_chain = mol.chain[firstidx]
                if curr_segid != previous_segid:
                    chain_name_list.append(previous_chain)
                    chain_id_list.append(ascii_uppercase[chain_count])
                    groups_per_chain.append(rr - previous_rid)
                    if all(mask & protein) or all(mask & nucleic):
                        entity_list.append(
                            {
                                "description": "polymer",
                                "type": "polymer",
                                "chainIndexList": [chain_count],
                                "sequence": sequences[curr_chain],
                            }
                        )
                    elif all(mask & water):
                        entity_list.append(
                            {
                                "description": "water",
                                "type": "water",
                                "chainIndexList": [chain_count],
                                "sequence": "",
                            }
                        )
                    else:
                        entity_list.append(
                            {
                                "description": "non-polymer",
                                "type": "non-polymer",
                                "chainIndexList": [chain_count],
                                "sequence": "",
                            }
                        )
                    previous_segid = curr_segid
                    previous_chain = curr_chain
                    previous_rid = rr
                    chain_count += 1
                    chain_res_count = 0

            chain_name_list.append(curr_chain)
            chain_id_list.append(ascii_uppercase[chain_count])
            groups_per_chain.append(rr + 1 - previous_rid)

            groups = sorted(list(group_list.keys()))
            self.num_atoms = mol.numAtoms
            self.num_bonds = mol.numBonds
            self.num_groups = len(np.unique(mol.resname))
            self.num_chains = len(np.unique(mol.chain))
            self.num_models = 1
            self.structure_id = mol.viewname
            # initialise the arrays
            self.x_coord_list = mol.coords[:, 0, mol.frame].tolist()
            self.y_coord_list = mol.coords[:, 1, mol.frame].tolist()
            self.z_coord_list = mol.coords[:, 2, mol.frame].tolist()
            self.group_type_list = [groups.index(rt) for rt in residue_type]
            self.entity_list = entity_list
            self.b_factor_list = mol.beta.tolist()
            self.occupancy_list = mol.occupancy.tolist()
            self.atom_id_list = mol.serial.tolist()
            self.alt_loc_list = [x if x != "" else "\x00" for x in mol.altloc]
            self.ins_code_list = insertions
            self.sequence_index_list = sequence_index_list
            self.group_list = [group_list[key] for key in groups]
            self.chain_name_list = chain_name_list
            self.chain_id_list = chain_id_list
            self.bond_atom_list = mol.bonds[~accounted_bonds].flatten().tolist()
            self.bond_order_list = [bondmap[x] for x in mol.bondtype[~accounted_bonds]]
            self.sec_struct_list = sec_struct_list
            self.chains_per_model = [len(groups_per_chain)]
            self.groups_per_chain = groups_per_chain
            self.current_group = None
            self.bio_assembly = []
            self.r_free = None
            self.r_work = None
            self.resolution = 0
            self.title = ""
            self.deposition_date = None
            self.release_date = None
            self.experimental_methods = None
            self.unit_cell = None
            self.space_group = None

    enc = MolToMMTF(mol)
    write_mmtf(filename, enc, MolToMMTF.pass_data_on)
    return enc


def JSONwrite(mol, filename):
    from moleculekit.molecule import Molecule
    from moleculekit import __version__
    import json

    mol_dict = mol.toDict(fields=Molecule._all_fields)
    mol_dict["moleculekit_version"] = __version__

    with open(filename, "w") as f:
        json.dump(mol_dict, f)


_WRITERS = {
    "psf": PSFwrite,
    "pdb": PDBwrite,
    "pdb.gz": PDBwrite,
    "pdbqt": PDBQTwrite,
    "mol2": MOL2write,
    "sdf": SDFwrite,
    "gro": GROwrite,
    "coor": BINCOORwrite,
    "xtc": XTCwrite,
    "xsc": XSCwrite,
    "cif": CIFwrite,
    "mmtf": MMTFwrite,
    "dcd": DCDwrite,
    "netcdf": NETCDFwrite,
    "nc": NETCDFwrite,
    "ncdf": NETCDFwrite,
    "trr": TRRwrite,
    "binpos": BINPOSwrite,
    "xyz": XYZwrite,
    "xyz.gz": XYZwrite,
    # "bcif": BCIFwrite,
    "inpcrd": INPCRDwrite,
    "crd": INPCRDwrite,
    "json": JSONwrite,
}


for ext in _MDTRAJ_SAVERS:
    if ext not in _WRITERS:
        _WRITERS[ext] = MDTRAJwrite
