# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import logging
import unittest
import tempfile
import numpy as np
import pandas as pd
import os
from moleculekit.molecule import Molecule, mol_equal, UniqueResidueID
from moleculekit.tools.autosegment import autoSegment2
from moleculekit.util import sequenceID


logger = logging.getLogger(__name__)


def _fixup_water_names(mol):
    """Rename WAT / OW HW HW atoms as O H1 H2"""
    mol.set("name", "O", sel="resname WAT and name OW")
    mol.set("name", "H1", sel="resname WAT and name HW and index % 2 == 1")
    mol.set("name", "H2", sel="resname WAT and name HW and index % 2 == 0")


def _warn_if_contains_DUM(mol):
    """Warn if any DUM atom is there"""
    if any(mol.atomselect("resname DUM")):
        logger.warning(
            "OPM's DUM residues must be filtered out before preparation. Continuing, but crash likely."
        )


def _check_chain_and_segid(mol, verbose):
    import string

    emptychains = mol.chain == ""
    emptysegids = mol.segid == ""

    if np.all(emptychains) and np.all(emptysegids):
        raise RuntimeError(
            "No chains or segments defined in Molecule.chain / Molecule.segid. Please assign either to continue with preparation."
        )

    if np.all(emptychains) and np.any(~emptysegids):
        logger.info(
            "No chains defined in Molecule. Using segment IDs as chains for protein preparation."
        )
        mol = mol.copy()
        chain_alphabet = np.array(
            list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
        )
        mol.chain[:] = chain_alphabet[sequenceID(mol.segid)]

    if np.any(~emptysegids) and np.any(~emptychains):
        chainseq = sequenceID(mol.chain)
        segidseq = sequenceID(mol.segid)
        if not np.array_equal(chainseq, segidseq):
            logger.warning(
                "Both chains and segments are defined in Molecule.chain / Molecule.segid, however they are inconsistent. "
                "Protein preparation will use the chain information."
            )

    if verbose:
        chainids = np.unique(mol.chain)
        if np.any([len(cc) > 1 for cc in chainids]):
            raise RuntimeError(
                "The chain field should only contain a single character."
            )

        print("\n---- Molecule chain report ----")
        for c in chainids:
            chainatoms = np.where(mol.chain == c)[0]
            firstatom = chainatoms[0]
            lastatom = chainatoms[-1]
            print(f"Chain {c}:")
            print(
                f"    First residue: {_fmt_res(mol.resname[firstatom], mol.resid[firstatom], mol.insertion[firstatom])}"
            )
            print(
                f"    Final residue: {_fmt_res(mol.resname[lastatom], mol.resid[lastatom], mol.insertion[lastatom])}"
            )
        print("---- End of chain report ----\n")

    return mol


def _generate_nonstandard_residues_ff(
    mol,
    definition,
    forcefield,
    _molkit_ff=True,
    outdir=None,
    ignore_ns_errors=False,
    residue_smiles=None,
):
    import tempfile
    from moleculekit.tools.preparation_customres import _get_custom_ff
    from moleculekit.tools.preparation_customres import (
        _process_custom_residue,
        _template_residue_from_smiles,
        _mol_to_dat_def,
        _mol_to_xml_def,
        _prepare_for_parameterize,
    )

    uqprot_resn = np.unique(mol.get("resname", sel="protein"))
    not_in_ff = []
    for resn in uqprot_resn:
        if not forcefield.has_residue(resn):
            not_in_ff.append(resn)

    if len(not_in_ff) == 0:
        return definition, forcefield

    try:
        from aceprep.prepare import rdk_prepare
    except ImportError:
        if ignore_ns_errors:
            return definition, forcefield
        raise RuntimeError(
            "To protonate non-canonical aminoacids you need the aceprep library. Please contact Acellera info@acellera.com for more information or set ignore_ns_errors=True to ignore non-canonical residues in the protonation (this will leave the residues unprotonated)."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        for res in not_in_ff:
            try:
                logger.info(f"Attempting to template non-canonical residue {res}...")
                # This removes the non-canonical hydrogens from the original mol object
                mol.remove((mol.resname == res) & (mol.element == "H"), _logger=False)
                molc = mol.copy()

                # Hacky way of getting the first molecule, if there are copies
                molresn = molc.resname == res
                firstname = molc.name[molresn][0]
                lastname = molc.name[molresn][-1]
                start = np.where(molresn & (molc.name == firstname))[0][0]
                end = np.where(molresn & (molc.name == lastname))[0][0]
                # Remove all other stuff
                molc.filter(f"index {start} to {end}", _logger=False)

                if len(np.unique(molc.name)) != molc.numAtoms:
                    raise RuntimeError(
                        f"Residue {res} contains duplicate atom names. Please rename the atoms to have unique names."
                    )

                smiles = None
                if residue_smiles is not None and res in residue_smiles:
                    smiles = residue_smiles[res]

                tmol = _template_residue_from_smiles(molc, res, smiles=smiles)
                cres = _process_custom_residue(tmol, res)

                _mol_to_xml_def(cres, os.path.join(tmpdir, f"{res}.xml"))
                _mol_to_dat_def(cres, os.path.join(tmpdir, f"{res}.dat"))
                if outdir is not None:
                    os.makedirs(outdir, exist_ok=True)
                    pres = _prepare_for_parameterize(cres)
                    pres.write(os.path.join(outdir, f"{res}.cif"))
                logger.info(f"Succesfully templated non-canonical residue {res}.")
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to protonate non-canonical residue {res}. Please remove it from the protein or mutate it to continue preparation. Detailed error message: {e}"
                )
        definition, forcefield = _get_custom_ff(user_ff=tmpdir, molkit_ff=_molkit_ff)
    return definition, forcefield


def _detect_nonpeptidic_bonds(mol):
    coordination_ions = ("Ca", "Zn", "Mn", "Cu", "Fe", "Ni", "Mo", "Re", "Na")

    prot_idx = mol.atomselect("protein or resname ACE NME", indexes=True)
    ion_idx = np.where(np.isin(mol.element, coordination_ions))[0]
    same_resname = np.isin(
        mol.resname[ion_idx], [cc.upper() for cc in coordination_ions]
    )
    ion_idx = ion_idx[same_resname]

    # Bonds where only 1 atom belongs to protein
    bond_mask = np.isin(mol.bonds, prot_idx)
    # Exclude ions
    bond_mask[np.any(np.isin(mol.bonds, ion_idx), axis=1), :] = False
    inter_bonds = np.sum(bond_mask, axis=1) == 1

    bonds = mol.bonds[inter_bonds, :]
    b_prot_idx = bonds[bond_mask[inter_bonds]]
    b_other_idx = bonds[~bond_mask[inter_bonds]]

    if not len(b_prot_idx):
        return []

    logger.info(
        f"Found {len(b_prot_idx)} covalent bonds from protein to non-protein molecules."
    )
    return np.vstack([b_prot_idx, b_other_idx]).T


def _delete_no_titrate(pka_list, no_titr):
    pkas = []
    for res in pka_list:
        key = (res["res_num"], res["chain_id"].strip(), res["ins_code"].strip())
        if key not in no_titr:
            pkas.append(res)
        else:
            logger.info(
                f"Skipped titration of residue {res['res_name']}:{key[1]}:{key[0]}{key[2]}"
            )
    return pkas


def _apply_patches_force_prot(biomolecule, force_protonation):
    for res in biomolecule.residues:
        key = (res.res_seq, res.chain_id, res.ins_code)
        if key in force_protonation:
            logger.info(
                f"Forcing protonation of residue {res.name}:{key[1]}:{key[0]}{key[2]} to {force_protonation[key]}"
            )
            biomolecule.apply_patch(force_protonation[key], res)


def _pdb2pqr(
    pdb_file,
    definition,
    forcefield_,
    ph=7.4,
    assign_only=False,
    clean=False,
    debump=True,
    opt=True,
    drop_water=False,
    ligand=None,
    titrate=True,
    neutraln=False,
    neutralc=False,
    no_prot=None,
    no_opt=None,
    no_titr=None,
    propka_args=None,
):
    from pdb2pqr.io import get_molecule
    from pdb2pqr import forcefield, hydrogens
    from pdb2pqr.debump import Debump
    from pdb2pqr.main import is_repairable, run_propka, setup_molecule
    from pdb2pqr.main import drop_water as drop_water_func
    from propka.lib import build_parser

    propka_parser = build_parser()
    propka_parser.add_argument("--keep-chain", action="store_true", default=False)

    if propka_args is None:
        propka_args = {}
    propka_args_list = []
    for key, value in propka_args.items():
        propka_args_list += [key, value]
    propka_args_list += ["--log-level", "WARNING", "--pH", str(ph), "STEFAN"]
    propka_args = propka_parser.parse_args(propka_args_list)

    if assign_only or clean:
        debump = False
        opt = False

    pdblist, _ = get_molecule(pdb_file)
    if drop_water:
        pdblist = drop_water_func(pdblist)

    biomolecule, definition, ligand = setup_molecule(pdblist, definition, ligand)
    biomolecule.set_termini(neutraln, neutralc)
    biomolecule.update_bonds()
    if clean:
        return None, None, biomolecule

    hydrogen_handler = hydrogens.create_handler()
    debumper = Debump(biomolecule)
    pka_list = None
    if assign_only:
        # TODO - I don't understand why HIS needs to be set to HIP for
        # assign-only
        biomolecule.set_hip()
    else:
        if is_repairable(biomolecule, ligand is not None):
            biomolecule.repair_heavy()

        biomolecule.update_ss_bridges()
        if debump:
            try:
                debumper.debump_biomolecule()
            except ValueError as err:
                err = f"Unable to debump biomolecule. {err}"
                raise ValueError(err)
        if titrate:
            biomolecule.remove_hydrogens()
            pka_list, _ = run_propka(propka_args, biomolecule)
            # Remove terminal pkas
            pka_list = [
                row
                for row in pka_list
                if row["group_label"].startswith(row["res_name"])
            ]

            # STEFAN mod: Delete pka values for residues we should not titrate
            pkas = _delete_no_titrate(pka_list, no_titr)
            biomolecule.apply_pka_values(
                forcefield_.name,
                ph,
                {
                    f"{row['res_name']} {row['res_num']} {row['chain_id']}": row["pKa"]
                    for row in pkas
                },
            )

        biomolecule.add_hydrogens(no_prot)  # STEFAN mod: Don't protonate residues
        if debump:
            debumper.debump_biomolecule()

        hydrogen_routines = hydrogens.HydrogenRoutines(debumper, hydrogen_handler)
        if opt:
            hydrogen_routines.set_optimizeable_hydrogens()
            biomolecule.hold_residues(no_opt)  # STEFAN mod: Don't optimize residues
            hydrogen_routines.initialize_full_optimization()
        else:
            hydrogen_routines.initialize_wat_optimization()
        hydrogen_routines.optimize_hydrogens()
        hydrogen_routines.cleanup()

    biomolecule.set_states()
    matched_atoms, missing_atoms = biomolecule.apply_force_field(forcefield_)
    if ligand is not None:
        logger.warning("Using ZAP9 forcefield for ligand radii.")
        ligand.assign_parameters()
        lig_atoms = []
        for residue in biomolecule.residues:
            tot_charge = 0
            for pdb_atom in residue.atoms:
                # Only check residues with HETATM
                if pdb_atom.type == "ATOM":
                    break
                try:
                    mol2_atom = ligand.atoms[pdb_atom.name]
                    pdb_atom.radius = mol2_atom.radius
                    pdb_atom.ffcharge = mol2_atom.charge
                    tot_charge += mol2_atom.charge
                    lig_atoms.append(pdb_atom)
                except KeyError:
                    err = f"Can't find HETATM {residue.name} {residue.res_seq} {pdb_atom.name} in MOL2 file"
                    logger.warning(err)
                    missing_atoms.append(pdb_atom)
        matched_atoms += lig_atoms

    name_scheme = forcefield.Forcefield("amber", definition, None)
    biomolecule.apply_name_scheme(name_scheme)

    return missing_atoms, pka_list, biomolecule


def _atomsel_to_hold(mol_in, sel):
    idx = mol_in.atomselect(sel, indexes=True)
    if len(idx) == 0:
        raise RuntimeError(f"Could not select any atoms with atomselection {sel}")
    residues = [(mol_in.resid[i], mol_in.chain[i], mol_in.insertion[i]) for i in idx]
    residues = list(set(residues))
    if len(residues) != 1:
        raise RuntimeError(
            f"no_opt selection {sel} selected multiple residues {residues}. Please be more specific on which residues to hold."
        )
    return residues[0]


def _get_hold_residues(
    mol_in, no_opt, no_prot, no_titr, force_protonation, nonpeptidic_bonds
):
    # Converts atom selections used for no optimization residues and no protonation residues
    # to tuples of (resid, chain, insertion) which are used by pdb2pqr as unique residue identifiers
    _no_opt = []
    if no_opt is not None:
        for sel in no_opt:
            _no_opt.append(_atomsel_to_hold(mol_in, sel))

    _no_prot = []
    if no_prot is not None:
        for sel in no_prot:
            _no_prot.append(_atomsel_to_hold(mol_in, sel))

    _no_titr = []
    if no_titr is not None:
        for sel in no_titr:
            _no_titr.append(_atomsel_to_hold(mol_in, sel))

    _force_prot = {}
    if force_protonation is not None:
        for sel, resn in force_protonation:
            _force_prot[_atomsel_to_hold(mol_in, sel)] = resn

    # Add residues which should not be protonated to the residues which should not be titrated
    _no_titr += _no_prot
    # Add residues which have forced protonations to the residues which should not be titrated
    _no_titr += list(_force_prot.keys())

    if len(nonpeptidic_bonds) != 0:
        for nn in nonpeptidic_bonds:
            r1 = UniqueResidueID.fromMolecule(mol_in, idx=nn[0])
            r1 = f"{r1.resname}:{r1.chain}:{r1.resid}{r1.insertion}"
            r2 = UniqueResidueID.fromMolecule(mol_in, idx=nn[1])
            r2 = f"{r2.resname}:{r2.chain}:{r2.resid}{r2.insertion}"
            logger.info(
                f"Freezing protein residue {r1} bonded to non-protein molecule {r2}"
            )
            val = [
                (mol_in.resid[nn[0]], mol_in.chain[nn[0]], mol_in.insertion[nn[0]]),
                (mol_in.resid[nn[1]], mol_in.chain[nn[1]], mol_in.insertion[nn[1]]),
            ]
            _no_opt += val
            _no_titr += val
            if val[0] not in _force_prot:
                # Only disable protonation if there is no force-protonation
                _no_prot += val

    return _no_opt, _no_prot, _no_titr


def _check_frozen_histidines(mol_in, _no_prot):
    # Histidines require protonation by pdb2pqr. We cannot freeze them
    frozen_his = []
    for key in _no_prot:
        sel = (
            (mol_in.resid == key[0])
            & (mol_in.chain == key[1])
            & (mol_in.insertion == key[2])
        )
        if mol_in.resname[sel][0] == "HIS":
            frozen_his.append(key)

    if len(frozen_his):
        res = ", ".join([f"HIS:{key[1]}:{key[0]}{key[2]}" for key in frozen_his])
        raise RuntimeError(
            f"Histidines {res} are not auto-titrated. You need to manually define protonation states for them using the force_protonation argument."
        )


def _prepare_nucleics(mol):
    # Renames residues to the names expected by PARSE. Fixes issues with 5' atoms.
    nucl_mask = mol.atomselect("nucleic")
    uq_resn = mol.resname[nucl_mask]

    # Clean 5' dangling P atom
    for ch in np.unique(mol.get("chain", "nucleic")):
        # First residue of that chain
        resid = mol.resid[(mol.chain == ch) & mol.atomselect("nucleic")][0]
        remsel = (
            (mol.chain == ch)
            & (mol.resid == resid)
            & np.isin(mol.name, ("P", "OP1", "OP2"))
        )
        mol.remove(remsel, _logger=False)

    mapping = {
        "T": "DT",
        "U": "RU",
        "G": "RG",
        "C": "RC",
        "A": "RA",
        "DG": "DG",
        "DC": "DC",
        "DA": "DA",
        "DT": "DT",
    }
    for res in uq_resn:
        if res in mapping:
            mol.resname[mol.resname == res] = mapping[res]


def _fix_protonation_resnames(mol):
    # This function patches the names of residues to their correct protonation state if they have specific hydrogens
    uqres = sequenceID((mol.resid, mol.insertion, mol.chain))
    for uq in uqres:
        resatm = uqres == uq
        resname = mol.resname[resatm][0]
        names = mol.name[resatm]
        if (resname == "GLU") and ("HE2" in names):
            mol.resname[resatm] = "GLH"
        elif (resname == "ASP") and ("HD2" in names):
            mol.resname[resatm] = "ASH"
        elif (resname == "HIS") and all(np.isin(("HD1", "HD2", "HE1", "HE2"), names)):
            mol.resname[resatm] = "HIP"
        elif (resname == "HIS") and all(np.isin(("HD1", "HD2", "HE1"), names)):
            mol.resname[resatm] = "HID"
        elif (resname == "HIS") and all(np.isin(("HD2", "HE1", "HE2"), names)):
            mol.resname[resatm] = "HIE"


def proteinPrepare(
    mol_in,
    pH=7.0,
    verbose=0,
    returnDetails=False,
    hydrophobicThickness=None,
    holdSelection=None,
    _loggerLevel=None,
):
    logger.warning(
        "proteinPrepare has been deprecated in favor of the systemPrepare function and will soon be removed. "
        "Please look at the documentation of systemPrepare for more information."
    )
    return systemPrepare(
        mol_in,
        pH=pH,
        verbose=verbose,
        return_details=returnDetails,
        hydrophobic_thickness=hydrophobicThickness,
        _logger_level=_loggerLevel,
    )


def systemPrepare(
    mol_in,
    titration=True,
    pH=7.4,
    force_protonation=None,
    no_opt=None,
    no_prot=None,
    no_titr=None,
    hold_nonpeptidic_bonds=True,
    verbose=True,
    return_details=False,
    hydrophobic_thickness=None,
    plot_pka=None,
    ignore_ns_errors=False,
    _logger_level="ERROR",
    _molkit_ff=True,
    outdir=None,
    residue_smiles=None,
):
    """Prepare molecular systems through protonation and h-bond optimization.

    The preparation routine protonates and optimizes protein and nucleic residues.
    It will also take into account any non-protein, non-nucleic molecules for the pKa calculation
    but will not attempt to protonate or optimize those.

    Returns a Molecule object, where residues have been renamed to follow
    internal conventions on protonation (below). Coordinates are changed to
    optimize the H-bonding network.

    The following residue names are used in the returned molecule:

    === ===============================
    ASH Neutral ASP
    CYX SS-bonded CYS
    CYM Negative CYS
    GLH Neutral GLU
    HIP Positive HIS
    HID Neutral HIS, proton HD1 present
    HIE Neutral HIS, proton HE2 present
    LYN Neutral LYS
    TYM Negative TYR
    AR0 Neutral ARG
    === ===============================

    ========= ======= =========
    Charge +1 Neutral Charge -1
    ========= ======= =========
    -         ASH     ASP
    -         CYS     CYM
    -         GLH     GLU
    HIP       HID/HIE -
    LYS       LYN     -
    -         TYR     TYM
    ARG       AR0     -
    ========= ======= =========

    A detailed table about the residues modified is returned (as a second return value) when
    return_details is True .

    If hydrophobic_thickness is set to a positive value 2*h, a warning is produced for titratable residues
    having -h<z<h and are buried in the protein by less than 75%. Note that the heuristic for the
    detection of membrane-exposed residues is very crude; the "buried fraction" computation
    (from propKa) is approximate; also, in the presence of cavities,
    residues may be solvent-exposed independently from their z location.


    Notes
    -----

    Features:
     - assigns protonation states via propKa
     - flips residues to optimize H-bonding network
     - debumps collisions
     - fills in missing atoms, e.g. hydrogen atoms


    Parameters
    ----------
    mol_in : moleculekit.molecule.Molecule
        the object to be optimized
    titration : bool
        If set to True it will use propka to set the titration state of residues. Otherwise it will just add and optimize the hydrogens.
    pH : float
        pH to decide titration
    verbose : bool
        verbosity
    return_details : bool
        whether to return just the prepared Molecule (False, default) or a molecule *and* a pandas DataFrame
        object including computed properties
    hydrophobic_thickness : float
        the thickness of the membrane in which the protein is embedded, or None if globular protein.
        Used to provide a warning about membrane-exposed residues.
    force_protonation : list of tuples
        Allows the user to force specific protonations on residues. This can be done by providing a list of tuples,
        one for each residue we want to force. i.e. [("protein and resid 40", "HID")] will force the protonation of
        the first atomselection to the second resname. Atomselections should be valid VMD atomselections.
    no_opt : list of str
        Allows the user to disable optimization for specific residues. For example if the user determines that a
        residue flip or change of coordinates performed by this method causes issues in the structure, they can
        disable optimization on that residue by passing an atomselection for the residue to hold. i.e. ["protein and resid 23"].
    no_prot : list of str
        Same as no_opt but disables the addition of hydrogens to specific residues.
    no_titr : list of str
        Same as no_opt but disables the titration of specific residues.
    hold_nonpeptidic_bonds : bool
        When set to True, systemPrepare will automatically not optimize, protonate or titrate protein residues
        which are covalently bound to non-protein molecules. When set to False, systemPrepare will optimize them
        ignoring the covalent bond, meaning it may break the bonds or add hydrogen atoms between the bonds.
    plot_pka : str
        Provide a file path with .png extension to draw the titration diagram for the system residues.
    ignore_ns_errors : bool
        If False systemPrepare will issue an error when it fails to protonate non-canonical residues in the protein.
        If True it will ignore errors on non-canonical residues leaving them unprotonated.
    outdir : str
        A path where to save custom residue cif files used for building

    Returns
    -------
    mol_out : moleculekit.molecule.Molecule
        the molecule titrated and optimized. The molecule object contains an additional attribute,
    details : pandas.DataFrame
        A table of residues with the corresponding protonation states, pKas, and other information


    Examples
    --------
    >>> tryp = Molecule('3PTB')
    >>> tryp_op, df = systemPrepare(tryp, return_details=True)
    >>> tryp_op.write('/tmp/3PTB_prepared.pdb')
    >>> df.to_excel("/tmp/tryp-report.csv")
    >>> df                                                        # doctest: +NORMALIZE_WHITESPACE
    resname protonation  resid insertion chain segid       pKa    buried
    0       ILE         ILE     16               A     0  7.413075  0.839286
    1       VAL         VAL     17               A     0       NaN       NaN
    2       GLY         GLY     18               A     0       NaN       NaN
    3       GLY         GLY     19               A     0       NaN       NaN
    4       TYR         TYR     20               A     0  9.590845  0.146429
    ..      ...         ...    ...       ...   ...   ...       ...       ...
    282     HOH         WAT    804               A     1       NaN       NaN
    283     HOH         WAT    805               A     1       NaN       NaN
    284     HOH         WAT    807               A     1       NaN       NaN
    285     HOH         WAT    808               A     1       NaN       NaN
    286     HOH         WAT    809               A     1       NaN       NaN

    [287 rows x 8 columns]

    >>> tryp_op = systemPrepare(tryp, pH=1.0)
    >>> tryp_op.write('/tmp/3PTB_pH1.pdb')

    The following will force the preparation to freeze residues 36 and 49 in place
    >>> tryp_op = systemPrepare(tryp, no_opt=["protein and resid 36", "chain A and resid 49"])

    The following will disable protonation on residue 32 of the protein
    >>> tryp_op = systemPrepare(tryp, no_prot=["protein and resid 32",])

    The following will disable titration and protonation on residue 32
    >>> tryp_op = systemPrepare(tryp, no_titr=["protein and resid 32",], no_prot=["protein and resid 32",])

    The following will force residue 40 protonation to HIE and 57 to HIP
    >>> tryp_op = systemPrepare(tryp, force_protonation=[("protein and resid 40", "HIE"), ("protein and resid 57", "HIP")])
    """
    try:
        from pdb2pqr.config import VERSION
    except ImportError:
        raise ImportError(
            "pdb2pqr not installed. To use the system preparation features please do `conda install pdb2pqr -c acellera -c conda-forge`"
        )
    from moleculekit.tools.preparation_customres import _get_custom_ff

    old_level = logger.getEffectiveLevel()
    if not verbose:
        logger.setLevel(logging.WARNING)

    if _logger_level is not None:
        # logger.setLevel(_loggerLevel)
        logging.getLogger(f"PDB2PQR{VERSION}").setLevel(_logger_level)
        logging.getLogger(f"PDB2PQR{VERSION}").propagate = False
        logging.getLogger("pdb2pqr").setLevel(_logger_level)
        logging.getLogger("propka").setLevel(_logger_level)
    logger.debug("Starting.")

    mol_in = mol_in.copy()

    _warn_if_contains_DUM(mol_in)

    mol_in = _check_chain_and_segid(mol_in, verbose)
    mol_orig = mol_in.copy()

    _prepare_nucleics(mol_in)
    _fix_protonation_resnames(mol_in)

    definition, forcefield = _get_custom_ff(molkit_ff=_molkit_ff)
    definition, forcefield = _generate_nonstandard_residues_ff(
        mol_in,
        definition,
        forcefield,
        _molkit_ff,
        outdir,
        ignore_ns_errors=ignore_ns_errors,
        residue_smiles=residue_smiles,
    )

    nonpept = None
    if hold_nonpeptidic_bonds:
        nonpept = _detect_nonpeptidic_bonds(mol_in)

    if force_protonation is not None:
        for sel, resn in force_protonation:
            # Remove side-chain hydrogens
            mol_in.remove(f"({sel}) and not backbone and element H", _logger=False)
            # Rename to desired protonation state
            mol_in.set("resname", resn, sel=sel)

    _no_opt, _no_prot, _no_titr = _get_hold_residues(
        mol_in, no_opt, no_prot, no_titr, force_protonation, nonpept
    )
    _check_frozen_histidines(mol_in, _no_prot)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpin = os.path.join(tmpdir, "input.pdb")
        mol_in.write(tmpin)

        missedres, pka_df, biomolecule = _pdb2pqr(
            tmpin,
            ph=pH,
            titrate=titration,
            no_opt=_no_opt,
            no_prot=_no_prot,
            no_titr=_no_titr,
            definition=definition,
            forcefield_=forcefield,
        )
        mol_out = _biomolecule_to_molecule(biomolecule)

    # Diagnostics
    missedres = set([m.residue.name for m in missedres])
    if len(missedres):
        logger.warning(
            f"The following residues have not been optimized: {', '.join(missedres)}"
        )

    mol_out.box = mol_in.box
    _fixup_water_names(mol_out)

    df = _create_table(mol_orig, mol_out, pka_df)

    _list_modifications(df)

    if titration:
        if plot_pka is not None:
            try:
                _get_pka_plot(df, plot_pka)
            except Exception as e:
                logger.error(f"Failed at generating pKa plot with error {e}")
        _warn_pk_close_to_ph(df, pH)
        if hydrophobic_thickness:
            # TODO: I think this only works if the protein is assumed aligned to the membrane and the membrane is centered at Z=0
            _warn_buried_residues(df, mol_out, hydrophobic_thickness)

    logger.setLevel(old_level)

    if return_details:
        return mol_out, df
    return mol_out


def _biomolecule_to_molecule(biomolecule):
    propmap = [
        ("type", "record"),
        ("serial", "serial"),
        ("name", "name"),
        ("alt_loc", "altloc"),
        ("res_name", "resname"),
        ("chain_id", "chain"),
        ("res_seq", "resid"),
        ("ins_code", "insertion"),
        ("occupancy", "occupancy"),
        ("temp_factor", "beta"),
        ("seg_id", "segid"),
        ("element", "element"),
        ("charge", "formalcharge"),
    ]
    mol = Molecule().empty(len(biomolecule.atoms))
    mol.coords = np.zeros((mol.numAtoms, 3, 1), dtype=Molecule._dtypes["coords"])
    bonds = []
    for i, atom in enumerate(biomolecule.atoms):
        for pp1, pp2 in propmap:
            val = getattr(atom, pp1)
            if pp1 == "charge":
                if "+" in val:
                    val = int(val.replace("+", ""))
                elif "-" in val:
                    val = -1 * int(val.replace("-", ""))
                else:
                    val = 0
            if pp1 == "element":
                val = str(val).capitalize()
            mol.__dict__[pp2][i] = val
        mol.coords[i, :, 0] = [atom.x, atom.y, atom.z]
        bonds += [[i, x.serial - 1] for x in atom.bonds]

    # guessedBonds = mol._guessBonds()
    # bb = mol.atomselect("backbone", indexes=True)
    # bb = guessedBonds[np.all(np.isin(guessedBonds, bb), axis=1)]
    # mol.bonds = np.vstack(bb.tolist() + bonds)

    return mol


_table_dtypes = {
    "resname": str,
    "protonation": str,
    "resid": int,
    "insertion": str,
    "chain": str,
    "segid": str,
    "pKa": float,
    "buried": float,
}


def _create_table(mol_in, mol_out, pka_df):
    import pandas as pd

    uq_tuple = []
    for e in zip(
        mol_in.resname, mol_in.resid, mol_in.insertion, mol_in.chain, mol_in.segid
    ):
        if e not in uq_tuple:
            uq_tuple.append(e)

    data = []
    for tup in uq_tuple:
        old_resn = tup[0]
        new_mask = (
            (mol_out.resid == tup[1])
            & (mol_out.insertion == tup[2])
            & (mol_out.chain == tup[3])
            & (mol_out.segid == tup[4])
        )

        if not np.any(new_mask):
            raise RuntimeError(
                f"Unable to find residue {' '.join(map(str, tup))} after preparation"
            )
        new_idx = np.where(new_mask)[0][0]
        resname = mol_out.resname[new_idx]
        resid = mol_out.resid[new_idx]
        insertion = mol_out.insertion[new_idx]
        chain = mol_out.chain[new_idx]
        segid = mol_out.segid[new_idx]
        curr_data = [old_resn, resname, resid, insertion, chain, segid]
        if pka_df is not None:
            found = False
            for propkadata in pka_df:
                if (
                    propkadata["res_num"] == resid
                    and propkadata["ins_code"].strip() == insertion.strip()
                    and propkadata["chain_id"] == chain
                ):
                    curr_data += [propkadata["pKa"], propkadata["buried"]]
                    found = True
                    break
            if not found:
                curr_data += [np.nan, np.nan]
        data.append(curr_data)

    cols = [
        "resname",
        "protonation",
        "resid",
        "insertion",
        "chain",
        "segid",
    ]
    dtypes = _table_dtypes
    if pka_df is not None:
        cols += ["pKa", "buried"]
    else:
        del dtypes["pKa"]
        del dtypes["buried"]
    df = pd.DataFrame(data=data, columns=cols)
    df = df.astype(dtypes)
    return df


def _fmt_res(resname, resid, insertion, chain=""):
    return f"{resname:<4s} {resid:>4d}{insertion.strip():<s}{chain:>2s}"


def _list_modifications(df):
    for _, row in df[df.resname != df.protonation].iterrows():
        if row.resname in ["HOH", "WAT"]:
            continue

        old_resn = row.resname
        new_resn = row.protonation
        ch = row.chain
        rid = row.resid
        ins = row.insertion.strip()
        logger.info(
            f"Modified residue {_fmt_res(old_resn, rid, ins, ch)} to {new_resn}"
        )


def _warn_pk_close_to_ph(df, pH, tol=1.0):
    dubious = df[abs(df.pKa - pH) < tol]
    if len(dubious):
        logger.warning(
            f"Dubious protonation state: the pKa of {len(dubious)} residues is within {tol:.1f} units of pH {pH:.1f}."
        )
        for _, dr in dubious.iterrows():
            logger.warning(
                f"Dubious protonation state:    {_fmt_res(dr.resname, dr.resid, dr.insertion, dr.chain)} (pKa={dr.pKa:5.2f})"
            )


def _warn_buried_residues(df, mol_out, hydrophobic_thickness, maxBuried=0.75):
    ht = hydrophobic_thickness / 2.0
    count = 0
    for _, row in df[df.buried > maxBuried].iterrows():
        sele = (
            (mol_out.chain == row.chain)
            & (mol_out.resid == row.resid)
            & (mol_out.insertion == row.insertion)
        )
        mean_z = np.mean(mol_out.coords[sele, 2, 0])
        if -ht < mean_z < ht:
            count += 1

    if count:
        logger.warning(
            f"Predictions for {count} residues may be incorrect because they are "
            + f"exposed to the membrane ({-ht:.1f}<z<{ht:.2f} and buried<{maxBuried:.1f}%)."
        )


def _get_pka_plot(df, outname, pH=7.4, figSizeX=13, dpk=1.0, font_size=12):
    """Internal function to build the protonation diagram"""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patheffects as PathEffects

    if not np.any(df.pKa.notnull()):
        logger.warning(
            "No pka plot was generated due to no pKa values calculated by propka."
        )
        return

    # Shading
    Xe = np.array([[1, 0], [1, 0]])

    # Shading colors http://matplotlib.org/examples/pylab_examples/custom_cmap.html
    neutral_grey = (0.7, 0.7, 0.7)
    my_red = (0.98, 0.41, 0.29)
    my_blue = (0.42, 0.68, 0.84)
    grey_red = LinearSegmentedColormap.from_list("grey_red", [neutral_grey, my_red])
    grey_blue = LinearSegmentedColormap.from_list("grey_blue", [neutral_grey, my_blue])
    eps = 0.01  # Tiny overprint to avoid very thin white lines
    outline = [PathEffects.withStroke(linewidth=2, foreground="w")]

    # Color for pk values
    pKa_color = "black"
    pKa_fontsize = 8
    dtxt = 0  # Displacement

    # Or we could change the figure size, which scales axes
    # http://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    plt.rc("font", family="monospace")
    plt.rc("font", size=font_size)  # controls default text sizes
    plt.rc("axes", titlesize=font_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=font_size)  # legend fontsize
    plt.rc("figure", titlesize=font_size)  # fontsize of the figure title

    # Constants
    acidicResidues = ["ASP", "GLU", "TYR", "C-"]

    df = df[df.pKa < 99]

    # Format residue labels
    labels = []
    pKas = []
    restypes = []
    for _, row in df.iterrows():
        dub = "(!)" if abs(row.pKa - pH) < dpk else ""
        labels.append(
            f"{dub} {row.chain}:{row.resid}{row.insertion.strip()}-{row.resname}"
        )
        pKas.append(row.pKa)
        restypes.append("neg" if row.resname in acidicResidues else "pos")

    N = len(df)
    xmin, xmax = xlim = 0, 14
    ymin, ymax = ylim = -1, N

    width = 0.8  # Of each band

    # So, arbitrarily, 40 residues are square
    sizePerBand = figSizeX * (N / 40)
    figsize = (figSizeX, sizePerBand)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, xlim=xlim, ylim=ylim, autoscale_on=False)

    ax.xaxis.tick_top()
    ax.set_xlabel("pKa")
    ax.xaxis.set_label_position("top")

    ax.yaxis.set_ticks(range(N))
    ax.yaxis.set_ticklabels(labels)
    ax.invert_yaxis()

    for i in range(N):
        left = xmin
        right = xmax
        top = i + width / 2
        bottom = i - width / 2
        pk = pKas[i]
        restype = restypes[i]

        if restype == "neg":
            ax.imshow(
                Xe * 0,
                interpolation="none",
                cmap=grey_blue,
                vmin=0,
                vmax=1,
                extent=(left, pk - dpk, bottom, top),
                alpha=1,
            )
            ax.imshow(
                np.fliplr(Xe),
                interpolation="bicubic",
                cmap=grey_blue,
                vmin=0,
                vmax=1,
                extent=(pk - dpk - eps, pk + dpk, bottom, top),
                alpha=1,
            )
            ax.imshow(
                1 + Xe * 0,
                interpolation="none",
                cmap=grey_blue,
                vmin=0,
                vmax=1,
                extent=(pk + dpk - eps, right, bottom, top),
                alpha=1,
            )
            ax.text(
                pk - dtxt,
                i,
                " {:.2f} ".format(pk),
                color=pKa_color,
                fontsize=pKa_fontsize,
                horizontalalignment="right",
                zorder=30,
                path_effects=outline,
                weight="bold",
            )
        else:
            ax.imshow(
                1 + Xe * 0,
                interpolation="none",
                cmap=grey_red,
                vmin=0,
                vmax=1,
                extent=(left, pk - dpk, bottom, top),
                alpha=1,
            )
            ax.imshow(
                Xe,
                interpolation="bicubic",
                cmap=grey_red,
                vmin=0,
                vmax=1,
                extent=(pk - dpk - eps, pk + dpk, bottom, top),
                alpha=1,
            )
            ax.imshow(
                Xe * 0,
                interpolation="none",
                cmap=grey_red,
                vmin=0,
                vmax=1,
                extent=(pk + dpk - eps, right, bottom, top),
                alpha=1,
            )
            ax.text(
                pk + dtxt,
                i,
                " {:.2f} ".format(pk),
                color=pKa_color,
                fontsize=pKa_fontsize,
                horizontalalignment="left",
                zorder=30,
                path_effects=outline,
                weight="bold",
            )
        ax.add_line(
            Line2D([pk, pk], [bottom, top], linewidth=3, color="white", zorder=2)
        )

        # ax.add_line(Line2D([pk,pk], [bottom,top], linewidth=3, color='blue'))

    # Shaded vertical band at pH
    ax.axvline(x=pH - dpk, linewidth=2, color="black", alpha=0.2, linestyle="dashed")
    ax.axvline(x=pH + dpk, linewidth=2, color="black", alpha=0.2, linestyle="dashed")
    ax.axvline(x=pH, linewidth=3, color="black", alpha=0.5)
    ax.text(
        pH - dpk,
        ymax,
        " 90% protonated",
        rotation=90,
        horizontalalignment="right",
        verticalalignment="bottom",
        style="italic",
        path_effects=outline,
    )
    ax.text(
        pH + dpk,
        ymax,
        " 10% protonated",
        rotation=90,
        horizontalalignment="left",
        verticalalignment="bottom",
        style="italic",
        path_effects=outline,
    )

    ax.set_aspect("auto")
    plt.savefig(outname, dpi=300)
    plt.close(fig)


# The below is used for testing only
try:
    from aceprep.prepare import rdk_prepare
except ImportError:
    ACEPREP_EXISTS = False
else:
    ACEPREP_EXISTS = True


class _TestPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from moleculekit.home import home

        self.home = home(dataDir="test-systemprepare")

    def _compare_results(self, refpdb, refdf_f, pmol: Molecule, df: pd.DataFrame):
        from moleculekit.util import tempname

        # # Use this to update tests
        # pmol.filter("not water", _logger=False)
        # pmol.write(refpdb, writebonds=False)
        # df.to_csv(refdf_f, index=False)

        refdf = pd.read_csv(
            refdf_f, dtype=_table_dtypes, keep_default_na=False, na_values=[""]
        )
        refdf = refdf.fillna("")
        df = df.fillna("")
        try:
            pd.testing.assert_frame_equal(refdf, df)
        except AssertionError:
            if len(df) != len(refdf):
                raise AssertionError(
                    "Different number of residues were found in the DataFrames!"
                )
            try:
                for key in [
                    "resname",
                    "protonation",
                    "resid",
                    "insertion",
                    "chain",
                    "segid",
                    "pKa",
                    "buried",
                ]:
                    refv = refdf[key].values
                    newv = df[key].values
                    diff = refv != newv
                    if key in ("pKa", "buried"):
                        nans = refv == ""
                        refv = refv[~nans].astype(np.float32)
                        newv = newv[~nans].astype(np.float32)
                        diff = np.abs(refv - newv) > 1e-4
                    if any(diff):
                        print(f"Found difference in field: {key}")
                        print(f"Reference values: {refv[diff]}")
                        print(f"New values:       {newv[diff]}")
            except Exception:
                pass

            df_out = tempname(suffix=".csv")
            df.to_csv(df_out, index=False)
            raise AssertionError(f"Failed comparison of {refdf_f} to {df_out}")

        refmol = Molecule(refpdb)
        refmol.filter("not water", _logger=False)
        pmol.filter("not water", _logger=False)
        assert mol_equal(
            refmol, pmol, exceptFields=["serial"], fieldPrecision={"coords": 1e-3}
        )

    def test_systemPrepare(self):
        pdbids = ["3PTB", "1A25", "1U5U", "1UNC", "6A5J"]
        for pdb in pdbids:
            with self.subTest(pdbid=pdb):
                test_home = os.path.join(self.home, pdb)
                mol = Molecule(os.path.join(test_home, f"{pdb}.pdb"))
                pmol, df = systemPrepare(mol, return_details=True)
                self._compare_results(
                    os.path.join(test_home, f"{pdb}_prepared.pdb"),
                    os.path.join(test_home, f"{pdb}_prepared.csv"),
                    pmol,
                    df,
                )

    def test_systemprepare_ligand(self):
        test_home = os.path.join(self.home, "test-prepare-with-ligand")
        mol = Molecule(os.path.join(test_home, "5EK0_A.pdb"))
        pmol, df = systemPrepare(mol, return_details=True)
        self._compare_results(
            os.path.join(test_home, "5EK0_A_prepared.pdb"),
            os.path.join(test_home, "5EK0_A_prepared.csv"),
            pmol,
            df,
        )
        # Now remove the ligands and check again what the pka is
        mol.filter('not resname PX4 "5P2"')
        pmol, df = systemPrepare(mol, return_details=True)
        self._compare_results(
            os.path.join(test_home, "5EK0_A_prepared_nolig.pdb"),
            os.path.join(test_home, "5EK0_A_prepared_nolig.csv"),
            pmol,
            df,
        )

    def test_reprotonate(self):
        pmol, df = systemPrepare(Molecule("3PTB"), return_details=True)
        assert df.protonation[df.resid == 40].iloc[0] == "HIE"
        assert df.protonation[df.resid == 57].iloc[0] == "HIP"
        assert df.protonation[df.resid == 91].iloc[0] == "HID"

        pmol.mutateResidue("protein and resid 40", "HID")
        _, df2 = systemPrepare(pmol, titration=False, return_details=True)
        assert df2.protonation[df2.resid == 40].iloc[0] == "HID"
        assert df2.protonation[df2.resid == 57].iloc[0] == "HIP"
        assert df2.protonation[df2.resid == 91].iloc[0] == "HID"

        pmol, df = systemPrepare(
            Molecule("3PTB"),
            force_protonation=(
                ("protein and resid 40", "HID"),
                ("protein and resid 91", "HIE"),
            ),
            return_details=True,
        )
        assert df.protonation[df.resid == 40].iloc[0] == "HID"
        assert df.protonation[df.resid == 57].iloc[0] == "HIP"
        assert df.protonation[df.resid == 91].iloc[0] == "HIE"

    def test_auto_freezing(self):
        test_home = os.path.join(self.home, "test-auto-freezing")
        mol = Molecule(os.path.join(test_home, "2B5I.pdb"))

        pmol, df = systemPrepare(mol, return_details=True, hold_nonpeptidic_bonds=True)
        self._compare_results(
            os.path.join(test_home, "2B5I_prepared.pdb"),
            os.path.join(test_home, "2B5I_prepared.csv"),
            pmol,
            df,
        )

    def test_auto_freezing_and_force(self):
        test_home = os.path.join(self.home, "test-auto-freezing")
        mol = Molecule(os.path.join(test_home, "5DPX_A.pdb"))

        pmol, df = systemPrepare(
            mol,
            return_details=True,
            hold_nonpeptidic_bonds=True,
            force_protonation=[
                ("protein and resid 105 and chain A", "HIE"),
                ("protein and resid 107 and chain A", "HIE"),
                ("protein and resid 110 and chain A", "HIE"),
                ("protein and resid 181 and chain A", "HIE"),
                ("protein and resid 246 and chain A", "HIE"),
            ],
        )
        self._compare_results(
            os.path.join(test_home, "5DPX_A_prepared.pdb"),
            os.path.join(test_home, "5DPX_A_prepared.csv"),
            pmol,
            df,
        )

    @unittest.skipUnless(ACEPREP_EXISTS, "Can only run with aceprep installed")
    def test_nonstandard_residues(self):
        test_home = os.path.join(self.home, "test-nonstandard-residues")
        files = {
            "1A4W.pdb": "1A4W_prepared",
            "5VBL.pdb": "5VBL_prepared",
            "2QRV.pdb": "2QRV_prepared",
        }
        res_smiles = {
            "200": "c1cc(ccc1C[C@@H](C(=O)O)N)Cl",
            "HRG": "C(CCNC(=N)N)C[C@@H](C(=O)O)N",
            "OIC": "C1CC[C@H]2[C@@H](C1)C[C@H](N2)C(=O)O",
            "TYS": "c1cc(ccc1C[C@@H](C(=O)O)N)OS(=O)(=O)O",
            "SAH": "c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CSCC[C@@H](C(=O)O)N)O)O)N",
        }
        for inf, outf in files.items():
            mol = Molecule(os.path.join(test_home, inf))
            if inf == "2QRV.pdb":
                mol = autoSegment2(mol, fields=("chain", "segid"))

            with self.subTest(file=inf + "_smiles"):
                pmol, df = systemPrepare(
                    mol,
                    return_details=True,
                    hold_nonpeptidic_bonds=True,
                    residue_smiles=res_smiles,
                )

                self._compare_results(
                    os.path.join(test_home, f"{outf}.pdb"),
                    os.path.join(test_home, f"{outf}.csv"),
                    pmol,
                    df,
                )

            with self.subTest(file=inf + "_aceprep"):
                pmol, df = systemPrepare(
                    mol, return_details=True, hold_nonpeptidic_bonds=True
                )

                self._compare_results(
                    os.path.join(test_home, f"{outf}.pdb"),
                    os.path.join(test_home, f"{outf}.csv"),
                    pmol,
                    df,
                )

    @unittest.skipIf(ACEPREP_EXISTS, "Can only run WITHOUT aceprep installed")
    def test_nonstandard_residue_hard_ignore_ns(self):
        test_home = os.path.join(self.home, "test-nonstandard-residues")
        mol = Molecule(os.path.join(test_home, "5VBL.pdb"))

        pmol, df = systemPrepare(
            mol,
            return_details=True,
            hold_nonpeptidic_bonds=True,
            _molkit_ff=False,
            ignore_ns_errors=True,
        )
        self._compare_results(
            os.path.join(test_home, "5VBL_prepared_ignore_ns.pdb"),
            os.path.join(test_home, "5VBL_prepared_ignore_ns.csv"),
            pmol,
            df,
        )

    def test_rna_protein_complex(self):
        test_home = os.path.join(self.home, "test-rna-protein-complex")
        mol = Molecule(os.path.join(test_home, "3WBM.pdb"))

        pmol, df = systemPrepare(mol, return_details=True)

        self._compare_results(
            os.path.join(test_home, "3WBM_prepared.pdb"),
            os.path.join(test_home, "3WBM_prepared.csv"),
            pmol,
            df,
        )

    def test_dna(self):
        test_home = os.path.join(self.home, "test-dna")
        mol = Molecule(os.path.join(test_home, "1BNA.pdb"))

        pmol, df = systemPrepare(mol, return_details=True)

        self._compare_results(
            os.path.join(test_home, "1BNA_prepared.pdb"),
            os.path.join(test_home, "1BNA_prepared.csv"),
            pmol,
            df,
        )

    def test_cyclic_peptides(self):
        test_home = os.path.join(self.home, "test-cyclic-peptides")
        mol = Molecule(os.path.join(test_home, "5VAV.pdb"))

        pmol, df = systemPrepare(mol, return_details=True)

        self._compare_results(
            os.path.join(test_home, "5VAV_prepared.pdb"),
            os.path.join(test_home, "5VAV_prepared.csv"),
            pmol,
            df,
        )

    @unittest.skipUnless(ACEPREP_EXISTS, "Can only run with aceprep installed")
    def test_cyclic_peptides_noncanonical(self):
        test_home = os.path.join(self.home, "test-cyclic-peptides")
        mol = Molecule(os.path.join(test_home, "4TOT_E.pdb"))

        pmol, df = systemPrepare(mol, return_details=True)

        self._compare_results(
            os.path.join(test_home, "4TOT_E_prepared.pdb"),
            os.path.join(test_home, "4TOT_E_prepared.csv"),
            pmol,
            df,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
    import doctest

    doctest.testmod()
