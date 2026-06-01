# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import logging
import tempfile
import numpy as np
import os
from moleculekit.molecule import Molecule, UniqueResidueID
from moleculekit.tools.backbone import check_backbone
from moleculekit.util import sequenceID

logger = logging.getLogger(__name__)


class MissingTopologyError(Exception):
    pass


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
            firstres = _fmt_res(
                mol.resname[firstatom], mol.resid[firstatom], mol.insertion[firstatom]
            )
            lastres = _fmt_res(
                mol.resname[lastatom], mol.resid[lastatom], mol.insertion[lastatom]
            )
            print(f"Chain {c}:")
            if firstres == lastres:
                print(f"    {'residue:':>14}  {firstres}")
            else:
                print(f"    {'First residue:':<16}{firstres}")
                print(f"    {'Final residue:':<16}{lastres}")
        print("---- End of chain report ----\n")

    return mol


def _generate_nonstandard_residues_ff(
    mol,
    definition,
    forcefield,
    detect_specs=None,
):
    import tempfile
    from moleculekit.tools.preparation_customres import _get_custom_ff
    from moleculekit.tools.preparation_customres import (
        _process_custom_residue,
        _mol_to_dat_def,
        _mol_to_xml_def,
    )
    from moleculekit.tools.nonstandard_residues import ChainResidueSpec

    # Only chain-resident non-canonical amino acids need FF templating here.
    # Free / covalent / scaffold ligands are handled separately (or held by
    # ``hold_nonpeptidic_bonds``); canonical AAs are already in the FF.
    # The residue is expected to have been templated upstream by
    # ``mol.templateResidueFromSmiles`` so we can extract it from ``mol``
    # directly via the spec's residue identity.
    spec_by_resname = {}
    if detect_specs:
        for spec in detect_specs:
            if not isinstance(spec, ChainResidueSpec):
                continue
            # Resname under which the residue appears in mol RIGHT NOW.
            # _template_renamed_canonical_residues has already applied any
            # rename to a custom name; for plain NCAAs with no rename,
            # resname is what's in mol.
            current_resname = spec.new_resname or spec.resname
            spec_by_resname.setdefault(str(current_resname), spec)

    not_in_ff = [r for r in spec_by_resname if not forcefield.has_residue(r)]

    if len(not_in_ff) == 0:
        return definition, forcefield

    with tempfile.TemporaryDirectory() as tmpdir:
        for res in not_in_ff:
            spec = spec_by_resname[res]
            rid = spec.residue
            mask = (
                (mol.resname == res)
                & (mol.segid == str(rid.segid))
                & (mol.chain == str(rid.chain))
                & (mol.resid == int(rid.resid))
                & (mol.insertion == str(rid.insertion))
            )
            if not mask.any():
                raise RuntimeError(
                    f"detect_specs entry for residue {res} "
                    f"({rid}) not found in the input structure."
                )
            molc = mol.copy()
            molc.filter(np.where(mask)[0], _logger=False)

            if len(np.unique(molc.name)) != molc.numAtoms:
                raise RuntimeError(
                    f"Residue {res} contains duplicate atom names. Please rename the atoms to have unique names."
                )

            # Ensure the H bonded to OXT is named HXT
            if "OXT" in molc.name:
                neigh = molc.getNeighbors(np.where(molc.name == "OXT")[0][0])
                for n in neigh:
                    if molc.element[n] == "H":
                        molc.name[n] = "HXT"

            molc.chain[:] = ""
            molc.segid[:] = ""
            cres = _process_custom_residue(molc)
            # Rename to correct resname
            cres.resname[:] = res

            _mol_to_xml_def(cres, os.path.join(tmpdir, f"{res}.xml"))
            _mol_to_dat_def(cres, os.path.join(tmpdir, f"{res}.dat"))

        definition, forcefield = _get_custom_ff(user_ff=tmpdir)
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


# PDB2PQR's built-in protein force-field templates cover these resnames
# directly; renaming a canonical AA to one of them is sufficient and no
# SMILES re-template is needed (PDB2PQR's hydrogen logic places the right
# atoms for the variant).
_PDB2PQR_KNOWN_VARIANTS = {
    "CYX", "CYM", "LYN", "HID", "HIE", "HIP",
    "TYM", "ASH", "GLH", "AR0",
}


def _restore_trimmed_canonical_sidechains(mol, detect_specs):
    """Reconstruct the heavy-atom sidechain of every canonical protein
    residue in ``mol`` whose sidechain is **completely** missing,
    using :func:`moleculekit.tools.mutate.mutate_residue` (Dunbrack
    backbone-dependent rotamer library, lowest-clash rotamer).

    Restoration only fires when the residue's heavy atoms are a
    subset of backbone + CB. Any residue carrying even one atom past
    CB is left alone: a partial sidechain almost always signals an
    intentional truncation (an isopeptide-bonded GLU keeps CG / CD
    but loses OE1 / OE2; a covalent ligand on LYS keeps CG-CE but
    loses NH3+ hydrogens; user-prepared structures may drop
    sidechain atoms deliberately). Re-templating them with a Dunbrack
    rotamer would silently introduce the omitted atoms again and
    break the intended chemistry.

    Excluded by construction:
      - GLY (no sidechain) and ALA (CB is the whole sidechain).
      - Non-canonical residues and any spec'd residues (NCAA or
        canonical-anchor renames handled by
        :func:`_template_renamed_canonical_residues`).
      - Residues without a full N / CA / C backbone -
        :func:`mutate_residue` requires it for Kabsch alignment.

    Mutates ``mol`` in place. Logs every reconstructed residue.
    """
    from moleculekit.tools.mutate import mutate_residue
    from moleculekit.residues import PROTEIN_RESIDUE_NAMES

    BACKBONE_AND_CB = frozenset(
        {"N", "CA", "C", "O", "OXT", "CB", "H", "H1", "H2", "H3", "HA"}
    )
    NO_RESTORE_RESNAMES = frozenset({"GLY", "ALA"})

    spec_keys = set()
    if detect_specs:
        for spec in detect_specs:
            r = spec.residue
            spec_keys.add(
                (str(r.segid), str(r.chain), int(r.resid), str(r.insertion))
            )

    # First pass: scan residues and record the IDs of the trimmed ones.
    # Per-iteration we rebuild the boolean mask from these IDs because
    # ``mutate_residue`` inserts new atoms and shifts every later
    # index, invalidating any mask captured up front.
    _, res_atom_idxs = mol.getResidues(return_idx=True)
    targets = []
    for atom_idxs in res_atom_idxs:
        i0 = int(atom_idxs[0])
        resname = str(mol.resname[i0])
        if resname not in PROTEIN_RESIDUE_NAMES or resname in NO_RESTORE_RESNAMES:
            continue
        key = (
            str(mol.segid[i0]),
            str(mol.chain[i0]),
            int(mol.resid[i0]),
            str(mol.insertion[i0]),
        )
        if key in spec_keys:
            continue
        names_present = {str(n) for n in mol.name[atom_idxs]}
        if not names_present.issubset(BACKBONE_AND_CB):
            continue
        if not {"N", "CA", "C"}.issubset(names_present):
            continue
        targets.append(key + (resname,))

    for segid, chain, resid, insertion, resname in targets:
        mask = (
            (mol.segid == segid)
            & (mol.chain == chain)
            & (mol.resid == resid)
            & (mol.insertion == insertion)
        )
        if not mask.any():
            continue
        try:
            mutate_residue(mol, sel=mask, newres=resname, rotamer_mode="best")
            logger.info(
                f"Restored missing sidechain atoms for {resname}:{chain}:{resid}"
            )
        except Exception as exc:
            logger.warning(
                f"Could not restore sidechain for {resname}:{chain}:{resid}: "
                f"{exc}. Falling back to PDB2PQR reconstruction."
            )


def _template_renamed_canonical_residues(mol, specs):
    """Rename + re-template every ChainResidueSpec whose
    ``resname`` is a canonical AA and that has a
    ``new_resname``. Mutates ``mol`` in place.

    For renames TO a known PDB2PQR variant (CYX, LYN, HID, ...) the
    helper just renames; PDB2PQR's built-in template handles
    protonation.

    For renames TO an auto-generated custom name (X##) the helper also
    calls :meth:`Molecule.templateResidueFromSmiles` with the canonical
    SMILES variant returned by
    :func:`moleculekit.tools._anchor_variants.canonical_anchor_smiles`.
    rdkit's valence math then places the right hydrogens at the
    junction (1 H on LYS NZ secondary amide, 0 H on GLU CD amide
    carbonyl, ...); heavy atoms displaced by the crosslink are
    auto-stripped by
    :func:`moleculekit.rdkittools._try_strip_unmatched_terminals`
    inside ``templateResidueFromSmiles`` (signal A:
    ``rdkittools.py:265-269``).
    """
    from moleculekit.tools.nonstandard_residues import (
        ChainResidueSpec, PROTEIN_RESNAMES,
    )
    from moleculekit.tools._anchor_variants import canonical_anchor_smiles

    for spec in specs:
        if not isinstance(spec, ChainResidueSpec):
            continue
        if spec.resname not in PROTEIN_RESNAMES:
            continue
        if not spec.new_resname:
            continue

        rid = spec.residue
        res_mask = (
            (mol.resname == str(spec.resname))
            & (mol.chain == str(rid.chain))
            & (mol.resid == int(rid.resid))
            & (mol.insertion == str(rid.insertion))
            & (mol.segid == str(rid.segid))
        )
        res_atom_idxs = np.where(res_mask)[0]
        if len(res_atom_idxs) == 0:
            raise RuntimeError(
                f"Residue {rid} from ChainResidueSpec not found in mol "
                f"(resname filter {spec.resname!r} returned 0)."
            )

        # Rename.
        mol.resname[res_atom_idxs] = str(spec.new_resname)

        if str(spec.new_resname) in _PDB2PQR_KNOWN_VARIANTS:
            continue  # PDB2PQR handles it natively

        if not spec.anchor_atom:
            raise RuntimeError(
                f"ChainResidueSpec for {rid.chain}:{rid.resid}{rid.insertion} "
                f"({spec.resname}->{spec.new_resname}) has no "
                f"anchor_atom but needs re-templating. The detector should "
                f"have populated this field."
            )

        smiles = canonical_anchor_smiles(spec.resname, spec.anchor_atom)

        # res_mask was computed against the original resname and the
        # atoms still live at the same indices after the rename above,
        # so it's still the correct selection here.
        mol.templateResidueFromSmiles(res_mask, smiles, addHs=True, guessBonds=True)


def _canonicalize_ncaa_h_names(mol, detect_specs):
    """Rename the amide H, alpha H, and carboxyl-OH H of each
    chain-resident NCAA in ``mol`` to the AMBER conventions ``H``,
    ``HA``, and ``HXT``. The FF template written by
    :func:`_process_custom_residue` always uses these names, so
    without this rename the auto-generated ``H1``/``H2``/... that
    ``mol.templateResidueFromSmiles(addHs=True)`` produces would not
    match the FF template and PDB2PQR's biomolecule pass would either
    treat the residue as N-terminal (expecting NH3+) or fail to walk
    the path to ``CA``.

    The ``OXT -> HXT`` rename specifically avoids a name collision
    PDB2PQR special-cases: the third NH3+ proton on an N-terminal
    canonical residue is named ``H3``, and PDB2PQR's
    ``set_reference_distance`` check trusts that name to mean an
    N-terminal proton. RDKit's generic ``H1``/``H2``/``H3``/... naming
    can place an ``H3`` on a non-N-terminal residue's carboxyl OXT,
    which then trips the same check (PDB2PQR has no bond info for our
    CustomResidue and so cannot walk the path to ``CA``). Renaming to
    ``HXT`` (the AMBER carboxyl-OH name PDB2PQR already special-cases
    via the ``HO`` alias path) sidesteps the collision.
    """
    if not detect_specs:
        return
    from moleculekit.tools.nonstandard_residues import ChainResidueSpec

    for spec in detect_specs:
        if not isinstance(spec, ChainResidueSpec):
            continue
        rid = spec.residue
        current_resname = spec.new_resname or spec.resname
        res_mask = (
            (mol.resname == str(current_resname))
            & (mol.segid == str(rid.segid))
            & (mol.chain == str(rid.chain))
            & (mol.resid == int(rid.resid))
            & (mol.insertion == str(rid.insertion))
        )
        res_idx = np.where(res_mask)[0]
        if not len(res_idx):
            continue
        res_set = set(int(i) for i in res_idx)

        for backbone_name, h_target in (("N", "H"), ("CA", "HA"), ("OXT", "HXT")):
            bb_atoms = res_idx[mol.name[res_idx] == backbone_name]
            if not len(bb_atoms):
                continue
            for nb in mol.getNeighbors(int(bb_atoms[0])):
                if int(nb) in res_set and mol.element[int(nb)] == "H":
                    mol.name[int(nb)] = h_target
                    break


def _delete_no_titrate(pka_list, no_titr):
    pkas = []
    logged = set()
    for res in pka_list:
        key = (res["res_num"], res["chain_id"].strip(), res["ins_code"].strip())
        if key not in no_titr:
            pkas.append(res)
            continue
        # PDB2PQR emits one pka_list row per ionizable site within a residue
        # (e.g. HEM has 4 pyrrole Ns + 2 propionate COOHs = 6 rows). Log once
        # per residue to avoid 6x duplicate "Skipping titration" lines.
        if key in logged:
            continue
        logged.add(key)
        logger.info(
            f"Skipping titration of residue {res['res_name']}:{key[1]}:{key[0]}{key[2]}"
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
    biomolecule.set_termini(neutraln=neutraln, neutralc=neutralc)
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
    residues = [
        (mol_in.resid[i].item(), mol_in.chain[i], mol_in.insertion[i]) for i in idx
    ]
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
            res = _atomsel_to_hold(mol_in, sel)
            _no_opt.append(res)
            logger.info(f"Skipping optimization of residue {res[1]}:{res[0]}{res[2]}")

    _no_prot = []
    if no_prot is not None:
        for sel in no_prot:
            res = _atomsel_to_hold(mol_in, sel)
            _no_prot.append(res)
            logger.info(f"Skipping protonation of residue {res[1]}:{res[0]}{res[2]}")

    _no_titr = []
    if no_titr is not None:
        for sel in no_titr:
            _no_titr.append(_atomsel_to_hold(mol_in, sel))

    _force_prot = {}
    if force_protonation is not None:
        for sel, resn in force_protonation:
            res = _atomsel_to_hold(mol_in, sel)
            _force_prot[res] = resn
            logger.info(
                f"Forcing protonation of residue {res[1]}:{res[0]}{res[2]} to {resn}"
            )

    # Add residues which should not be protonated to the residues which should not be titrated
    _no_titr += _no_prot
    # Add residues which have forced protonations to the residues which should not be titrated
    _no_titr += list(_force_prot.keys())

    # Standard protein residues sitting at a covalent junction to a
    # non-protein molecule. Each entry: ((resid, chain, insertion), resname,
    # anchor_atom_name). These residues must not be titrated or geometry-
    # optimized (that could disturb the covalent linkage), but they should
    # still get their hydrogens added by PDB2PQR; the hydrogen displaced by
    # the covalent bond is removed afterwards (see _drop_displaced_anchor_h).
    _frozen = []
    if len(nonpeptidic_bonds) != 0:
        for nn in nonpeptidic_bonds:
            r1 = UniqueResidueID.fromMolecule(mol_in, idx=nn[0])
            r2 = UniqueResidueID.fromMolecule(mol_in, idx=nn[1])
            logger.info(
                f"Freezing protein residue {r1.resname}:{r1.chain}:{r1.resid}{r1.insertion} "
                f"bonded to non-protein molecule {r2.resname}:{r2.chain}:{r2.resid}{r2.insertion}"
            )
            prot_key = (
                mol_in.resid[nn[0]].item(),
                mol_in.chain[nn[0]],
                mol_in.insertion[nn[0]],
            )
            other_key = (
                mol_in.resid[nn[1]].item(),
                mol_in.chain[nn[1]],
                mol_in.insertion[nn[1]],
            )
            _no_opt += [prot_key, other_key]
            _no_titr += [prot_key, other_key]
            # Never let PDB2PQR protonate the non-protein partner; it is
            # templated separately. The protein junction residue is left out
            # of _no_prot on purpose so PDB2PQR still hydrogenates it.
            _no_prot.append(other_key)
            _frozen.append(
                (prot_key, str(mol_in.resname[nn[0]]), str(mol_in.name[nn[0]]))
            )

    return _no_opt, _no_prot, _no_titr, _frozen


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
    resnames = ("T", "U", "G", "C", "A", "DG", "DC", "DA", "DT", "RU", "RG", "RC", "RA")
    # Renames residues to the names expected by PARSE. Fixes issues with 5' atoms.
    nucleic_sel = f"nucleic and resname {' '.join(resnames)}"
    nucl_mask = mol.atomselect(nucleic_sel)
    uq_resn = mol.resname[nucl_mask]

    if len(set(mol.resid[nucl_mask])) > 2:
        # Need at least two nucleic residues to apply fixes
        # Clean 5' dangling P atom
        for ch in np.unique(mol.get("chain", nucleic_sel)):
            # First residue of that chain
            resid = mol.resid[(mol.chain == ch) & mol.atomselect(nucleic_sel)][0]
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
    uqres = mol.getResidues(return_idx=False)
    for uq in set(uqres):
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


def _fix_backbone_amide_h_names(mol):
    """Rename a misnamed backbone amide H to ``H`` on canonical amino acid
    residues. Some PDBs (e.g. cyclic peptides like 5VAV) name the lone
    peptide-bonded amide hydrogen ``H1`` instead of ``H``, which breaks
    PDB2PQR's reference-template walk with
    "Found gap in biomolecule structure for atom ... H1". We identify the
    amide H geometrically: among hydrogens in the residue, the unique one
    bonded to backbone N (within typical N-H bond distance) is the amide
    H. A true N-terminal NH3+ has multiple Hs bonded to N and is left
    untouched.
    """
    from moleculekit.residues import PROTEIN_RESIDUES

    canonical = {rr.resname for rr in PROTEIN_RESIDUES} | {
        v for rr in PROTEIN_RESIDUES for v in rr.resname_variants
    }
    nh_min, nh_max = 0.8, 1.2  # typical N-H bond length ~1.01 Å
    coords = mol.coords[:, :, 0]
    uqres = mol.getResidues(return_idx=False)
    for uq in set(uqres):
        resatm = uqres == uq
        resname = mol.resname[resatm][0]
        if resname not in canonical or resname == "PRO":
            continue
        res_idx = np.where(resatm)[0]
        n_atoms = res_idx[mol.name[res_idx] == "N"]
        if not len(n_atoms):
            continue
        n_idx = int(n_atoms[0])
        h_idx = res_idx[mol.element[res_idx] == "H"]
        if not len(h_idx):
            continue
        d = np.linalg.norm(coords[h_idx] - coords[n_idx], axis=1)
        bonded = h_idx[(d >= nh_min) & (d <= nh_max)]
        if len(bonded) != 1:
            continue
        target = int(bonded[0])
        old = mol.name[target]
        if old == "H":
            continue
        mol.name[target] = "H"
        logger.info(
            f"Renamed backbone amide {old} to H on {resname} "
            f"{mol.resid[n_idx]} {mol.chain[n_idx]}"
        )


def _capture_formal_charges(mol, detect_specs):
    """Capture the non-zero per-atom formal charges of every residue in
    ``detect_specs`` as a list of ``(UniqueAtomID, charge)`` tuples, so
    they survive the PDB2PQR roundtrip - which rebuilds the molecule with
    ``formalcharge`` reset to zero.

    Scoped to spec residues deliberately: those are the non-canonical
    residues and renamed canonical anchors the caller set up via
    :meth:`Molecule.templateResidueFromSmiles`, which PDB2PQR leaves
    alone. Plain canonical residues are excluded - PDB2PQR may
    legitimately change their protonation, and downstream
    parameterization does not need their formal charges. Captured as
    :class:`UniqueAtomID` so resname / index changes through PDB2PQR do
    not break the round-trip.
    """
    from moleculekit.molecule import UniqueAtomID

    if not detect_specs:
        return []
    in_spec = np.zeros(mol.numAtoms, dtype=bool)
    for spec in detect_specs:
        r = spec.residue
        in_spec |= (
            (mol.segid == str(r.segid))
            & (mol.chain == str(r.chain))
            & (mol.resid == int(r.resid))
            & (mol.insertion == str(r.insertion))
        )
    out = []
    for i in np.where(in_spec)[0]:
        charge = int(mol.formalcharge[i])
        if charge != 0:
            out.append((UniqueAtomID.fromMolecule(mol, idx=int(i)), charge))
    return out


def _restore_formal_charges(mol, preserved_charges):
    """Re-resolve each captured atom against ``mol`` and restore its
    formal charge. Uses the same relaxed lookup as :func:`_restore_bonds`
    so atoms in residues renamed by systemPrepare still resolve."""
    for uaid, charge in preserved_charges:
        idx = _find_atom_relaxed(mol, uaid)
        if idx is not None:
            mol.formalcharge[idx] = charge


def _capture_bonds(mol, detect_specs):
    """Walk ``mol.bonds`` and return a list of
    ``(UniqueAtomID, UniqueAtomID, is_h_bond, bondtype)`` tuples for
    every bond touching a residue that needs preservation across the
    PDB2PQR roundtrip (which strips ``mol.bonds`` entirely).

    A residue needs preservation when either:

    * it appears in ``detect_specs`` (a non-canonical residue, or a
      canonical AA at a non-peptide junction like a CYS-CYS disulfide,
      ASN N-glycosylation, etc.), or
    * its resname is not in
      :data:`moleculekit.tools.nonstandard_residues._CANONICAL_RESNAMES`
      (covers NCAAs / ligands the caller templated themselves but
      bypassed spec auto-detection via ``detect_specs=[]``).

    Bonds where both endpoints live in canonical, non-spec residues are
    dropped: downstream builders (tLeap, OpenMM, ...) rebuild them from
    FF templates more correctly than a name-based restore can. PDB2PQR
    also renames some canonical atoms (RNA ``OP1``/``OP2`` -> ``O1P``/
    ``O2P``), which would make a blanket restore noisily fail.

    Captured as ``UniqueAtomID`` pairs so resname / index changes
    through PDB2PQR don't break the round-trip. ``is_h_bond`` lets the
    restore step silently drop bonds whose H endpoint pdb2pqr removed
    (e.g. when re-protonating), while still warning when a heavy-heavy
    bond breaks. Bondtype is carried through so single/double/aromatic
    order is preserved.
    """
    from moleculekit.molecule import UniqueAtomID
    from moleculekit.tools.nonstandard_residues import _CANONICAL_RESNAMES

    if mol.bonds is None or len(mol.bonds) == 0:
        return []
    if mol.bondtype is None or len(mol.bondtype) != len(mol.bonds):
        raise RuntimeError(
            f"mol.bondtype has length "
            f"{0 if mol.bondtype is None else len(mol.bondtype)} but "
            f"mol.bonds has length {len(mol.bonds)}. They must be the same. "
            "Use ``mol.guessBonds()`` to populate both consistently, or "
            "assign ``mol.bonds`` and ``mol.bondtype`` together."
        )

    # Per-atom flag: True if the atom belongs to a residue whose bonds
    # must survive the roundtrip. Match spec residues by
    # (segid, chain, resid, insertion); resname is excluded because
    # ``_template_renamed_canonical_residues`` may have already mutated
    # ``mol.resname`` while leaving ``spec.residue.resname`` untouched.
    needs_capture = ~np.isin(mol.resname, list(_CANONICAL_RESNAMES))
    if detect_specs:
        for spec in detect_specs:
            r = spec.residue
            needs_capture |= (
                (mol.segid == str(r.segid))
                & (mol.chain == str(r.chain))
                & (mol.resid == int(r.resid))
                & (mol.insertion == str(r.insertion))
            )

    out = []
    for i, (a, b) in enumerate(mol.bonds):
        a, b = int(a), int(b)
        if not (needs_capture[a] or needs_capture[b]):
            continue
        is_h_bond = "H" in mol.element[[a, b]]
        out.append(
            (
                UniqueAtomID.fromMolecule(mol, idx=a),
                UniqueAtomID.fromMolecule(mol, idx=b),
                bool(is_h_bond),
                str(mol.bondtype[i]),
            )
        )
    return out


def _restore_bonds(mol, preserved_pairs):
    """Re-resolve each captured bond against ``mol`` and add it back.
    Resname can change between input and post-systemPrepare mol (e.g.
    CYS -> CYX after disulfide detection); we use a relaxed lookup that
    ignores resname so heavy-atom bonds still resolve. If an endpoint
    can't be resolved and the bond involved a hydrogen, drop silently
    (pdb2pqr strips and re-adds hydrogens). For broken heavy-heavy
    bonds we warn since that indicates real connectivity loss. The
    original bondtype is preserved.
    """
    new_bonds = []
    new_btypes = []
    for uaid_a, uaid_b, is_h_bond, btype in preserved_pairs:
        a = _find_atom_relaxed(mol, uaid_a)
        b = _find_atom_relaxed(mol, uaid_b)
        if a is None or b is None:
            if not is_h_bond:
                logger.warning(
                    "systemPrepare: failed to restore covalent bond %s - %s "
                    "(endpoint not found in prepared mol).",
                    uaid_a,
                    uaid_b,
                )
            continue
        new_bonds.append([a, b])
        new_btypes.append(btype)
    if not new_bonds:
        return
    arr = np.asarray(new_bonds, dtype=np.uint32)
    btype_arr = np.array(new_btypes, dtype=object)
    if mol.bonds is None or len(mol.bonds) == 0:
        mol.bonds = arr
        mol.bondtype = btype_arr
    else:
        mol.bonds = np.vstack([mol.bonds, arr])
        mol.bondtype = np.hstack([mol.bondtype, btype_arr])


def _find_atom_relaxed(mol, uaid):
    """Locate an atom in ``mol`` matching ``uaid`` by
    segid+chain+resid+insertion+name. Resname and altloc are intentionally
    ignored so that residues renamed by systemPrepare (CYS -> CYX, etc.)
    still resolve. Returns the atom index or ``None`` if no exact unique
    match exists."""
    mask = mol.name == uaid.name
    if uaid.segid:
        mask &= mol.segid == uaid.segid
    if uaid.chain:
        mask &= mol.chain == uaid.chain
    mask &= mol.resid == int(uaid.resid)
    if uaid.insertion:
        mask &= mol.insertion == uaid.insertion
    idxs = np.where(mask)[0]
    if len(idxs) == 1:
        return int(idxs[0])
    return None


def _restore_termini_bonds(mol):
    """Re-attach the terminal atoms PDB2PQR adds after ``_capture_bonds``
    has already run. These are the only post-capture atoms we care
    about keeping bonded in the prepared mol because they're the only
    ones that participate in cluster parameterization (where unbonded
    atoms type as ``DU`` and break antechamber). Bonds in regular
    residues whose Hs PDB2PQR added or renamed are intentionally left
    unbonded - tLeap resolves those by name via the ff14SB templates.

    Patches handled (all use the standard AMBER names):
        - CTERM:        OXT -> C
        - NEUTRAL-CTERM: HO  -> OXT (acid hydrogen, if present)
        - NTERM:        H2, H3 -> N (extra NH3+ hydrogens; H is the
                                     pre-existing amide H, already
                                     bonded by ``_restore_bonds``)
    """
    if mol.numAtoms == 0 or mol.element is None:
        return

    if mol.bonds is None or len(mol.bonds) == 0:
        bonded_atoms = set()
    else:
        bonded_atoms = set(int(i) for i in np.asarray(mol.bonds).ravel())

    uqres = mol.getResidues(
        fields=("segid", "chain", "resid", "insertion"), return_idx=False
    )

    # Each entry: (orphan_name, partner_name) - bond the first to the
    # second within the same residue, but only if the orphan is
    # currently unbonded. ``HO`` / ``HXT`` are different conventions
    # for the same acidic H on the C-terminal carboxyl OXT.
    patches = [
        ("OXT", "C"),
        ("HO", "OXT"),
        ("HXT", "OXT"),
        ("H2", "N"),
        ("H3", "N"),
    ]

    new_bonds = []
    for orphan_name, partner_name in patches:
        orphan_idxs = np.where(mol.name == orphan_name)[0]
        if not len(orphan_idxs):
            continue
        for o_idx in orphan_idxs:
            o_idx = int(o_idx)
            if o_idx in bonded_atoms:
                continue
            partner_idx = np.where(
                (uqres == uqres[o_idx]) & (mol.name == partner_name)
            )[0]
            if len(partner_idx) != 1:
                continue
            new_bonds.append([int(partner_idx[0]), o_idx])

    if not new_bonds:
        return

    arr = np.asarray(new_bonds, dtype=np.uint32)
    btype_arr = np.array(["1"] * len(new_bonds), dtype=object)
    if mol.bonds is None or len(mol.bonds) == 0:
        mol.bonds = arr
        mol.bondtype = btype_arr
    else:
        mol.bonds = np.vstack([mol.bonds, arr])
        mol.bondtype = np.hstack([mol.bondtype, btype_arr])


def _apply_terminal_formal_charges(mol, detect_specs):
    """Set ``formalcharge`` on the terminal atoms PDB2PQR adds to
    chain-resident spec residues.

    PDB2PQR's CTERM / NTERM patches add the terminal heavy atom or
    extra hydrogens but always leave ``formalcharge`` at zero (PDB2PQR
    discards formal charges entirely). After ``_restore_termini_bonds``
    the bonds are back, so we can infer ionisation purely from the H
    count on the backbone ``N`` and the terminal ``OXT``:

      - 3 H on ``N``     -> NH3+ (charged NTERM patch)
      - <3 H on ``N``    -> neutral NH2 (NEUTRAL-NTERM patch)
      - 0 H on ``OXT``   -> COO-  (charged CTERM patch)
      - >=1 H on ``OXT`` -> neutral COOH (NEUTRAL-CTERM patch)

    Only applied to :class:`ChainResidueSpec` residues - those are the
    ones we own and that downstream cluster parameterization sees.
    PDB2PQR-driven protonation of plain canonical residues is left to
    PDB2PQR.
    """
    from moleculekit.tools.nonstandard_residues import ChainResidueSpec

    def _count_h_neighbors(atom_idx):
        return sum(
            1 for nb in mol.getNeighbors(atom_idx) if mol.element[nb] == "H"
        )

    for spec in detect_specs:
        if not isinstance(spec, ChainResidueSpec):
            continue
        if not (spec.is_n_term or spec.is_c_term):
            continue
        rid = spec.residue
        current_resname = spec.new_resname or spec.resname
        res_mask = (
            (mol.resname == str(current_resname))
            & (mol.segid == str(rid.segid))
            & (mol.chain == str(rid.chain))
            & (mol.resid == int(rid.resid))
            & (mol.insertion == str(rid.insertion))
        )
        res_idx = np.where(res_mask)[0]
        if len(res_idx) == 0:
            continue
        res_names = mol.name[res_idx]

        if spec.is_n_term:
            n_idx = res_idx[res_names == "N"]
            if len(n_idx) == 1 and _count_h_neighbors(int(n_idx[0])) == 3:
                mol.formalcharge[n_idx[0]] = 1

        if spec.is_c_term:
            oxt_idx = res_idx[res_names == "OXT"]
            if len(oxt_idx) == 1 and _count_h_neighbors(int(oxt_idx[0])) == 0:
                mol.formalcharge[oxt_idx[0]] = -1


def _assert_specs_templated(mol, detect_specs):
    """Raise if any chain-resident non-canonical residue in ``detect_specs``
    has not been templated by the caller. NCAAs must be templated via
    ``mol.templateResidueFromSmiles(..., addHs=True)`` before they're
    handed to ``systemPrepare`` — otherwise PDB2PQR matches them to
    their nearest canonical (e.g. ALC -> LEU, NLE -> LYS) and adds an H
    that ``_restore_termini_bonds`` can't re-attach, failing late in
    ``_assert_specs_bonded`` with a misleading "renamed canonical
    residues have unbonded atoms" message.

    A templated NCAA has both hydrogens AND a fully-bonded heavy-atom
    skeleton. We check both: heavy-atom H presence catches the
    no-template case; the all-atoms-bonded check additionally catches
    partial inputs where the user added Hs by hand but skipped bonds.

    Canonical AAs at a junction (``resname in PROTEIN_RESNAMES``, e.g.
    a CYS that becomes CYX) are re-templated inside ``systemPrepare`` by
    ``_template_renamed_canonical_residues`` and so are skipped here.
    """
    from moleculekit.tools.nonstandard_residues import (
        ChainResidueSpec, PROTEIN_RESNAMES,
    )

    if mol.bonds is None or len(mol.bonds) == 0:
        bonded = set()
    else:
        bonded = set(int(i) for i in np.asarray(mol.bonds).ravel())

    bad = []
    for spec in detect_specs:
        if not isinstance(spec, ChainResidueSpec):
            continue
        if spec.resname in PROTEIN_RESNAMES:
            continue
        rid = spec.residue
        res_mask = (
            (mol.resname == str(spec.resname))
            & (mol.segid == str(rid.segid))
            & (mol.chain == str(rid.chain))
            & (mol.resid == int(rid.resid))
            & (mol.insertion == str(rid.insertion))
        )
        idxs = np.where(res_mask)[0]
        if len(idxs) == 0:
            continue
        has_h = bool((mol.element[idxs] == "H").any())
        all_bonded = all(int(i) in bonded for i in idxs)
        if not has_h or not all_bonded:
            bad.append(
                f"{spec.resname}{rid.resid}{rid.insertion}:{rid.chain}"
            )
    if bad:
        raise RuntimeError(
            f"systemPrepare: chain-resident non-canonical residue(s) "
            f"{bad} are in detect_specs but have not been templated "
            f"(missing hydrogens or unbonded atoms). Call "
            f"mol.templateResidueFromSmiles(<sel>, <smiles>, addHs=True) "
            f"for each NCAA before passing it via detect_specs, so its "
            f"bonds and hydrogens are well-defined before the PDB2PQR "
            f"roundtrip."
        )


def _assert_specs_bonded(mol, detect_specs):
    """Defensive check: every atom in a custom-renamed canonical residue
    must be bonded after the PDB2PQR roundtrip + ``_restore_termini_bonds``.
    These residues end up in cluster parameterization (antechamber +
    parmchk2) where an unbonded atom types as ``DU`` and crashes the
    pipeline.

    Residues renamed to a PDB2PQR-known variant (``CYX``, ``LYN``,
    ``HID`` ...) are excluded: ff14SB has those templates natively, so
    tLeap handles their bonds via its own templates plus the
    ``bond`` / ``loadAmberPrep`` glue - they never reach cluster
    parameterization and orphan atoms here are harmless.

    If PDB2PQR ever applies a new patch to a custom-renamed residue
    that adds atoms with names we don't handle (the current
    ``_restore_termini_bonds`` covers OXT, HO, H2, H3), this assert
    flags it loudly instead of letting the bug ship to the cluster
    step.
    """
    from moleculekit.tools.nonstandard_residues import ChainResidueSpec

    if mol.bonds is None or len(mol.bonds) == 0:
        bonded = set()
    else:
        bonded = set(int(i) for i in np.asarray(mol.bonds).ravel())

    bad = []
    for spec in detect_specs:
        if not isinstance(spec, ChainResidueSpec):
            continue
        # PDB2PQR-known variants (CYX/LYN/HID/...) are handled by ff14SB
        # natively; their orphan atoms here don't reach cluster
        # parameterization and are harmless.
        if (
            spec.new_resname
            and str(spec.new_resname) in _PDB2PQR_KNOWN_VARIANTS
        ):
            continue
        rid = spec.residue
        current_resname = spec.new_resname or spec.resname
        res_mask = (
            (mol.resname == str(current_resname))
            & (mol.segid == str(rid.segid))
            & (mol.chain == str(rid.chain))
            & (mol.resid == int(rid.resid))
            & (mol.insertion == str(rid.insertion))
        )
        for idx in np.where(res_mask)[0]:
            if int(idx) not in bonded:
                bad.append(
                    f"{current_resname}{rid.resid}{rid.insertion}:"
                    f"{rid.chain}:{mol.name[idx]}"
                )
    if bad:
        raise RuntimeError(
            f"systemPrepare: renamed canonical residues have unbonded "
            f"atoms after the PDB2PQR roundtrip: {bad}. "
            f"_restore_termini_bonds currently handles only the standard "
            f"terminal patch atoms (OXT, HO, H2, H3). If PDB2PQR has "
            f"started applying a new patch to these residues, extend the "
            f"helper with the new atom -> partner pairs."
        )


def _apply_detect_spec_renames(mol, detect_specs):
    """Apply ``ChainResidueSpec.new_resname`` to ``mol`` in place. This
    is a safety net: ``_template_renamed_canonical_residues`` already
    applies the same rename pre-PDB2PQR. The post-PDB2PQR rename here
    catches specs whose rename was skipped upstream (or whose residue
    was reconstructed by PDB2PQR under the original name).

    Matching residues by ``(segid, chain, resid, insertion)``."""
    from moleculekit.tools.nonstandard_residues import ChainResidueSpec

    for spec in detect_specs:
        if not isinstance(spec, ChainResidueSpec):
            continue
        if not spec.new_resname:
            continue
        rid = spec.residue
        res_mask = (
            (mol.segid == str(rid.segid))
            & (mol.chain == str(rid.chain))
            & (mol.resid == int(rid.resid))
            & (mol.insertion == str(rid.insertion))
        )
        if res_mask.any():
            mol.resname[res_mask] = str(spec.new_resname)


def systemPrepare(
    mol_in: Molecule,
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
    _logger_level="ERROR",
    titrate=None,
    detect_specs=None,
    restore_missing_sidechains=False,
):
    """Prepare a molecular system by adding hydrogens, assigning protonation
    states, and optimizing the H-bond network.

    Wraps PDB2PQR + PROPKA. Protein and nucleic residues are protonated and
    optimized; non-protein, non-nucleic residues contribute to the pKa
    calculation but are not themselves protonated or optimized here. The
    input molecule is not mutated — a new :class:`Molecule` is returned.

    The returned molecule uses the following protonation-aware resnames:

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

    Notes
    -----

    What this function does:

    - Assigns protonation states via PROPKA.
    - Flips Asn/Gln/His sidechains to optimize the H-bond network.
    - Debumps clashes introduced by added hydrogens.
    - Adds missing heavy atoms and hydrogens via PDB2PQR's force-field
      templates.
    - Detects non-standard residues and sidechain crosslinks (disulfides,
      metal-coordinating Cys/His/Tyr, isopeptides, glycosylations, ...)
      and renames the affected residues to canonical variants (``CYX``,
      ``XX#`` buckets) so PDB2PQR's templates apply and the bonds are
      preserved. See ``detect_specs`` below.

    If ``hydrophobic_thickness`` is set to a positive value ``2*h``, a
    warning is produced for titratable residues with ``-h < z < h`` that
    are buried by less than 75%. The heuristic is crude (it assumes the
    protein is aligned with the membrane centered at ``z=0``) and the
    "buried fraction" estimate (from PROPKA) is approximate, so cavity-
    facing residues may appear solvent-exposed regardless of ``z``.

    Parameters
    ----------
    mol_in : moleculekit.molecule.Molecule
        Input molecule. Not mutated — an internal copy is taken on the
        first frame only.
    titration : bool
        If True, run PROPKA to assign titration states. If False, just add
        and optimize hydrogens at the input resnames' default protonation.
    pH : float
        Solution pH used by PROPKA to pick titration states. Default 7.4.
    force_protonation : list of tuple[str, str], optional
        Force specific protonation states on individual residues. Each
        entry is ``(atomselection, resname)``, e.g.
        ``[("protein and resid 40", "HID")]`` forces residue 40 to HID.
        Atom selections use VMD syntax.
    no_opt : list of str, optional
        Atom selections of residues to exclude from H-bond / sidechain
        flip optimization. Use this when an optimizer flip would degrade
        a known-good structure (e.g. a residue in a metal site).
    no_prot : list of str, optional
        Atom selections of residues to exclude from hydrogen addition.
    no_titr : list of str, optional
        Atom selections of residues to exclude from titration.
    hold_nonpeptidic_bonds : bool
        When True (default), protein residues that are covalently bonded
        to non-protein partners are automatically added to ``no_opt``,
        ``no_prot`` and ``no_titr`` so their bonds aren't broken and
        their boundary atoms aren't over-protonated. Set to False to let
        PDB2PQR/PROPKA process them ignoring the non-peptide bond.
    verbose : bool
        If False, demote this module's logger to WARNING for the call.
    return_details : bool
        If True, additionally return a pandas DataFrame with per-residue
        protonation / pKa info.
    hydrophobic_thickness : float, optional
        Membrane thickness ``2*h`` in Angstrom (None for globular
        proteins). Triggers a warning for titratable residues that fall
        inside the bilayer and are not well-buried (see Notes).
    plot_pka : str, optional
        Path to a ``.png`` file. When set, writes a titration diagram for
        the system's titratable residues.
    titrate : list[str], optional
        Restrict titration to a subset of the canonical AAs in the table
        above (e.g. ``["HIS", "ARG"]``). All other AAs in the table are
        added to ``no_titr``. When None (default), every AA in the table
        is titratable.
    detect_specs : list[PerResidueSpec], optional
        Per-residue specs from
        :func:`moleculekit.tools.nonstandard_residues.detectNonStandardResidues`.
        When left as ``None`` (the default), ``systemPrepare`` runs
        :func:`detectNonStandardResidues` itself so canonical residues at
        non-peptide junctions are renamed (e.g. CYS-Cys disulfide ->
        ``CYX``, a TYR coordinating a heme Fe -> a ``TY#`` bucket) and
        their sidechains re-templated before PDB2PQR runs. Pass an
        explicit list to bypass the auto-detection; pass
        ``detect_specs=[]`` to skip non-standard residue handling
        entirely. The applied specs are always returned so they can be
        forwarded to the downstream parameterizer / builder.
    restore_missing_sidechains : bool, optional
        If True, sidechain atoms removed by PDB2PQR are templated back in
        after preparation. Specifically, canonical protein residues whose
        entire heavy-atom sidechain is absent (only backbone + CB remain)
        are reconstructed using the Dunbrack rotamer library before
        PDB2PQR runs, so PDB2PQR's 10 %-missing-atom threshold does not
        reject the structure on a partial input. Default False.

    Returns
    -------
    mol_out : moleculekit.molecule.Molecule
        The protonated and optimized molecule.
    detect_specs : list[PerResidueSpec]
        The non-standard-residue specs that were applied (the caller's
        ``detect_specs`` argument if supplied, otherwise the auto-detected
        list). Empty when ``detect_specs=[]`` was passed in.
    details : pandas.DataFrame
        Per-residue protonation states, pKas and buried fractions.
        Returned only when ``return_details=True``.

    Raises
    ------
    ImportError
        If ``pdb2pqr`` is not installed.

    Examples
    --------
    >>> tryp = Molecule('3PTB')
    >>> tryp_op, specs, df = systemPrepare(tryp, return_details=True)
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

    Acidic pH:

    >>> tryp_op, specs = systemPrepare(tryp, pH=1.0)
    >>> tryp_op.write('/tmp/3PTB_pH1.pdb')

    Freeze residues 36 and 49 in place:

    >>> tryp_op, specs = systemPrepare(tryp, no_opt=["protein and resid 36", "chain A and resid 49"])

    Disable protonation on residue 32:

    >>> tryp_op, specs = systemPrepare(tryp, no_prot=["protein and resid 32",])

    Disable titration *and* protonation on residue 32:

    >>> tryp_op, specs = systemPrepare(tryp, no_titr=["protein and resid 32",], no_prot=["protein and resid 32",])

    Force residue 40 to HIE and 57 to HIP:

    >>> tryp_op, specs = systemPrepare(tryp, force_protonation=[("protein and resid 40", "HIE"), ("protein and resid 57", "HIP")])

    Skip non-standard residue detection entirely (legacy behavior):

    >>> tryp_op, specs = systemPrepare(tryp, detect_specs=[])

    Forward ``specs`` to a downstream builder for non-canonical residues
    (NCAAs, sidechain crosslinks, metal-coordinating residues):

    >>> # from htmd.builder.nonstandard import parameterizeFromSpecs
    >>> # parameterizeFromSpecs(specs, tryp_op, outdir="./params")

    See also
    --------
    moleculekit.tools.nonstandard_residues.detectNonStandardResidues
    moleculekit.molecule.Molecule.templateResidueFromSmiles
    moleculekit.molecule.Molecule.mutateResidue
    moleculekit.tools.modelling.model_gaps
    moleculekit.tools.autosegment.autoSegment
    """
    try:
        from pdb2pqr.config import VERSION
    except ImportError:
        raise ImportError(
            "pdb2pqr not installed. To use the system preparation features please do `conda install pdb2pqr -c acellera -c conda-forge`"
        )
    from moleculekit.tools.preparation_customres import _get_custom_ff
    from moleculekit.util import ensurelist

    # We don't want to modify the original molecule in place so we create a new molecule
    mol_in = mol_in.copy(frames=[0])

    # Auto-detect non-standard residue specs when the caller didn't supply
    # them, so the canonical-anchor renames + SMILES re-template path fires
    # for cases the caller didn't know about (TYR-O coordinating a heme Fe,
    # CYS-S thiolate to a metal centre, ...). Pass an explicit list (or
    # ``detect_specs=[]``) to bypass this.
    if detect_specs is None:
        from moleculekit.tools.nonstandard_residues import (
            detectNonStandardResidues,
        )
        detect_specs = detectNonStandardResidues(mol_in)

    # Restore canonical residues whose entire sidechain is missing
    # using moleculekit's Dunbrack-rotamer mutator, so PDB2PQR's
    # 10%-missing-atom threshold does not reject the structure on a
    # partial input. Off by default - users who want it pass
    # ``restore_missing_sidechains=True``. Runs BEFORE templating /
    # bond capture so the reconstructed atoms participate in
    # everything downstream.
    if restore_missing_sidechains:
        _restore_trimmed_canonical_sidechains(mol_in, detect_specs)

    # Rename + re-template canonical AAs with sidechain crosslinks
    # BEFORE PDB2PQR sees them. Renamed residues either land on a
    # PDB2PQR-known variant (CYX, LYN, ...) or get a custom resname
    # whose hydrogens come from templateResidueFromSmiles.
    if detect_specs:
        _template_renamed_canonical_residues(mol_in, detect_specs)
        # Canonical-AA junctions are now templated by us above; anything
        # still untemplated is a caller-supplied NCAA that wasn't fed
        # through templateResidueFromSmiles. Fail fast here rather than
        # let PDB2PQR match it to a near-canonical and crash late in
        # _assert_specs_bonded with a misleading message.
        _assert_specs_templated(mol_in, detect_specs)

    # Canonicalize the rdkit-generic H names (H1, H2, ...) produced by
    # templateResidueFromSmiles to AMBER conventions (H on N, HA on CA)
    # BEFORE capturing bonds. If we capture first, the captured atom
    # names include rdkit's H3 / H4 / ..., and PDB2PQR may later create
    # an atom with the same generic name (e.g. the third N-terminal
    # NH3+ H is named "H3" by PDB2PQR) at a different position. The
    # name-based bond restore then resolves the captured bond to the
    # wrong atom.
    _canonicalize_ncaa_h_names(mol_in, detect_specs)

    # Capture bonds touching spec / non-canonical residues before the
    # PDB2PQR roundtrip strips them, so we can restore them on mol_out
    # at the end. This preserves the connectivity downstream builders
    # can't regenerate (disulfides, glycosidic bonds, stapled-peptide
    # sidechain bonds, scaffolded-peptide thioethers, intra-ligand
    # bonds, ...). Bonds entirely inside canonical, non-spec residues
    # are intentionally dropped: builders rebuild them more correctly
    # from FF templates, and PDB2PQR can rename canonical atoms (RNA
    # OP1/OP2 -> O1P/O2P), which would make a blanket restore fail.
    _preserved_bonds = _capture_bonds(mol_in, detect_specs)
    _preserved_formal_charges = _capture_formal_charges(mol_in, detect_specs)

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

    check_backbone(mol_in)

    if no_opt is not None:
        no_opt = ensurelist(no_opt)
    if no_prot is not None:
        no_prot = ensurelist(no_prot)
    if no_titr is not None:
        no_titr = ensurelist(no_titr)
    if titrate is not None:
        titrate = set(ensurelist(titrate))
        all_aas = {"ASP", "CYS", "GLU", "HIS", "LYS", "TYR", "ARG"}
        possible_titrations = {
            "ASP": ("ASP", "ASH"),
            "CYS": ("CYS", "CYX", "CYM"),
            "GLU": ("GLU", "GLH"),
            "HIS": ("HIS", "HID", "HIE", "HIP"),
            "LYS": ("LYS", "LYN"),
            "TYR": ("TYR", "TYM"),
            "ARG": ("ARG", "AR0"),
        }
        other = all_aas - titrate
        no_titr = no_titr or []
        for aa in other:
            _idx = mol_in.atomselect(
                f"resname {' '.join(possible_titrations[aa])}", indexes=True
            )
            residues = [
                (mol_in.resid[i], mol_in.chain[i], mol_in.insertion[i]) for i in _idx
            ]
            residues = list(set(residues))
            for res in residues:
                _sel = f"resid {res[0]} and chain {res[1]}"
                if len(res[2]):
                    _sel += f" and insertion {res[2]}"
                no_titr.append(_sel)

    mol_in = mol_in.copy()

    _warn_if_contains_DUM(mol_in)

    mol_in = _check_chain_and_segid(mol_in, verbose)
    mol_orig = mol_in.copy()

    _prepare_nucleics(mol_in)
    _fix_protonation_resnames(mol_in)
    _fix_backbone_amide_h_names(mol_in)

    definition, forcefield = _get_custom_ff()
    definition, forcefield = _generate_nonstandard_residues_ff(
        mol_in, definition, forcefield, detect_specs=detect_specs
    )

    nonpept = []
    if hold_nonpeptidic_bonds:
        nonpept = _detect_nonpeptidic_bonds(mol_in)

    if force_protonation is not None:
        for sel, resn in force_protonation:
            # Remove side-chain hydrogens
            mol_in.remove(f"({sel}) and not backbone and element H", _logger=False)
            # Rename to desired protonation state
            mol_in.set("resname", resn, sel=sel)

    _no_opt, _no_prot, _no_titr, _frozen = _get_hold_residues(
        mol_in, no_opt, no_prot, no_titr, force_protonation, nonpept
    )
    _check_frozen_histidines(mol_in, _no_prot + [key for key, _, _ in _frozen])

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
    mol_out.boxangles = mol_in.boxangles
    mol_out.time = mol_in.time
    mol_out.step = mol_in.step
    mol_out.fileloc = mol_in.fileloc
    _fixup_water_names(mol_out)

    # Restore the bonds and formal charges we captured before the
    # PDB2PQR roundtrip stripped / zeroed them.
    if _preserved_bonds:
        _restore_bonds(mol_out, _preserved_bonds)
    if _preserved_formal_charges:
        _restore_formal_charges(mol_out, _preserved_formal_charges)

    # PDB2PQR's CTERM / NEUTRAL-CTERM / NTERM / NEUTRAL-NTERM patches
    # are the only patches we expect to add atoms to renamed canonical
    # residues; their additions (OXT, HO, H2, H3) are all reattached
    # by name here. Anything else is unhandled and breaks downstream
    # cluster parameterization (orphan atoms type as ``DU`` in
    # antechamber); the assert below catches that case.
    _restore_termini_bonds(mol_out)
    if detect_specs:
        _apply_terminal_formal_charges(mol_out, detect_specs)

    if detect_specs:
        _assert_specs_bonded(mol_out, detect_specs)

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

    if detect_specs:
        _apply_detect_spec_renames(mol_out, detect_specs)

    logger.setLevel(old_level)

    if return_details:
        return mol_out, detect_specs, df
    return mol_out, detect_specs


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
