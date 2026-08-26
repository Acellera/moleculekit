# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
"""Reconcile PROPKA's view of a structure with what the caller already decided.

PROPKA predicts pKas from an empirical model that reads a structure the way a
PDB file describes it: elements come from the atom-name columns, chemistry is
perceived from coordinates, and a residue it does not recognise is typed by
geometry alone. That is the right set of assumptions for a bare crystal
structure and the wrong one here, because by the time :mod:`preparation` calls
PROPKA the caller has usually said more than the file does.

A residue templated with
:meth:`~moleculekit.molecule.Molecule.templateResidueFromSmiles` carries
explicit hydrogens, bond orders and formal charges. Those are assertions about
protonation, and re-deriving them from geometry throws them away. The helpers
here hand PROPKA what it cannot work out for itself:

  - which residues are non-canonical, so their sidechains are grouped as
    ligands rather than ignored (:func:`_noncanonical_sidechain_plan`,
    :func:`_mark_ligand_like_sidechains`)
  - the real element, where the two-letter symbol was misread from the
    atom-name column (:func:`_correct_propka_elements`)
  - SYBYL atom types derived from the recorded bond orders rather than from a
    distance threshold (:func:`_apply_sybyl_types`)
  - the input's own formal charge, in place of PROPKA's titration verdict
    (:func:`_apply_templated_formal_charges`)
  - metals, both the ones buried inside a cofactor residue that PROPKA's
    residue-name lookup cannot see, and the groups coordinating them
    (:func:`_add_cofactor_metal_groups`,
    :func:`_hold_metal_coordinated_groups`)
  - PDB2PQR's terminus decision, in place of the one PROPKA infers from the
    text (:func:`_clear_phantom_termini`)

:func:`_run_propka` is the entry point and applies these in the order PROPKA's
own setup requires.
"""

import logging

import numpy as np

from moleculekit.tools.mutate import BACKBONE_ATOMS as _MUTATE_BACKBONE_ATOMS
from moleculekit.tools.nonstandard_residues import (
    _bonded_atom_indices,
    _residue_is_templated,
)

logger = logging.getLogger(__name__)


def _propka_atom_key(atom):
    """``(chain, resid, insertion, atom_name)`` for a PROPKA atom."""
    return (
        str(atom.chain_id),
        int(atom.res_num),
        str(atom.icode).strip(),
        str(atom.name).strip(),
    )


def _pdb2pqr_terminus_decisions(biomolecule):
    """Report what PDB2PQR's ``set_termini`` decided about each residue.

    Returns ``(n_term, c_term, considered)``, sets of
    ``(resid, chain, insertion)`` keys. ``considered`` holds the residues
    ``set_termini`` actually judges: it only assigns termini to ``aa.Amino``
    residues, so the absence of a flag on anything else is not a decision and
    must not be read as one. A templated non-canonical residue reaches PDB2PQR
    as an ``aa.Amino`` subclass and so is judged normally.
    """
    from pdb2pqr import aa

    n_term = set()
    c_term = set()
    considered = set()
    for residue in biomolecule.residues:
        if not isinstance(residue, aa.Amino):
            continue
        key = (
            int(residue.res_seq),
            str(residue.chain_id),
            str(residue.ins_code).strip(),
        )
        considered.add(key)
        if getattr(residue, "is_n_term", 0):
            n_term.add(key)
        if getattr(residue, "is_c_term", 0):
            c_term.add(key)
    return n_term, c_term, considered


def _clear_phantom_termini(molecule, biomolecule):
    """Drop PROPKA terminus flags that PDB2PQR did not itself assign.

    PROPKA infers termini textually from the PDB stream it is handed: the first
    ``ATOM`` residue after a ``TER`` becomes ``N+`` and an ``OXT`` becomes
    ``C-``. It therefore never sees ``set_termini``'s reasoning, neither its
    cyclic-chain distance guard nor the ``n_term_blocked`` / ``c_term_blocked``
    flags :func:`_stamp_non_termini` sets. Two consequences:

    * a head-to-tail cyclic peptide gets a spurious ``N+`` (charge +1, model
      pKa 8.0) on the very amide nitrogen that closes the ring;
    * PROPKA matches the N-terminal residue on the residue *number* alone and
      ignores the insertion code, so in chymotrypsin-numbered structures every
      residue sharing that number gets an ``N+`` (in 1A4W both ``ASP1A:L`` and
      the mid-chain ``CYS1:L``).

    Either way a fictitious cation shifts the predicted pKa of every titratable
    group near it. A terminus flag also makes PROPKA build ``NtermGroup`` /
    ``CtermGroup`` in preference to the backbone group, so the flags have to be
    reconciled before groups are extracted; clearing one lets PROPKA build the
    normal backbone group for that atom instead.

    Only clears, never adds: where PDB2PQR never judged a residue, PROPKA's own
    call is left alone. Returns the number of flags cleared.
    """
    n_term, c_term, considered = _pdb2pqr_terminus_decisions(biomolecule)
    cleared = 0
    for conformation in molecule.conformations.values():
        for atom in conformation.atoms:
            if not atom.terminal:
                continue
            key = (
                int(atom.res_num),
                str(atom.chain_id),
                str(atom.icode).strip(),
            )
            if key not in considered:
                continue
            allowed = n_term if atom.terminal == "N+" else c_term
            if key not in allowed:
                logger.debug(
                    f"Clearing PROPKA {atom.terminal} flag on "
                    f"{atom.res_name.strip()} {atom.res_num}{atom.chain_id}: "
                    "PDB2PQR did not assign that terminus."
                )
                atom.terminal = None
                cleared += 1
    return cleared


def _nucleic_residue_keys(biomolecule):
    """``(chain, resid, insertion)`` keys PDB2PQR parsed as nucleic acids.

    Needed because PDB2PQR renames RNA to its internal R-prefixed form
    (``A`` becomes ``RA``), which no canonical-resname table recognises. Nucleic
    chemistry is not what sidechain re-typing is for: PROPKA already models the
    bases through its own nucleotide model pKa values, and ligand-typing them
    invents groups that move the bases across pH 7.4. Matching on the parsed
    class rather than the resname keeps the exclusion naming-independent.
    """
    from pdb2pqr import na

    keys = set()
    for residue in biomolecule.residues:
        if isinstance(residue, na.Nucleic):
            keys.add(
                (
                    str(residue.chain_id),
                    int(residue.res_seq),
                    str(residue.ins_code).strip(),
                )
            )
    return keys


# Reuse the shared backbone definition, plus OXT: a C-terminal carboxyl oxygen
# is backbone too, and re-typing it would cost the residue its terminus group.
_BACKBONE_ATOM_NAMES = frozenset(_MUTATE_BACKBONE_ATOMS | {"OXT"})


def _noncanonical_sidechain_plan(src_mol, detect_specs):
    """Work out which sidechain atoms PROPKA should treat as ligand atoms, and
    which of them carry a formal charge the input already states.

    PROPKA can only reach a sidechain through its ligand-typing path, and that
    path requires ``atom.type == 'hetatm'``
    (``propka.group.is_ligand_group_by_groups``). Its protein path is no help:
    it matches sidechains by ``resname-atomname`` against a table holding just
    twelve resnames, the canonical residues with a titratable or polar
    sidechain. Anything else falls through every branch and only its backbone
    ``N`` and ``C`` are ever seen, so a phosphotyrosine's phosphate or a
    carboxyglutamate's two carboxylates contribute nothing whatsoever to
    neighbouring pKa values.

    Marked are the force-field-shipped modified amino acids (which
    ``detectNonStandardResidues`` deliberately reports no spec for, since they
    need no user template) and anything the force field does not recognise at
    all. The standard residues are left alone even where PROPKA models no
    sidechain group for them: re-typing a methionine or a phenylalanine would
    invite ligand typing to invent groups on chemistry PROPKA is right to
    ignore. Backbone atoms are always left alone, because
    ``is_protein_group`` needs ``atom.type == 'atom'`` to build the ``BBN`` /
    ``BBC`` groups through which a residue's backbone donates and accepts
    hydrogen bonds, and to build genuine terminus groups.

    Returns ``(mark, templated_charges)``. ``mark`` is a set of
    ``(chain, resid, insertion, atom_name)`` keys to re-type; keying on identity
    rather than resname keeps it valid across the detect-spec renames.
    ``templated_charges`` maps the same keys to ``mol.formalcharge``, and only for
    residues actually templated: an untemplated residue keeps the
    field's default of zero, which carries no information and must not be read
    as "neutral". Where the input states nothing, PROPKA's own group charges are
    the better answer, exactly as they are for metals.
    """
    from moleculekit.residues import MODIFIED_PROTEIN_RESIDUE_NAMES
    from moleculekit.tools.nonstandard_residues import (
        _CANONICAL_RESNAMES,
        getResidueMask,
        requiresTemplate,
    )

    mark = set()
    templated_charges = {}
    if src_mol is None:
        return mark, templated_charges

    bonded_idx = _bonded_atom_indices(src_mol)

    # ``requiresTemplate`` states an obligation; only _residue_is_templated
    # shows it was met. Without that an untemplated cofactor's default
    # ``formalcharge`` of zero reads as a decision that it is neutral.
    templated_atoms = np.zeros(src_mol.numAtoms, dtype=bool)
    for spec in detect_specs or ():
        if not requiresTemplate(spec):
            continue
        res_mask = getResidueMask(src_mol, spec)
        if res_mask.any() and _residue_is_templated(src_mol, res_mask, bonded_idx):
            templated_atoms |= res_mask

    for idx in range(src_mol.numAtoms):
        resname = str(src_mol.resname[idx]).strip()
        if resname not in MODIFIED_PROTEIN_RESIDUE_NAMES:
            if resname in _CANONICAL_RESNAMES:
                continue
        name = str(src_mol.name[idx]).strip()
        if name in _BACKBONE_ATOM_NAMES:
            continue
        key = (
            str(src_mol.chain[idx]),
            int(src_mol.resid[idx]),
            str(src_mol.insertion[idx]).strip(),
            name,
        )
        mark.add(key)
        if templated_atoms[idx]:
            templated_charges[key] = int(src_mol.formalcharge[idx])
    return mark, templated_charges


def _mark_ligand_like_sidechains(molecule, mark):
    """Re-type the sidechain atoms in ``mark`` so PROPKA's ligand typing sees
    them. Must run before bonding and protonation, since SYBYL typing (which
    every ligand-group branch keys on) is only applied to ligand atoms.
    """
    retyped = set()
    for conformation in molecule.conformations.values():
        for atom in conformation.atoms:
            key = _propka_atom_key(atom)
            if atom.type == "atom" and key in mark:
                atom.type = "hetatm"
                retyped.add(key)
    return retyped


def _correct_propka_elements(molecule, src_mol, mark):
    """Give PROPKA the real element for each re-typed atom.

    PROPKA derives an atom's element from the *atom-name* columns of the PDB
    line rather than the element columns (``propka.atom``), and truncates a
    four-character name to a single letter. A selenomethionine selenium named
    ``SE`` therefore becomes sulfur and a heme iron named ``FE`` becomes
    fluorine. Everything downstream inherits the mistake: the wrong covalent
    radius loses the Se-CE bond, so a selenoether looks like a one-neighbour
    terminal thiol, gets protonated, and is handed the ``SH`` model pKa of 10.0.

    Fixing the element instead of filtering the consequences also improves bond
    perception and SYBYL typing, which every ligand-group branch keys on.
    Scoped to the atoms sidechain re-typing exposes, since those are the ones
    whose chemistry PROPKA is now being asked to interpret; canonical backbone
    and sidechain names are single-letter and already correct.
    """
    if src_mol is None:
        return 0
    true_element = {}
    for idx in range(src_mol.numAtoms):
        true_element[
            (
                str(src_mol.chain[idx]),
                int(src_mol.resid[idx]),
                str(src_mol.insertion[idx]).strip(),
                str(src_mol.name[idx]).strip(),
            )
        ] = str(src_mol.element[idx]).strip().title()

    fixed = 0
    for conformation in molecule.conformations.values():
        for atom in conformation.atoms:
            key = _propka_atom_key(atom)
            if key not in mark:
                continue
            actual = true_element.get(key)
            if actual and actual != str(atom.element).strip().title():
                logger.debug(
                    f"Correcting PROPKA element for {atom.res_name.strip()} "
                    f"{atom.name.strip()}: {atom.element} -> {actual}"
                )
                atom.element = actual
                fixed += 1
    return fixed


def _apply_sybyl_types(molecule, src_mol, mark):
    """Hand PROPKA the SYBYL types moleculekit can derive, instead of its own.

    PROPKA's ligand-group detection keys entirely on SYBYL atom types, and it
    derives them geometrically because it has no bond orders to read: rings and
    aromaticity from coordinates, double bonds from a 1.3 A threshold calibrated
    for carbon. That misses a P=O at ~1.5 A, so every terminal phosphate oxygen
    becomes an independent titratable site and the group is over-charged by one.

    :func:`moleculekit.tools.sybyl.sybylTypes` reads the bond orders and lets
    RDKit perceive the rest. Setting ``sybyl_assigned`` makes PROPKA's own
    guessing skip the atom. Where the structure records no bond orders there is
    nothing better to offer, so PROPKA keeps its geometric answer.

    Returns the number of atoms typed.
    """
    from moleculekit.tools.sybyl import sybylTypes

    mask = np.zeros(src_mol.numAtoms, dtype=bool)
    keys = {}
    for idx in range(src_mol.numAtoms):
        key = (
            str(src_mol.chain[idx]),
            int(src_mol.resid[idx]),
            str(src_mol.insertion[idx]).strip(),
            str(src_mol.name[idx]).strip(),
        )
        if key in mark:
            mask[idx] = True
            keys[idx] = key

    if not mask.any():
        return 0
    typed = sybylTypes(src_mol, mask)

    by_key = {keys[idx]: sybyl for idx, sybyl in typed.items() if idx in keys}
    applied = 0
    for conformation in molecule.conformations.values():
        for atom in conformation.atoms:
            sybyl = by_key.get(_propka_atom_key(atom))
            if sybyl is not None:
                atom.sybyl_type = sybyl
                atom.sybyl_assigned = True
                applied += 1
    return applied


def _apply_templated_formal_charges(molecule, parameters, templated_charges, ph):
    """Replace PROPKA's titration verdict with the input's formal charge on groups
    built from templated residues.

    A group's charge centre is already the right place to put that charge:
    ``Group.setup_atoms`` averages the group's own atoms, so a carboxylate sits
    between its two oxygens and a guanidinium among its three nitrogens. Only
    the *state* needs replacing, and the mechanism for a fixed charge that still
    contributes an electrostatic term is the ion path: ``get_ions`` selects on
    ``group.residue_type`` being a key of ``parameters.ions``, and
    ``set_ion_determinants`` then shifts every titratable group within the
    Coulomb cutoff, screened by mutual burial.

    A group whose atoms sum to zero is left non-titratable and contributes only
    its volume, which is what a neutral templated sidechain should do.

    Groups PROPKA already agrees with are left strictly alone. Overriding one
    would not change its charge, and it would cost real information: pinning a
    group makes it non-titratable, which drops it out of
    ``get_titratable_groups`` - the very list ``set_determinants`` walks - so the
    ligand stops donating and accepting hydrogen bonds to its neighbours. On
    2QRV that silently removed the SAH cofactor's hydrogen bonds and moved two
    nearby acids by up to 2 pKa units. The override is for the cases PROPKA gets
    wrong or cannot see at all, such as a heme iron it has no model pKa for.
    """
    fixed = 0
    for conformation in molecule.conformations.values():
        for group in conformation.groups:
            atoms = set(group.interaction_atoms_for_acids) | set(
                group.interaction_atoms_for_bases
            )
            atoms.add(group.atom)
            keys = [_propka_atom_key(a) for a in atoms if a.element != "H"]
            if not keys or any(k not in templated_charges for k in keys):
                continue
            charge = sum(templated_charges[k] for k in keys)
            if group.titratable and group.model_pka_set:
                # what PROPKA's own titration would conclude at this pH
                charged_form = float(parameters.charge.get(group.type, 0.0))
                implied = 0.0
                if charged_form < 0 and ph > group.model_pka:
                    implied = charged_form
                elif charged_form > 0 and ph < group.model_pka:
                    implied = charged_form
                if implied == charge:
                    continue
            group.titratable = False
            group.charge = float(charge)
            if charge:
                ion_key = f"MK{group.label.strip().replace(' ', '_')}"[:20]
                parameters.ions[ion_key] = float(charge)
                group.residue_type = ion_key
            fixed += 1
    return fixed


# Fallback only: used when a structure carries no usable bond types, so there
# are no "mc" coordination markers to read. A first-shell contact to a metal is
# ~1.8-2.6 A; 2.8 leaves headroom without reaching a van der Waals contact.
_METAL_COORDINATION_FALLBACK_DIST = 2.8


def _metal_coordinated_atom_keys(src_mol):
    """``(chain, resid, insertion, name)`` keys of atoms coordinating a metal.

    Two signals, unioned rather than tried in turn. The ``"mc"`` bond type the
    readers set from LINK / struct_conn records is read under the same trust
    rule the rest of moleculekit uses, that the types only mean anything when
    they line up with the bonds. But it covers far less than its name suggests:
    a cofactor's own metal is an atom *inside* that residue, so its bonds are
    not link records and are typed ``"un"`` like any other. 1U5U carries 104
    bonds, none of them ``"mc"``, including the inter-residue Fe-Tyr353
    coordination. Geometry is therefore the primary signal in practice and
    ``"mc"`` the refinement, and a structure can easily need both at once: a
    link-recorded zinc alongside a heme iron that is recorded nowhere.

    PROPKA is no help here either. It does not bond these atoms, and it infers
    the element of an atom named ``FE`` as fluorine.
    """
    from moleculekit.periodictable import METAL_ELEMENTS

    keys = set()
    if src_mol is None:
        return keys

    elements = np.char.title(np.char.strip(src_mol.element.astype(str)))
    is_metal = np.isin(elements, list(METAL_ELEMENTS))
    if not is_metal.any():
        return keys

    def key(idx):
        return (
            str(src_mol.chain[idx]),
            int(src_mol.resid[idx]),
            str(src_mol.insertion[idx]).strip(),
            str(src_mol.name[idx]).strip(),
        )

    bondtypes = (
        src_mol.bondtype if len(src_mol.bondtype) == len(src_mol.bonds) else None
    )
    if bondtypes is not None:
        for i, (a, b) in enumerate(src_mol.bonds):
            if str(bondtypes[i]) != "mc":
                continue
            a, b = int(a), int(b)
            for donor, partner in ((a, b), (b, a)):
                if is_metal[partner] and not is_metal[donor]:
                    keys.add(key(donor))
    coords = src_mol.coords[:, :, 0]
    metals = coords[is_metal]
    cutoff_sq = _METAL_COORDINATION_FALLBACK_DIST**2
    for idx in np.where(~is_metal)[0]:
        delta = metals - coords[idx]
        if np.any(np.einsum("ij,ij->i", delta, delta) < cutoff_sq):
            keys.add(key(idx))
    return keys


def _hold_metal_coordinated_groups(molecule, parameters, metal_keys, mark):
    """Stop metal-coordinated donors titrating, with the charge coordination
    actually leaves them.

    The two families end up in different places. An anionic donor is
    deprotonated by the metal and stays charged, the same conclusion
    ``rdkittools``' boundary-metal handling reaches when it strips a hydrogen
    and lowers the formal charge. A nitrogen donor is left *neutral*, not basic:
    a heme pyrrole nitrogen coordinating iron cannot accept a proton at all, yet
    PROPKA's ligand typing reads it as an aromatic amine and hands it the
    ``NAR`` charge of +1, letting four fictitious cations per heme drag down
    every basic residue nearby.

    Scoped to the atoms re-typing actually changed, not merely every residue it
    considered: an already-HETATM ligand was visible to PROPKA all along, so its
    groups are none of this function's business. Widening it to those moved a
    zinc-binding inhibitor's own predicted pKa by 7.6 units in 5DPX. Groups a
    templated formal charge already settled are skipped too. A canonical residue
    coordinating a metal -
    an aspartate on a calcium, a cysteine on a zinc - keeps its group untouched:
    those titrate, PROPKA has always modelled them, and overriding them moves
    the acids around a metal site by several pKa units.
    """
    held = 0
    for conformation in molecule.conformations.values():
        for group in conformation.groups:
            if not group.titratable:
                continue
            donors = set(group.interaction_atoms_for_acids) | set(
                group.interaction_atoms_for_bases
            )
            donors.add(group.atom)
            donor_keys = {_propka_atom_key(a) for a in donors if a.element != "H"}
            if not donor_keys & mark or not donor_keys & metal_keys:
                continue
            # coordination deprotonates: an anionic donor keeps its charge, a
            # would-be cationic one is simply neutral
            charge = min(float(parameters.charge.get(group.type, 0.0)), 0.0)
            group.titratable = False
            group.charge = charge
            if charge:
                ion_key = f"MC{group.label.strip().replace(' ', '_')}"[:20]
                parameters.ions[ion_key] = charge
                group.residue_type = ion_key
            held += 1
    return held


def _add_cofactor_metal_groups(molecule, parameters, templated_charges, mark):
    """Give a cofactor's own metal a fixed-charge group.

    ``is_ion_group`` recognises an ion by its *residue* name, so a standalone
    ``ZN`` residue is found but a metal that is one atom inside a larger residue
    is not: a heme iron's resname is ``HEM``. Nothing else picks it up either,
    once :func:`_correct_propka_elements` stops the element being misread as a
    halogen that ligand typing would type by accident. A buried trication would
    then contribute nothing at all, and the cofactor would present to PROPKA as
    its anions alone - on 1U5U that pushed a neighbouring arginine up by 3.3
    pKa units, in the opposite direction to the truth.

    Only metals the input gives a formal charge to are added, through the same ion
    channel the rest of this module uses, so the charge and its position both
    come from the templated chemistry.
    """
    from moleculekit.periodictable import METAL_ELEMENTS
    from propka.group import IonGroup

    added = 0
    for conformation in molecule.conformations.values():
        existing = {id(group.atom) for group in conformation.groups}
        for atom in list(conformation.atoms):
            if str(atom.element).strip().title() not in METAL_ELEMENTS:
                continue
            if id(atom) in existing:
                continue
            key = _propka_atom_key(atom)
            if key not in mark:
                continue
            charge = templated_charges.get(key, 0)
            if not charge:
                continue
            group = IonGroup(atom)
            ion_key = f"MM{atom.res_name.strip()}_{atom.name.strip()}"[:20]
            parameters.ions[ion_key] = float(charge)
            group.residue_type = ion_key
            conformation.setup_and_add_group(group)
            group.charge = float(charge)
            group.titratable = False
            added += 1
    return added


def _run_propka(propka_args, biomolecule, src_mol=None, detect_specs=None):
    """Run PROPKA on ``biomolecule`` and return its per-group pKa rows.

    Replaces ``pdb2pqr.main.run_propka`` for two reasons. It reconciles
    PROPKA's textually inferred termini against PDB2PQR's own decision
    (:func:`_clear_phantom_termini`), which has to happen after the atoms are
    read and before groups are extracted, and it skips the folding- and
    charge-profile report that ``run_propka`` builds over a pH 0 to 14 window
    and that this caller discards.

    When ``src_mol`` and ``detect_specs`` are given it also makes non-canonical
    sidechains visible to PROPKA (:func:`_noncanonical_sidechain_plan`) and,
    for templated residues, replaces PROPKA's titration verdict on
    them with the input's formal charge (:func:`_apply_templated_formal_charges`).

    The read sequence mirrors the PDB branch of
    ``propka.input.read_molecule_file``.
    """
    from io import StringIO

    import propka.input as pk_in
    import propka.lib
    from propka.molecular_container import MolecularContainer
    from propka.parameters import Parameters
    from pdb2pqr import io as pqr_io

    lines = pqr_io.print_biomolecule_atoms(
        atomlist=biomolecule.atoms,
        chainflag=propka_args.keep_chain,
        pdbfile=True,
    )
    with StringIO() as fpdb:
        fpdb.writelines(lines)
        parameters = pk_in.read_parameter_file(propka_args.parameters, Parameters())
        molecule = MolecularContainer(parameters, propka_args)
        molecule.name = "input"
        conformations, conformation_names = pk_in.read_pdb(
            fpdb, molecule.version.parameters, molecule
        )
        if len(conformations) == 0:
            raise RuntimeError(
                "PROPKA found no molecular conformations in the structure "
                "handed to it by PDB2PQR."
            )
        molecule.conformations = conformations
        molecule.conformation_names = conformation_names
        molecule.top_up_conformations()
        propka.lib.protein_precheck(
            molecule.conformations, molecule.conformation_names
        )
        _clear_phantom_termini(molecule, biomolecule)
        mark, templated_charges = _noncanonical_sidechain_plan(src_mol, detect_specs)
        if mark:
            nucleic = _nucleic_residue_keys(biomolecule)
            mark = {key for key in mark if key[:3] not in nucleic}
            templated_charges = {
                k: v for k, v in templated_charges.items() if k in mark
            }
        retyped = set()
        if mark:
            retyped = _mark_ligand_like_sidechains(molecule, mark)
            logger.debug(
                f"Re-typed {len(retyped)} non-canonical sidechain atoms so "
                "PROPKA can group them; corrected "
                f"{_correct_propka_elements(molecule, src_mol, mark)} elements "
                f"and assigned {_apply_sybyl_types(molecule, src_mol, mark)} "
                "SYBYL types"
            )
        molecule.version.setup_bonding_and_protonation(molecule)
        molecule.extract_groups()
        if templated_charges:
            logger.debug(
                "Applied templated formal charges to "
                f"{_apply_templated_formal_charges(molecule, parameters, templated_charges, propka_args.pH)} "
                "PROPKA groups"
            )
        if mark:
            logger.debug(
                "Added "
                f"{_add_cofactor_metal_groups(molecule, parameters, templated_charges, mark)}"
                " fixed-charge groups for cofactor-internal metals"
            )
        if retyped:
            metal_keys = _metal_coordinated_atom_keys(src_mol)
            logger.debug(
                "Held "
                f"{_hold_metal_coordinated_groups(molecule, parameters, metal_keys, retyped)}"
                " metal-coordinated PROPKA groups non-titratable"
            )
        for name in molecule.conformation_names:
            molecule.conformations[name].sort_atoms()
        molecule.find_covalently_coupled_groups()

    molecule.calculate_pka()

    rows = []
    for group in molecule.conformations["AVR"].groups:
        atom = group.atom
        rows.append(
            {
                "res_num": atom.res_num,
                "ins_code": atom.icode,
                "res_name": atom.res_name,
                "chain_id": atom.chain_id,
                "group_label": group.label,
                "group_type": getattr(group, "type", None),
                "pKa": group.pka_value,
                "model_pKa": group.model_pka,
                "buried": group.buried,
                "coupled_group": (
                    group.coupled_titrating_group.label
                    if group.coupled_titrating_group
                    else None
                ),
            }
        )
    return rows
