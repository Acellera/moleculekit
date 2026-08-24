# (c) 2015-2026 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
"""Review a protonation call against what surrounds it.

A residue whose predicted pKa sits far from the simulation pH is decided. One
within about a pH unit of it could go either way, and the deciding evidence is
its surroundings: a hydrogen bond at 2.8 A, a carboxylate 4 A out, a metal on a
histidine nitrogen. :func:`reviewProtonation` answers both halves of that
question, and :class:`ProtonationReview` renders the answer.
"""

from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Callable, cast
import math
import numpy as np
import logging

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule

# A resolved chain-name lookup: (chain, resid, insertion, segid) -> deposited
# chain name, or None when nothing names that residue. segid is part of the
# key because it is part of the chain_map key autoSegment's
# return_chain_map produces.
ChainResolver = Callable[[str, int, str, str], "str | None"]

logger = logging.getLogger(__name__)

# Named as REASONS a residue is worth confirming, not as the thing that
# caused them: a protein residue flagged by a nearby ligand's charge (ASP
# 189, flagged by BEN's amidinium) is not itself a ligand. Used at every
# append site, in the sort precedence below, and in reviewProtonation's
# docstring.
FLAG_PKA_MARGIN = "pKa margin"
FLAG_METAL_CONTACT = "metal contact"
FLAG_LIGAND_CHARGE = "ligand charge"

# Precedence for sorting a multiply-flagged residue: most decisive evidence
# first. A metal or ligand charge the pKa predictor never saw outranks a
# margin the predictor itself was merely unsure about. An explicit mapping
# rather than branching on substring membership (`"metal" in reasons`), so
# renaming a reason above cannot silently change the ordering.
_FLAG_PRECEDENCE = {
    FLAG_METAL_CONTACT: 0,
    FLAG_LIGAND_CHARGE: 1,
    FLAG_PKA_MARGIN: 2,
}


@dataclass
class Contact:
    """Nearest heavy-atom approach between the reviewed residue and a neighbour.

    ``bondtype`` is read from ``mol.bonds`` and ``mol.bondtype``, never
    guessed or inferred from distance: it is the raw string moleculekit
    records for the bond between ``own_atom`` and ``other_atom`` specifically
    (not merely a bond somewhere between the two residues), or ``None`` when
    no bond is RECORDED between that exact pair. ``None`` does not mean no
    bond exists: a molecule returned by
    :func:`~moleculekit.tools.preparation.systemPrepare` carries very few
    bonds (22 for 3231 atoms on prepared 3PTB), so almost nothing is
    annotated there, while a structure read from a file with connectivity, or
    one whose bonds were guessed or established by templating, annotates
    fully.
    """

    resname: str
    resid: int
    insertion: str
    chain: str
    deposited_chain: "str | None"
    own_atom: str
    other_atom: str
    distance: float
    bondtype: "str | None" = None


@dataclass
class ChargeContact:
    """Nearest approach to a charge-carrying group.

    The distance is to the atoms carrying the charge, not to the nearest atom
    of the molecule carrying them. In 3PTB the benzamidine's ring edge is 5.5 A
    from His57 while its amidinium nitrogens are 9.4 A, and it is the
    amidinium that would matter.

    ``bondtype`` is the same annotation :class:`Contact` carries, between the
    subject's nearest atom and the specific charge-carrying atom the distance
    above was measured to, read from ``mol.bonds`` and ``mol.bondtype`` and
    never guessed. ``None`` means no bond is RECORDED between that pair, not
    that no bond exists: see :class:`Contact` for the same limitation.
    """

    resname: str
    resid: int
    insertion: str
    chain: str
    deposited_chain: "str | None"
    label: str
    charge: int
    distance: float
    # Which rule produced the underlying ChargedGroup: "table", "phosphate",
    # "terminus", "formalcharge" or "ion". Provenance for the reader.
    source: str = "table"
    # True for every ion (no ionic-charge magnitude table exists in the
    # library): a zinc is +2 and a molybdate is 2-, so the renderer must show
    # the label alone, never "metal cation +1". See ChargedGroup for why.
    sign_only: bool = False
    bondtype: "str | None" = None


@dataclass
class ReviewedResidue:
    """One titratable residue whose predicted state is close enough to the pH
    to be worth confirming."""

    resname: str
    protonation: str
    resid: int
    insertion: str
    chain: str
    deposited_chain: "str | None"
    pKa: float
    margin: float
    buried: float
    # Which rule(s) put this residue in the report: FLAG_METAL_CONTACT,
    # FLAG_LIGAND_CHARGE and/or FLAG_PKA_MARGIN (see reviewProtonation's
    # docstring for what each means). Ordered most decisive first when more
    # than one applies (see _FLAG_PRECEDENCE).
    flagged_by: tuple = (FLAG_PKA_MARGIN,)
    contacts: "list[Contact]" = field(default_factory=list)
    charges: "list[ChargeContact]" = field(default_factory=list)
    # A bonded sequence neighbour's contact is dropped only when BOTH atoms
    # are backbone; a neighbour's sidechain reaching the subject is kept.
    # Counts how many pairs were dropped, so the omission is not silent.
    backbone_links_suppressed: int = 0
    # Further contacts or charged groups within radius but cut off by
    # max_contacts or max_charges. Zero when the parameter is None or the
    # limit was never reached.
    contacts_truncated: int = 0
    charges_truncated: int = 0


def _chain_label(chain, deposited):
    if deposited is None or deposited == chain:
        return f"{chain}"
    return f"{chain} (dep. {deposited})"


# Words a reader can use for a bond order or type recorded in mol.bondtype,
# keyed by the strings moleculekit's own writers already use: "1", "2" and
# "3" for a single, double or triple covalent bond, "ar" for aromatic, and
# "mc" for moleculekit's metal-coordination type (moleculekit/writers.py
# downgrades "mc" for formats that cannot express it and maps it to
# struct_conn as "metalc" in mmCIF). A type not in this table renders as its
# raw code rather than raising: an unusual recorded value is not a wrong
# answer, only one this table has not been taught a word for yet.
_BOND_TYPE_WORDS = {
    "1": "single",
    "2": "double",
    "3": "triple",
    "ar": "aromatic",
    "mc": "metal coordination",
}


def _bond_type_map(mol):
    """Map of an unordered atom-index pair to its recorded bond type.

    Read from ``mol.bonds`` and ``mol.bondtype`` together, never guessed and
    never inferred from distance. ``zip`` stops at the shorter of the two
    arrays, which matters because they can disagree in length: setting
    ``mol.bonds = mol._getBonds()`` appends every guessed bond to ``bonds``
    without adding a matching entry to ``bondtype``. A bond in that gap is
    silently left out of this map rather than raising on an out-of-range
    index, and a pair with no entry here annotates as unbonded, the same as a
    pair that carries no bond at all: see :class:`Contact` for why that is
    the documented behaviour rather than a defect.
    """
    out = {}
    for (i, j), bt in zip(mol.bonds, mol.bondtype):
        i, j = int(i), int(j)
        out[(i, j) if i < j else (j, i)] = str(bt)
    return out


def _bond_between(bond_types, i, j):
    """Recorded bond type between two atom indices, or None when unrecorded."""
    return bond_types.get((i, j) if i < j else (j, i))


def _render_tables(
    contacts,
    charges,
    contact_radius,
    charge_radius,
    indent,
    backbone_links_suppressed=0,
    contacts_truncated=0,
    charges_truncated=0,
):
    """Render the contact and charge tables at a given left margin.

    Shared by :class:`Environment` and :class:`ProtonationReview`, so the two
    cannot render the same measurement differently; the report nests these
    tables inside a residue block and so passes a deeper indent.

    ``backbone_links_suppressed``, ``contacts_truncated`` and
    ``charges_truncated`` each render as one extra line, right after the
    table they describe, whenever non-zero: an omission must not be silent.

    A row whose pair carries a recorded bond is annotated with it in words
    (see ``_BOND_TYPE_WORDS``), appended after the distance so the distance
    column itself never moves.
    """
    pad = " " * indent
    row = " " * (indent + 2)
    lines = [f"{pad}contacts (<= {contact_radius:.1f} A)"]
    if contacts:
        for c in contacts:
            # The atom pair is padded as one unit, not just its second half:
            # padding only other_atom left the distance column shifted
            # whenever own_atom's length varied (C-N vs ND1-OD2).
            pair = f"{c.own_atom}-{c.other_atom}"
            line = (
                f"{row}{c.resname:<4s} {c.resid:>4d} "
                f"{_chain_label(c.chain, c.deposited_chain):<12s} "
                f"{pair:<12s} {c.distance:5.2f}"
            )
            if c.bondtype is not None:
                line += f"  {_BOND_TYPE_WORDS.get(c.bondtype, c.bondtype)}"
            lines.append(line)
    else:
        lines.append(f"{row}none")
    if backbone_links_suppressed == 1:
        lines.append(f"{row}(1 backbone link to a sequence neighbour suppressed)")
    elif backbone_links_suppressed:
        lines.append(
            f"{row}({backbone_links_suppressed} backbone links to sequence "
            f"neighbours suppressed)"
        )
    if contacts_truncated == 1:
        lines.append(f"{row}(1 further contact within {contact_radius:.1f} A not shown)")
    elif contacts_truncated:
        lines.append(
            f"{row}({contacts_truncated} further contacts within "
            f"{contact_radius:.1f} A not shown)"
        )
    lines.append(f"{pad}charges (<= {charge_radius:.1f} A)")
    if charges:
        for g in charges:
            # sign_only groups (every ion; see ChargedGroup) show by label
            # alone, since the charge field is a sign marker, not a real
            # magnitude.
            charge_text = g.label if g.sign_only else f"{g.label} {g.charge:+d}"
            line = (
                f"{row}{g.resname:<4s} {g.resid:>4d} "
                f"{_chain_label(g.chain, g.deposited_chain):<12s} "
                f"{charge_text:<18s} {g.distance:5.2f}"
            )
            if g.bondtype is not None:
                line += f"  {_BOND_TYPE_WORDS.get(g.bondtype, g.bondtype)}"
            lines.append(line)
    else:
        lines.append(f"{row}none")
    if charges_truncated == 1:
        lines.append(f"{row}(1 further charge within {charge_radius:.1f} A not shown)")
    elif charges_truncated:
        lines.append(
            f"{row}({charges_truncated} further charges within "
            f"{charge_radius:.1f} A not shown)"
        )
    return lines


def _render_unclassified(unclassified, indent):
    """Render the unclassified-charges line at a given left margin.

    Shared by :class:`Environment` and :class:`ProtonationReview`: what
    nothing could classify, already scoped by the caller (see
    :func:`_scope_unclassified`) to residues near the subject and excluding
    the subject itself, so this function only renders whatever list it is
    given. A structure whose ligands were never templated lists every nearby
    one here, instead of quietly reporting no nearby charges. Returns no
    lines at all when there is nothing to say, so a caller can extend
    unconditionally.
    """
    pad = " " * indent
    if not unclassified:
        return []
    counts = {}
    for resname, _, _, _ in unclassified:
        counts[resname] = counts.get(resname, 0) + 1
    listed = ", ".join(
        resname if n == 1 else f"{resname}(x{n})"
        for resname, n in sorted(counts.items())
    )
    n = len(unclassified)
    noun = "residue carries" if n == 1 else "residues carry"
    return [
        f"{pad}{n} {noun} no table entry and no assigned formal charge: {listed}"
    ]


@dataclass
class Environment:
    """What surrounds a selection: its contacts and its nearby charges.

    Produced by :func:`describeEnvironment`. ``str()`` renders a report in
    the same table layout :class:`ProtonationReview` uses for each residue;
    the dataclass fields hold the same numbers for programmatic use, and
    :meth:`to_dict` serializes them.

    Attributes
    ----------
    contacts, charges : list of Contact, list of ChargeContact
        The two scans, nearest first.
    subject, contact_radius, charge_radius : str, float, float
        Human-readable identity of the selection described, e.g.
        ``"HIS 587 A"`` for one residue or ``"ZN 1001 A + OIR 2001 A"`` for
        several, and the two radii (A) the scans were run at.
    charge_sources, unclassified : dict, list
        ``charge_sources`` is provenance for the whole-molecule charge scan:
        a count of charge-carrying groups found by each classification rule.
        Not rendered by ``str()``, since a whole-system total would name
        internal rule identities under a per-selection report, but available
        here and in :meth:`to_dict`.

        ``unclassified`` lists the residues no rule could assign a charge to
        and IS rendered, scoped to those with a heavy atom within
        ``charge_radius`` of the selection, excluding the selection itself
        since such a residue could never have appeared in the charges table
        above. For the unscoped, whole-molecule answer, or the answer for a
        broad selection such as a whole binding site, use
        :func:`~moleculekit.tools.charged_groups.chargedGroups` directly.
    backbone_links_suppressed, contacts_truncated, charges_truncated : int, int, int
        What the contact and charge tables silently dropped: adjacent
        backbone-to-backbone pairs excluded by ``exclude_adjacent``, and
        further contacts or charges cut off by ``max_contacts`` or
        ``max_charges``. Disclosed as counts here and as a line in ``str()``
        whenever non-zero.
    """

    contacts: "list[Contact]"
    charges: "list[ChargeContact]"
    subject: str
    contact_radius: float
    charge_radius: float
    charge_sources: dict
    unclassified: list
    backbone_links_suppressed: int = 0
    contacts_truncated: int = 0
    charges_truncated: int = 0

    def to_dict(self) -> dict:
        """JSON-serializable view of the environment."""
        return asdict(self)

    def __str__(self):
        lines = [self.subject, ""]
        lines.extend(
            _render_tables(
                self.contacts,
                self.charges,
                self.contact_radius,
                self.charge_radius,
                indent=0,
                backbone_links_suppressed=self.backbone_links_suppressed,
                contacts_truncated=self.contacts_truncated,
                charges_truncated=self.charges_truncated,
            )
        )
        unclassified_lines = _render_unclassified(self.unclassified, indent=0)
        if unclassified_lines:
            lines.append("")
            lines.extend(unclassified_lines)
        return "\n".join(lines)


@dataclass
class ProtonationReview:
    """The protonation calls worth confirming, with the evidence for each.

    ``str()`` renders a report; the dataclass fields hold the same numbers for
    programmatic use, and :meth:`to_dict` serializes them.
    """

    residues: "list[ReviewedResidue]"
    pH: float
    margin: float
    contact_radius: float
    charge_radius: float
    # Stored like the other radii, so to_dict() records every parameter
    # that decided which residues appear. A report you cannot reproduce from
    # its own serialized form is a report you cannot check.
    metal_radius: float
    ligand_charge_radius: float
    # Residues systemPrepare's pKa column carries a non-null value for, but
    # whose value is PROPKA's not-titrated sentinel (abs(pKa) >= 90) rather
    # than a real prediction. Counted here, not just silently dropped from
    # n_titratable, because a silent drop is the exact failure this module
    # exists to prevent.
    no_usable_pka: int
    n_titratable: int
    charge_sources: dict
    # Scoped to residues with a heavy atom within charge_radius of ANY
    # reported residue, and excluding every reported residue itself: see
    # reviewProtonation's docstring and _scope_unclassified. NOT the
    # whole-molecule list chargedGroups(mol) returns.
    unclassified: list

    def to_dict(self) -> dict:
        """JSON-serializable view of the whole report."""
        return asdict(self)

    def __str__(self):
        lines = [
            f"PROTONATION TO CONFIRM (pH {self.pH:.1f}, "
            f"margin {self.margin:.1f}, metal {self.metal_radius:.1f} A, "
            f"ligand {self.ligand_charge_radius:.1f} A)"
            f"   {len(self.residues)} of {self.n_titratable} titratable residues",
            "",
        ]
        for r in self.residues:
            direction = "above" if r.pKa >= self.pH else "below"
            if r.buried is None or (isinstance(r.buried, float) and math.isnan(r.buried)):
                buried_str = "buried n/a"
            else:
                buried_str = f"buried {r.buried:.2f}"
            lines.append(
                f"  {r.resname:<4s} {r.resid:>4d}{r.insertion.strip():<1s} "
                f"{_chain_label(r.chain, r.deposited_chain):<12s}   {r.protonation:<4s} "
                f"pKa {r.pKa:5.2f}  {r.pKa - self.pH:+.2f} {direction} pH   "
                f"{buried_str}   flagged: {', '.join(r.flagged_by)}"
            )
            lines.extend(
                _render_tables(
                    r.contacts,
                    r.charges,
                    self.contact_radius,
                    self.charge_radius,
                    indent=4,
                    backbone_links_suppressed=r.backbone_links_suppressed,
                    contacts_truncated=r.contacts_truncated,
                    charges_truncated=r.charges_truncated,
                )
            )
            lines.append("")

        if self.no_usable_pka:
            noun = "residue carries" if self.no_usable_pka == 1 else "residues carry"
            verb = "was" if self.no_usable_pka == 1 else "were"
            lines.append(
                f"  {self.no_usable_pka} {noun} no usable pKa prediction and {verb} "
                f"not considered"
            )
        lines.extend(_render_unclassified(self.unclassified, indent=2))
        return "\n".join(lines)


def _load_details(details):
    """Accept the DataFrame systemPrepare returns, or a path to the CSV it was
    written to. Step 5.2 of a preparation workflow often runs in a fresh
    session reading its inputs off disk."""
    import pandas as pd

    if isinstance(details, pd.DataFrame):
        return details
    return pd.read_csv(details)


def _cell(value):
    """Text form of a details-table cell that may be missing.

    A residue with no insertion code is an empty string in the DataFrame
    systemPrepare returns and NaN once that has been through a CSV, so the two
    input paths must agree here or they produce different reports from the same
    data. Asked with ``pd.isna`` rather than by testing the sentinel's type, so
    a change of sentinel does not silently reintroduce the difference.
    """
    import pandas as pd

    return "" if pd.isna(value) else str(value)


def _resolve_chain_map(chain_map) -> "ChainResolver | None":
    """Accept a dict, a path to a JSON file, an already-resolved callable, or None.

    Returns a callable taking (chain, resid, insertion, segid) and giving the
    deposited chain, or None.

    The dict is the form :func:`autoSegment
    <moleculekit.tools.autosegment.autoSegment>` returns with
    ``return_chain_map=True``: one entry per residue of the segmented
    molecule, keyed by ``f"{chain}:{resid}:{insertion}:{segid}"``. Resolving
    per residue rather than per chain is what keeps a merged chain namable:
    1a25's six calcium merge into one new chain from deposited chains A and
    B, and only a per-residue key can still tell them apart.

    A path (``str`` or ``os.PathLike``) is read as JSON holding that same
    dict, for a review step running in a fresh session reading its inputs off
    disk. A plain callable is passed straight through: reviewProtonation
    resolves chain_map once for the whole report and passes the resolved
    callable back through here per residue, rather than re-reading a JSON
    file each time.

    Every branch returns a callable annotated to the same ``ChainResolver``
    shape rather than an inferred, unannotated lambda: an inferred signature
    that only loosely matched its neighbours is what once widened this
    function's return type to include ``object``, silently breaking the
    ``str | None`` contract on ``ReviewedResidue`` and
    ``Contact.deposited_chain`` two calls away.
    """
    import json
    import os

    if chain_map is None:
        return None

    if callable(chain_map):
        # Already resolved by an earlier call (see the docstring above);
        # trusted to have the ChainResolver shape already, not re-verified.
        return cast(ChainResolver, chain_map)

    if isinstance(chain_map, (str, os.PathLike)):
        with open(chain_map) as fh:
            chain_map = json.load(fh)

    def resolve_from_dict(
        chain: str, resid: int, insertion: str, segid: str
    ) -> "str | None":
        return chain_map.get(f"{chain}:{resid}:{insertion}:{segid}")

    return resolve_from_dict


def _residue_mask(mol, chain, resid, insertion, segid=None):
    """Atoms of one residue.

    ``segid`` is included whenever the details table carried one. Without it
    two residues sharing ``(chain, resid, insertion)`` are silently unioned
    into a single subject set, and the reported contacts and charges then
    belong to two different residues at once. ``chain_map`` avoids the same
    ambiguity by keying on all four fields, so this mask must not guess
    either.
    """
    mask = (
        (mol.chain == chain)
        & (mol.resid == int(resid))
        & (mol.insertion == insertion)
    )
    if segid is not None:
        mask &= mol.segid == segid
    return mask


def _charge_sources(groups):
    """Count of charge-carrying groups found by each classification rule.

    Provenance for a report: which rule, ``"table"``, ``"phosphate"``,
    ``"terminus"``, ``"ion"`` or ``"formalcharge"``, produced how many of the
    charges being described.
    """
    return dict(Counter(g.source for g in groups))


def _subject_residues(mol, own_idx):
    """(chain, resid, insertion) of every residue a subject selection covers.

    ``own_idx`` may span one residue or several; this returns one key per
    residue actually touched, not one per atom, so a selection that only
    reaches part of a residue's atoms (e.g. a single named atom) still
    excludes that residue whole. Feeds :func:`_is_subject_residue`, the one
    subject-exclusion test used everywhere the subject could otherwise
    surface in its own report.
    """
    return set(
        zip(
            mol.chain[own_idx].tolist(),
            mol.resid[own_idx].tolist(),
            mol.insertion[own_idx].tolist(),
        )
    )


def _is_subject_residue(chain, resid, insertion, subject_residues):
    """True when (chain, resid, insertion) is one of the subject's own residues.

    The one subject-exclusion test, used everywhere a residue that IS the
    subject could otherwise surface in its own report (as a contact, as a
    nearby charge, in the unclassified list): a residue's own evidence is
    not context for its own decision. :func:`_charge_proximity_keys` applies
    the same rule in its own vectorized form.
    """
    return (str(chain), int(resid), str(insertion)) in subject_residues


def _charge_proximity_keys(mol, heavy_idx, groups, radius):
    """(chain, resid, insertion) of every heavy atom in ``heavy_idx`` within
    ``radius`` of one of ``groups``, excluding each group's own residue.

    Shared by reviewProtonation's metal rule and its ligand-charge rule: a
    residue's own charge is not context for its own decision, the same
    invariant :func:`_is_subject_residue` states. Without it, a metal
    residue given a predicted pKa would coordinate itself at distance zero,
    and a charged ligand's own heavy atoms are trivially within radius of
    its own charge group; only the ligand case is observable in practice,
    since no metal in the library carries a real pKa. Vectorized over
    ``heavy_idx`` rather than calling :func:`_is_subject_residue` per atom,
    since this runs once per candidate group against every heavy atom in
    the molecule.

    Returns an empty set when ``radius`` is 0 (how both rules are switched
    off) or when there is nothing to test against.
    """
    from moleculekit.distance import cdist

    keys = set()
    if radius <= 0 or not groups or len(heavy_idx) == 0:
        return keys

    chains = mol.chain[heavy_idx]
    resids = mol.resid[heavy_idx]
    insertions = mol.insertion[heavy_idx]
    for g in groups:
        own_residue = (chains == g.chain) & (resids == g.resid) & (
            insertions == g.insertion
        )
        others = heavy_idx[~own_residue]
        if len(others) == 0:
            continue
        d = cdist(mol.coords[others, :, 0], mol.coords[g.atoms, :, 0])
        near = others[d.min(axis=1) <= radius]
        for i in near:
            keys.add((str(mol.chain[i]), int(mol.resid[i]), str(mol.insertion[i])))
    return keys


def _scope_unclassified(mol, unclassified, ref_coords, radius, subject_residues):
    """``unclassified`` residues that could plausibly explain a missing charge.

    :func:`~moleculekit.tools.charged_groups.chargedGroups` scans the WHOLE
    molecule, so its ``unclassified`` list names every residue no source
    could classify, however far away: MEASURED on 3PTB, a bound calcium's
    nearest benzamidine atom is 22.61 A away against a charge radius of 8.0.
    This keeps only a residue with a heavy atom within ``radius`` of
    ``ref_coords``, since only those could have contributed a charge the
    charge scan would have shown.

    Also drops any residue that is itself one of ``subject_residues`` (see
    :func:`_is_subject_residue`), regardless of distance: the charges table
    already excludes the subject by construction, so such a residue could
    never have appeared there to begin with. A caller wanting undetermined
    charge WITHIN a broad selection has a direct answer that needs no such
    exclusion: ``chargedGroups(mol, sel)`` scans only the selection.

    Returns an empty list when ``ref_coords`` carries no atoms, since
    nothing can be within radius of nothing.
    """
    from moleculekit.distance import cdist

    if len(ref_coords) == 0:
        return []

    kept = []
    for resname, resid, insertion, chain in unclassified:
        if _is_subject_residue(chain, resid, insertion, subject_residues):
            continue
        mask = (
            (mol.chain == chain)
            & (mol.resid == resid)
            & (mol.insertion == insertion)
            & (mol.element != "H")
        )
        idx = np.where(mask)[0]
        d = cdist(ref_coords, mol.coords[idx, :, 0])
        if d.min() <= radius:
            kept.append((resname, resid, insertion, chain))
    return kept


def _subject_label(mol, own_idx):
    """Human-readable identity of a selection.

    One residue reads as its own name, e.g. ``HIS 587 A``. A handful of
    residues are joined with ``+``, e.g. ``ZN 1001 A + OIR 2001 A``. Beyond
    five residues, individual names would not fit a header line, so the
    label collapses to a count per resname instead.
    """
    seen = {}
    order = []
    for i in own_idx:
        key = (str(mol.chain[i]), int(mol.resid[i]), str(mol.insertion[i]))
        if key not in seen:
            seen[key] = str(mol.resname[i])
            order.append(key)

    if len(order) <= 5:
        return " + ".join(f"{seen[key]} {key[1]}{key[2]} {key[0]}" for key in order)

    counts = {}
    for key in order:
        counts[seen[key]] = counts.get(seen[key], 0) + 1
    chains = sorted({key[0] for key in order})
    listed = ", ".join(f"{n}x {resname}" for resname, n in sorted(counts.items()))
    return f"{len(order)} residues ({listed}) in chain {'/'.join(chains)}"


def _polymer_keys(mol):
    """(chain, resid) of every residue that is part of a polymer chain.

    A residue is polymer here if it is protein (all of N, CA and C present),
    nucleic (any backbone link atom present), or a capping group covalently
    bonded into the chain it caps. Decided from backbone-atom presence rather
    than resname, exactly as :func:`_classify_residues
    <moleculekit.tools.autosegment._classify_residues>` decides polymer
    membership for segmentation, so a modified amino acid or nucleotide
    bonded into a chain (a phosphoserine, a sulfotyrosine) is recognised here
    too. A capping group counts even though, unlike a real terminus, it
    carries no charge of its own: whether a neighbour is bonded and whether
    it is charged are different questions, and this answers only the first.

    Used only to decide whether a resid-adjacent pair is a genuine bonded
    sequence neighbour or two unrelated residues that merely share
    consecutive numbers, which is only true within a polymer: a ligand or an
    ion numbered next to a protein residue is not bonded to it.
    """
    from moleculekit.tools.autosegment import PROTEIN_BB, NUCLEIC_LINK
    from moleculekit.residues import CAP_RESIDUE_NAMES

    _, residue_idx = mol.getResidues(sel="all", return_idx=True)
    keys = set()
    for idx in residue_idx:
        rep = idx[0]
        names = set(mol.name[idx])
        is_polymer = (
            all(a in names for a in PROTEIN_BB)
            or any(a in names for a in NUCLEIC_LINK)
            or str(mol.resname[rep]) in CAP_RESIDUE_NAMES
        )
        if is_polymer:
            keys.add((str(mol.chain[rep]), int(mol.resid[rep])))
    return keys


def _sequence_link_atoms(mol, chain, resid, res_atoms, polymer_keys):
    """This residue's own atoms bonded to an atom of a sequence-adjacent
    polymer residue (same chain, resid within one, itself polymer).

    Uses ``mol.bonds`` only, never guessed connectivity: an empty result
    here is read by the caller as "no bond information", not as "no link",
    which matters because those two mean different things.
    """
    if len(mol.bonds) == 0:
        return set()

    res_atom_set = set(int(i) for i in res_atoms)
    bonds = mol.bonds
    in0 = np.isin(bonds[:, 0], list(res_atom_set))
    in1 = np.isin(bonds[:, 1], list(res_atom_set))
    links = set()
    for i, j in bonds[in0 ^ in1]:
        i, j = int(i), int(j)
        mine, other = (i, j) if i in res_atom_set else (j, i)
        other_chain = str(mol.chain[other])
        other_resid = int(mol.resid[other])
        if (
            other_chain == chain
            and abs(other_resid - resid) <= 1
            and (other_chain, other_resid) in polymer_keys
        ):
            links.add(mine)
    return links


def _polymer_backbone_atoms(mol, chain, resid, polymer_keys, backbone_by_name):
    """Backbone atom indices of one polymer residue.

    When the molecule carries bonds and the residue has both of its sequence
    links (a bonded chain-link atom on the resid-1 side and one on the
    resid+1 side), the backbone is the shortest covalent path between those
    two atoms, walking only the residue's own internal bonds. That
    definition has no name dependence and no length limit, so it is correct
    for a non-canonical residue with an elongated main chain: the extra
    atoms sit on the path and are backbone regardless of what they are
    called.

    Proline is why a path is walked rather than the residue's whole atom set
    being trusted: its CD bonds back to N, closing a five-membered ring, so
    N and C are connected by two routes of different length (N-CA-C, length
    2; N-CD-CG-CB-CA-C, length 5) and only the shorter one is backbone.
    ``networkx.shortest_path`` on an unweighted graph returns the shorter one
    by construction.

    Falls back to a name-based backbone test (``backbone_by_name``, computed
    once by the caller from the N/CA/C/O and nucleic backbone names) when
    the molecule has no bonds at all, or when this residue has fewer than
    two sequence links, which is a chain terminus with no second link to
    walk a path to.

    A known gap, deliberately not handled: see :func:`describeEnvironment`'s
    ``exclude_adjacent`` parameter for the macrocyclic case a path walk
    cannot distinguish from a genuine shorter backbone.
    """
    import networkx as nx

    res_atoms = np.where((mol.chain == chain) & (mol.resid == resid))[0]
    by_name = set(int(i) for i in res_atoms if backbone_by_name[i])

    if len(mol.bonds) == 0:
        return by_name

    links = _sequence_link_atoms(mol, chain, resid, res_atoms, polymer_keys)
    if len(links) < 2:
        return by_name

    res_atom_set = set(int(i) for i in res_atoms)
    bonds = mol.bonds
    both_in_res = np.isin(bonds[:, 0], list(res_atom_set)) & np.isin(
        bonds[:, 1], list(res_atom_set)
    )
    graph = nx.Graph()
    graph.add_nodes_from(res_atom_set)
    for i, j in bonds[both_in_res]:
        graph.add_edge(int(i), int(j))

    start, end = sorted(links)[:2]
    try:
        path = nx.shortest_path(graph, start, end)
    except nx.NetworkXNoPath:
        return by_name
    return set(path)


def describeEnvironment(
    mol: "Molecule",
    sel,
    contact_radius: float = 4.0,
    charge_radius: float = 8.0,
    exclude_adjacent: bool = True,
    max_contacts: "int | None" = None,
    max_charges: "int | None" = None,
    chain_map=None,
    charged_groups=None,
) -> Environment:
    """What surrounds a selection: its nearest contacts and its nearby charges.

    This is the measurement :func:`reviewProtonation` makes for each residue
    it reports, exposed on its own because none of it has anything to do with
    pKa or protonation. The subject can be a single residue, a ligand, or a
    whole binding site: anything :meth:`Molecule.atomselect
    <moleculekit.molecule.Molecule.atomselect>` can resolve.

    Two scans are run against the rest of the molecule. A contact scan finds
    the nearest heavy-atom approach per neighbouring residue, covering
    hydrogen bonds, metal coordination and close steric neighbours. A wider
    charge scan finds the nearest approach to every charge-carrying group in
    the structure, because electrostatics reach further than contact does.
    Charges are measured to the atoms carrying the charge, not to the nearest
    atom of the group carrying them.

    When the reported atom pair itself carries a recorded bond, the row is
    annotated with what it is (see :attr:`Contact.bondtype` and
    :attr:`ChargeContact.bondtype` for the annotation rule and its one
    documented limitation: a prepared molecule carries very few bonds, so
    almost nothing is annotated there, while a structure with real
    connectivity annotates fully).

    A residue inside the selection is never reported as part of its own
    environment, however broad the selection: ``describeEnvironment(mol,
    "resname BEN")`` and ``describeEnvironment(mol, "within 8 of resname
    BEN")`` both exclude BEN from their own contacts, charges and
    unclassified list, since the charges table already excludes the subject
    by construction and such a residue could never have appeared there. A
    caller who instead wants undetermined charge WITHIN a broad selection
    has a direct answer that needs no such exclusion::

        from moleculekit.tools.charged_groups import chargedGroups
        groups, unclassified = chargedGroups(mol, "within 8 of resname BEN")

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The structure to describe. Needs no pH and no preparation details:
        this measurement has nothing to do with protonation state.
    sel : str or np.ndarray
        The subject. An atom selection string, a boolean mask, or an integer
        index array, exactly as accepted by :meth:`Molecule.atomselect
        <moleculekit.molecule.Molecule.atomselect>`. May cover one residue or
        several.
    contact_radius : float
        Radius (A) for the contact scan.
    charge_radius : float
        Radius (A) for the charge scan.
    exclude_adjacent : bool
        Affects the CONTACT scan only; the charge scan always reports an
        adjacent residue's charge. A neighbouring carboxylate shifts a pKa
        whether or not it is sequence-adjacent, and a free terminal charge
        sits on backbone N, O or OXT, so extending this rule to charges
        would hide real evidence rather than a peptide bond.

        For a partner within one resid of any subject residue, in the same
        chain, where both are part of a polymer (decided from backbone-atom
        presence, not resname), drop the contact only when both atoms of
        the reported pair are backbone: the peptide-bond geometry itself is
        uninformative, but a neighbour's sidechain reaching the subject is
        real chemistry and is kept. Does not apply when either residue is
        not part of a polymer, so a ligand or ion merely numbered next to a
        protein residue keeps that contact. Every link dropped this way is
        counted in ``backbone_links_suppressed``.

        Backbone atoms are the shortest covalent path between a residue's
        two sequence links when both are recorded as bonds, which needs no
        atom names and so is exact for a non-canonical residue with an
        elongated main chain. A chain terminus, or a structure with no
        bonds at all, falls back to a name-based test instead. A known gap:
        a macrocyclic non-canonical residue whose sidechain bridges its own
        two link atoms can present a shorter false path, giving an extra
        visible contact row rather than a missing one.
    max_contacts : int or None
        Keep at most this many contacts, nearest first. None keeps every
        contact within ``contact_radius``. A truncating value is disclosed
        both in the rendered table and in ``Environment.contacts_truncated``,
        never silently.
    max_charges : int or None
        Keep at most this many charged groups, nearest first. None keeps
        every group within ``charge_radius``, and a truncating value is
        disclosed the same way, in ``Environment.charges_truncated``.
    chain_map : dict, str or None
        Translation back to deposited chain names, accepted in the same forms
        as :func:`reviewProtonation` takes it.
    charged_groups : tuple or None
        The ``(groups, unclassified)`` pair :func:`chargedGroups
        <moleculekit.tools.charged_groups.chargedGroups>` returns for ``mol``.
        A caller describing many selections on the same molecule, such as
        :func:`reviewProtonation` scanning every flagged residue, computes
        this once and passes it in rather than paying for a whole-molecule
        charge scan per selection.

    Returns
    -------
    environment : :class:`Environment`
        ``print()`` it for a report; its fields hold the same data for
        programmatic use, and :meth:`Environment.to_dict` serializes them.

    Example
    -------
    >>> env = describeEnvironment(mol, 'chain "A" and resid 587')  # doctest: +SKIP
    >>> print(env)  # doctest: +SKIP
    """
    from moleculekit.tools.charged_groups import chargedGroups
    from moleculekit.residues import WATER_RESIDUE_NAMES

    own_mask = mol.atomselect(sel)
    own_idx = np.where(own_mask)[0]
    if len(own_idx) == 0:
        raise RuntimeError(
            "sel selected no atoms, so there is no environment to describe."
        )

    heavy = mol.element != "H"
    own = np.where(own_mask & heavy)[0]
    if len(own) == 0:
        raise RuntimeError(
            "sel selected no heavy atoms, so no contact or charge distance "
            "can be measured from it."
        )
    own_coords = mol.coords[own, :, 0]

    if charged_groups is None:
        groups, unclassified = chargedGroups(mol)
    else:
        groups, unclassified = charged_groups

    resolve_chain = _resolve_chain_map(chain_map)
    is_water = np.isin(mol.resname, list(WATER_RESIDUE_NAMES))
    subject_keys = set(zip(mol.chain[own_idx].tolist(), mol.resid[own_idx].tolist()))
    own_residues = _subject_residues(mol, own_idx)
    bond_types = _bond_type_map(mol)
    if exclude_adjacent:
        polymer_keys = _polymer_keys(mol)
        # find_backbone() is called directly, and with mol.bonds rather than
        # Molecule.atomselect("backbone"): that selector's higher-level
        # protein/nucleic residue classification turns out to depend on a
        # guessed bond graph even when asked not to guess one (verified on
        # 1kdx: with guessBonds=False it resolves only 4 of the structure's
        # 436 true backbone atoms), which would silently starve the
        # name-based fallback below of almost everything it exists to catch.
        # find_backbone() itself is a pure name lookup (N/CA/C/O and the
        # nucleic backbone names), refined only for a terminal oxygen using
        # whichever bonds mol already carries. Never guessed here either.
        from moleculekit.atomselect.analyze import find_backbone

        backbone_by_name = find_backbone(mol, mol.bonds, "protein") | find_backbone(
            mol, mol.bonds, "nucleic"
        )
    else:
        polymer_keys = set()
        backbone_by_name = np.zeros(mol.numAtoms, dtype=bool)

    contacts, backbone_links_suppressed, contacts_truncated = _scan_contacts(
        mol,
        own,
        own_coords,
        own_mask,
        heavy,
        is_water,
        subject_keys,
        polymer_keys,
        backbone_by_name,
        contact_radius,
        exclude_adjacent,
        resolve_chain,
        max_contacts,
        own_residues,
        bond_types,
    )
    charges, charges_truncated = _scan_charges(
        mol,
        own,
        own_coords,
        groups,
        charge_radius,
        resolve_chain,
        max_charges,
        own_residues,
        bond_types,
    )
    unclassified_nearby = _scope_unclassified(
        mol, unclassified, own_coords, charge_radius, own_residues
    )

    return Environment(
        contacts=contacts,
        charges=charges,
        subject=_subject_label(mol, own_idx),
        contact_radius=contact_radius,
        charge_radius=charge_radius,
        charge_sources=_charge_sources(groups),
        unclassified=unclassified_nearby,
        backbone_links_suppressed=backbone_links_suppressed,
        contacts_truncated=contacts_truncated,
        charges_truncated=charges_truncated,
    )


def reviewProtonation(
    mol: "Molecule",
    details,
    pH: float,
    margin: float = 1.0,
    contact_radius: float = 4.0,
    charge_radius: float = 8.0,
    metal_radius: float = 2.6,
    ligand_charge_radius: float = 4.0,
    chain_map=None,
    exclude_adjacent: bool = True,
    max_contacts: "int | None" = None,
    max_charges: "int | None" = None,
) -> ProtonationReview:
    """Which protonation calls are in doubt, and what surrounds each of them.

    Keeps a titratable residue for one of three reasons. Its predicted pKa
    lies within ``margin`` of ``pH`` (rule 1). A metal cation coordinates it
    within ``metal_radius``, whatever its pKa (rule 2). Or a heavy atom of
    it lies within ``ligand_charge_radius`` of a charge-carrying group on a
    non-polymer residue whose charge came from a formal charge assigned at
    templating, again whatever its pKa (rule 3). Each reported residue
    carries which of the three applied in its ``flagged_by`` field, as one or
    more of :data:`FLAG_METAL_CONTACT`, :data:`FLAG_LIGAND_CHARGE` and
    :data:`FLAG_PKA_MARGIN`, named as the REASON rather than as the thing
    that caused it: a protein residue flagged because a ligand's charge sits
    nearby is not itself a ligand.

    Rules 2 and 3 exist for the same reason. PROPKA models protein
    electrostatics, which is its job, so a protein carboxylate near a
    titratable residue is already accounted for in the pKa it predicts. It
    does not model a metal, and it does not model a formal charge assigned
    to a ligand at templating: neither one was there when the calculation
    ran. A confident pKa is therefore not evidence that either call is safe.
    Rule 3 is restricted to non-polymer residues for the same reason a
    protein-protein salt bridge stays out of it: PROPKA already saw that
    charge, so flagging it would tell the reader nothing new and would flag
    half a buried protein along the way.

    Then measures each reported residue's surroundings with
    :func:`describeEnvironment`: a contact scan and a wider charge scan, at
    the two separate radii below.

    A row whose pKa is PROPKA's not-titrated sentinel (abs(pKa) >= 90, seen
    as exactly 99.99 in practice) carries no usable prediction and is dropped
    before any rule sees it, the same as a row with no pKa at all. The count
    dropped this way is disclosed in the report and in
    :attr:`ProtonationReview.no_usable_pka`, and ``n_titratable`` counts only
    the rows left after the drop.

    :attr:`ProtonationReview.unclassified` lists, once for the whole report,
    every residue :func:`~moleculekit.tools.charged_groups.chargedGroups`
    could not classify that has a heavy atom within ``charge_radius`` of ANY
    reported residue, and never a residue that is itself reported. The same
    scoping :func:`describeEnvironment` applies to
    :attr:`Environment.unclassified` for a single selection, extended here
    to the whole set of reported residues so the report keeps one
    unclassified line rather than one per residue.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The prepared molecule, as returned by
        :func:`systemPrepare <moleculekit.tools.preparation.systemPrepare>`.
    details : pandas.DataFrame or str
        The per-residue table ``systemPrepare(return_details=True)`` returns,
        or a path to the CSV it was written to.
    pH : float
        The pH the structure was prepared at.
    margin : float
        Keep a residue when its pKa is within this many units of ``pH``.
    contact_radius : float
        Radius (A) for the contact scan.
    charge_radius : float
        Radius (A) for the charge scan.
    metal_radius : float
        A titratable residue with a heavy atom this close to a metal cation is
        reported whatever its predicted pKa, because the pKa predictor does not
        model the metal. Sized to a coordination bond: in 1R1J the Zn sits 1.97
        to 2.04 A from its three protein ligands. 0.0 switches rule 2 off.
    ligand_charge_radius : float
        A titratable residue with a heavy atom this close to a charge-carrying
        group on a non-polymer residue, sourced from a formal charge assigned
        at templating rather than from an ion, is reported whatever its
        predicted pKa, because the pKa predictor never saw that charge.
        Sized to a salt bridge: on 3PTB this reaches the Asp189 carboxylate
        2.9 A from the templated benzamidine's amidinium, and stops short of
        pulling in an unrelated disulfide cysteine 4.8 A away. A group whose
        charge belongs to the residue being reviewed is never matched against
        that residue, the same guard rule 2 applies to a metal residue's own
        charge. Ions are rule 2's business at its own tighter radius, so this
        rule never matches one, and the two stay disjoint by source. 0.0
        switches rule 3 off.
    chain_map : dict, str or None
        Translation back to deposited chain names, so a report names chains
        the reader can find in the deposited entry. The dict is the one
        :func:`autoSegment <moleculekit.tools.autosegment.autoSegment>`
        returns with ``return_chain_map=True``: one entry per residue, keyed
        by ``f"{chain}:{resid}:{insertion}:{segid}"``, so it stays correct
        inside a chain segmentation merged from more than one deposited
        chain. A string is read as a path to a JSON file holding that same
        dict. A residue whose key the map does not carry simply has no
        deposited name. Supplying a map that names none of the reviewed
        residues raises, since a report in which every chain is unnamed is
        indistinguishable from one built with no map at all.
    exclude_adjacent : bool
        Passed straight through to :func:`describeEnvironment`, which makes
        this measurement per residue: affects the CONTACT scan only, and
        drops a sequence-adjacent polymer neighbour's contact only when both
        atoms of the reported pair are backbone (see its docstring for how
        backbone is decided and the one documented gap). Every link dropped
        this way is counted in
        :attr:`ReviewedResidue.backbone_links_suppressed`.
    max_contacts : int or None
        Keep at most this many contacts per residue, nearest first. None
        keeps every contact within ``contact_radius``. A truncating value is
        disclosed both in the rendered table and in
        :attr:`ReviewedResidue.contacts_truncated`, never silently.
    max_charges : int or None
        Keep at most this many charged groups per residue, nearest first.
        None keeps every group within ``charge_radius``, disclosed the same
        way in :attr:`ReviewedResidue.charges_truncated`.

    Returns
    -------
    review : :class:`ProtonationReview`
        ``print()`` it for a report; walk ``review.residues`` for the numbers.

    Example
    -------
    >>> pmol, specs, df = systemPrepare(mol, pH=7.4, return_details=True)  # doctest: +SKIP
    >>> print(reviewProtonation(pmol, df, pH=7.4))  # doctest: +SKIP
    """
    from moleculekit.tools.charged_groups import chargedGroups

    df = _load_details(details)
    resolve_chain = _resolve_chain_map(chain_map)

    if "pKa" not in df.columns:
        raise RuntimeError(
            "The details table carries no pKa column, so no protonation call can "
            "be reviewed. Run systemPrepare with titration=True."
        )

    # See the docstring for why the sentinel is dropped here, before any
    # rule runs, so n_titratable and every selection rule agree on what
    # "titratable" means.
    has_pka = df.pKa.notna()
    is_sentinel = has_pka & (df.pKa.abs() >= 90)
    no_usable_pka = int(is_sentinel.sum())
    titratable = df[has_pka & ~is_sentinel].copy()
    titratable["_margin"] = (titratable.pKa - pH).abs()

    groups, unclassified = chargedGroups(mol)
    polymer_keys = _polymer_keys(mol)
    heavy_idx = np.where(mol.element != "H")[0]

    # Rule 2: a residue coordinating a metal cation is worth confirming whatever
    # its predicted pKa says, because the pKa predictor never saw the metal. In
    # 1R1J the histidine a Zn contradicts at 2.04 A is predicted at pKa 10.20,
    # margin 2.80: rule 1 does not reach it at any defensible margin, and the
    # same Zn's coordinating glutamate is predicted at -6.80. A margin filter
    # keeps the residues the predictor was unsure about and discards the ones it
    # was confidently wrong about, which is the wrong half.
    # A monatomic cation is a group whose charge field records only a sign,
    # on the positive side (branching on the flag rather than g.label text,
    # so a rename cannot silently switch this off).
    metal_groups = [g for g in groups if g.sign_only and g.charge > 0]
    metal_keys = _charge_proximity_keys(mol, heavy_idx, metal_groups, metal_radius)

    # Rule 3 (see docstring): non-polymer only, so a protein-protein salt
    # bridge PROPKA already modelled does not flood the results; restricted
    # to source "formalcharge" so it stays disjoint from rule 2, whose
    # charges are ions.
    ligand_groups = [
        g
        for g in groups
        if g.source == "formalcharge" and (g.chain, g.resid) not in polymer_keys
    ]
    ligand_keys = _charge_proximity_keys(
        mol, heavy_idx, ligand_groups, ligand_charge_radius
    )

    def _row_key(row):
        return (str(row.chain), int(row.resid), _cell(row.insertion))

    def _flag_rank(reasons):
        # See _FLAG_PRECEDENCE: most decisive first.
        return min(_FLAG_PRECEDENCE[r] for r in reasons)

    keep = []
    for _, row in titratable.iterrows():
        reasons = []
        key = _row_key(row)
        if key in metal_keys:
            reasons.append(FLAG_METAL_CONTACT)
        if key in ligand_keys:
            reasons.append(FLAG_LIGAND_CHARGE)
        if row["_margin"] <= margin:
            reasons.append(FLAG_PKA_MARGIN)
        if reasons:
            keep.append((row, tuple(reasons)))

    keep.sort(key=lambda rr: (_flag_rank(rr[1]), rr[0]["_margin"]))
    charge_sources = _charge_sources(groups)

    heavy = mol.element != "H"

    has_segid = "segid" in df.columns

    reviewed = []
    n_resolved = 0
    # Accumulated across every reported residue, so the aggregate
    # unclassified line below can be scoped to "near ANY reported residue"
    # and can exclude every one of them as its own subject, the same
    # invariant describeEnvironment applies to a single selection.
    reported_heavy_mask = np.zeros(mol.numAtoms, dtype=bool)
    reported_residues = set()
    for row, reasons in keep:
        chain = str(row.chain)
        resid = int(row.resid)
        insertion = _cell(row.insertion)
        segid = _cell(row.segid) if has_segid else None
        own_mask = _residue_mask(mol, chain, resid, insertion, segid)
        if not np.any(own_mask & heavy):
            raise RuntimeError(
                f"Residue {row.resname} {resid}{insertion} chain {chain} is in the "
                f"details table but not in the molecule, so the two do not "
                f"describe the same structure."
            )
        reported_heavy_mask |= own_mask & heavy
        reported_residues.update(_subject_residues(mol, np.where(own_mask)[0]))

        # Delegates the whole surroundings measurement to describeEnvironment
        # rather than keeping a second copy of the two scans, which would
        # drift apart. groups/unclassified and resolve_chain are passed
        # through so each is computed once for the whole report, not once
        # per reviewed residue.
        env = describeEnvironment(
            mol,
            own_mask,
            contact_radius=contact_radius,
            charge_radius=charge_radius,
            exclude_adjacent=exclude_adjacent,
            max_contacts=max_contacts,
            max_charges=max_charges,
            chain_map=resolve_chain,
            charged_groups=(groups, unclassified),
        )
        contacts = env.contacts
        charges = env.charges
        backbone_links_suppressed = env.backbone_links_suppressed
        contacts_truncated = env.contacts_truncated
        charges_truncated = env.charges_truncated

        deposited_chain = (
            resolve_chain(chain, resid, insertion, segid) if resolve_chain else None
        )
        if deposited_chain is not None:
            n_resolved += 1

        reviewed.append(
            ReviewedResidue(
                resname=str(row.resname),
                protonation=str(row.protonation),
                resid=resid,
                insertion=insertion,
                chain=chain,
                deposited_chain=deposited_chain,
                pKa=float(row.pKa),
                margin=float(row["_margin"]),
                flagged_by=reasons,
                buried=float(row.buried) if "buried" in df.columns else float("nan"),
                contacts=contacts,
                charges=charges,
                backbone_links_suppressed=backbone_links_suppressed,
                contacts_truncated=contacts_truncated,
                charges_truncated=charges_truncated,
            )
        )

    if resolve_chain is not None and reviewed and n_resolved == 0:
        raise RuntimeError(
            f"chain_map resolved none of the {len(reviewed)} reviewed residues, so "
            f"it does not describe this molecule. Its chains would all read as "
            f"unnamed, which is indistinguishable from a report built with no "
            f"chain_map at all, and the caller explicitly asked for deposited "
            f"chain names. Molecule chains are "
            f"{sorted(set(mol.chain.tolist()) - {''})}."
        )

    reported_coords = mol.coords[np.where(reported_heavy_mask)[0], :, 0]
    unclassified_nearby = _scope_unclassified(
        mol, unclassified, reported_coords, charge_radius, reported_residues
    )

    return ProtonationReview(
        residues=reviewed,
        pH=pH,
        margin=margin,
        contact_radius=contact_radius,
        charge_radius=charge_radius,
        metal_radius=metal_radius,
        ligand_charge_radius=ligand_charge_radius,
        no_usable_pka=no_usable_pka,
        n_titratable=int(len(titratable)),
        charge_sources=charge_sources,
        unclassified=unclassified_nearby,
    )


def _scan_contacts(
    mol,
    own,
    own_coords,
    own_mask,
    heavy,
    is_water,
    subject_keys,
    polymer_keys,
    backbone_by_name,
    radius,
    exclude_adjacent,
    resolve_chain,
    max_contacts,
    own_residues,
    bond_types,
):
    """Nearest heavy-atom approach per neighbouring residue, within ``radius``.

    ``subject_keys`` is the set of (chain, resid) pairs the subject selection
    covers, one pair per subject residue. For a partner in the same chain
    within one resid of ANY of them, where both residues are in
    ``polymer_keys``, the reported pair is dropped only when both of its
    atoms are backbone (see :func:`_polymer_backbone_atoms`): the peptide
    bond geometry between two polymer residues is fixed and uninformative,
    but a neighbour's sidechain reaching the subject is real chemistry and is
    kept. Outside a polymer, resid arithmetic means nothing, so the rule does
    not apply at all: a ligand or an ion numbered next to a protein residue
    is not bonded to it.

    ``own_residues`` (see :func:`_is_subject_residue`) drops a candidate
    whole, before the adjacency check above ever runs, whenever it belongs to
    one of the subject's own residues: ``own_mask`` already excludes every
    atom ``sel`` actually named, but a selection that names only some of a
    residue's atoms (a single named atom, say) would otherwise leave that
    residue's other atoms free to appear as a "contact" of the subject with
    itself.

    Each kept row is annotated with the recorded bond, if any, between
    ``own_atom_idx`` and ``j`` specifically (see :func:`_bond_between`): the
    contact scan reports the nearest atom pair per partner residue, which is
    usually the bonded pair when one exists but is not assumed to be, so the
    annotation is only ever for the exact pair being reported.

    Returns ``(contacts, backbone_links_suppressed, truncated)``: the kept
    contacts, nearest first, capped at ``max_contacts`` (every one of them
    when it is None); a count of pairs dropped as a bonded sequence
    neighbour; and a count of further contacts the cap itself cut off. Both
    counts exist so an omission is disclosed rather than silent.
    """
    from moleculekit.distance import cdist

    candidates = np.where(heavy & ~own_mask & ~is_water)[0]
    if len(candidates) == 0:
        return [], 0, 0
    d = cdist(own_coords, mol.coords[candidates, :, 0])
    near = np.where(d.min(axis=0) <= radius)[0]
    if len(near) == 0:
        return [], 0, 0

    keys = np.array(
        [
            f"{mol.chain[i]}:{mol.resid[i]}:{mol.insertion[i]}"
            for i in candidates[near]
        ]
    )
    out = []
    suppressed = 0
    for key in dict.fromkeys(keys):
        cols = near[keys == key]
        sub = d[:, cols]
        a, b = np.unravel_index(int(sub.argmin()), sub.shape)
        j = int(candidates[cols[b]])
        own_atom_idx = int(own[a])
        j_chain = str(mol.chain[j])
        j_resid = int(mol.resid[j])
        j_ins = str(mol.insertion[j])
        j_segid = str(mol.segid[j])

        if _is_subject_residue(j_chain, j_resid, j_ins, own_residues):
            continue

        if exclude_adjacent and (j_chain, j_resid) in polymer_keys:
            matches = [
                (c, r)
                for c, r in subject_keys
                if j_chain == c and abs(j_resid - r) <= 1 and (c, r) in polymer_keys
            ]
            if matches:
                partner_backbone = _polymer_backbone_atoms(
                    mol, j_chain, j_resid, polymer_keys, backbone_by_name
                )
                subject_backbone = set()
                for c, r in matches:
                    subject_backbone |= _polymer_backbone_atoms(
                        mol, c, r, polymer_keys, backbone_by_name
                    )
                if own_atom_idx in subject_backbone and j in partner_backbone:
                    suppressed += 1
                    continue

        out.append(
            Contact(
                resname=str(mol.resname[j]),
                resid=j_resid,
                insertion=j_ins,
                chain=j_chain,
                deposited_chain=(
                    resolve_chain(j_chain, j_resid, j_ins, j_segid)
                    if resolve_chain
                    else None
                ),
                own_atom=str(mol.name[own_atom_idx]),
                other_atom=str(mol.name[j]),
                distance=float(sub.min()),
                bondtype=_bond_between(bond_types, own_atom_idx, j),
            )
        )
    out.sort(key=lambda c: c.distance)
    kept = out[:max_contacts]
    return kept, suppressed, len(out) - len(kept)


def _scan_charges(
    mol, own, own_coords, groups, radius, resolve_chain, max_charges, own_residues, bond_types
):
    """Nearest approach to each charge-carrying group within ``radius``.

    Measures to the group's own atoms. A group belonging to one of the
    subject's own residues is skipped (see :func:`_is_subject_residue`): its
    own charge is not context for its own decision.

    Each kept row is annotated with the recorded bond, if any, between the
    subject's nearest atom and the specific charge-carrying atom the distance
    was measured to (see :func:`_bond_between`): a metal ion's charge group
    and the residue atom it coordinates are typically this exact pair, so a
    charge row can carry the same annotation a contact row for the same pair
    does.

    Returns ``(charges, truncated)``: the kept groups, nearest first, capped
    at ``max_charges`` (every one of them when it is None); and a count of
    further groups the cap cut off, so a truncation is disclosed rather than
    silent.
    """
    from moleculekit.distance import cdist

    out = []
    for g in groups:
        if _is_subject_residue(g.chain, g.resid, g.insertion, own_residues):
            continue
        d = cdist(own_coords, mol.coords[g.atoms, :, 0])
        dist = float(d.min())
        if dist > radius:
            continue
        a, b = np.unravel_index(int(d.argmin()), d.shape)
        own_atom_idx = int(own[a])
        charge_atom_idx = int(g.atoms[b])
        out.append(
            ChargeContact(
                resname=g.resname,
                resid=g.resid,
                insertion=g.insertion,
                chain=g.chain,
                deposited_chain=(
                    resolve_chain(
                        g.chain, g.resid, g.insertion, str(mol.segid[int(g.atoms[0])])
                    )
                    if resolve_chain
                    else None
                ),
                label=g.label,
                charge=g.charge,
                distance=dist,
                source=g.source,
                sign_only=g.sign_only,
                bondtype=_bond_between(bond_types, own_atom_idx, charge_atom_idx),
            )
        )
    out.sort(key=lambda g: g.distance)
    kept = out[:max_charges]
    return kept, len(out) - len(kept)
