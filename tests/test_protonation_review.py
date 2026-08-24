import pytest

from moleculekit.tools.protonation_review import (
    FLAG_METAL_CONTACT,
    FLAG_LIGAND_CHARGE,
    FLAG_PKA_MARGIN,
)


def _review():
    from moleculekit.tools.protonation_review import (
        Contact,
        ChargeContact,
        ReviewedResidue,
        ProtonationReview,
    )

    his = ReviewedResidue(
        resname="HIS",
        protonation="HIP",
        resid=57,
        insertion="",
        chain="A",
        deposited_chain="A",
        pKa=7.44,
        margin=0.04,
        buried=0.83,
        contacts=[
            Contact("ASP", 102, "", "A", "A", "ND1", "OD2", 2.67),
            Contact("SER", 195, "", "A", "A", "NE2", "OG", 2.96),
        ],
        charges=[ChargeContact("ASP", 102, "", "A", "A", "carboxylate", -1, 2.67)],
    )
    tyr = ReviewedResidue(
        resname="TYR",
        protonation="TYR",
        resid=39,
        insertion="",
        chain="A",
        deposited_chain="A",
        pKa=8.24,
        margin=0.84,
        buried=0.0,
        contacts=[],
        charges=[],
    )
    return ProtonationReview(
        residues=[his, tyr],
        pH=7.4,
        margin=1.0,
        contact_radius=4.0,
        charge_radius=8.0,
        metal_radius=2.6,
        ligand_charge_radius=4.0,
        no_usable_pka=0,
        n_titratable=43,
        charge_sources={"table": 38, "terminus": 2, "ion": 1, "formalcharge": 9},
        unclassified=[("BEN", 300, "", "A")],
    )


def test_report_renders_the_header_and_counts():
    text = str(_review())
    assert "pH 7.4" in text
    assert "margin 1.0" in text
    assert "2 of 43" in text


def test_report_shows_each_residue_with_its_margin_direction():
    text = str(_review())
    assert "HIS" in text and "57" in text and "HIP" in text
    assert "7.44" in text
    assert "+0.04" in text, "a pKa above pH must read as above"
    assert "0.83" in text


def test_report_lists_contacts_and_charges_under_their_radii():
    text = str(_review())
    assert "contacts (<= 4.0 A)" in text
    assert "charges (<= 8.0 A)" in text
    assert "ND1-OD2" in text
    assert "2.67" in text
    assert "carboxylate" in text


def test_report_says_none_rather_than_leaving_a_blank():
    text = str(_review())
    assert text.count("none") >= 2, "an empty scan must say so explicitly"


def test_report_shows_unclassified_but_not_charge_source_provenance():
    """The stage-dependence guard is the unclassified line: had templating not
    run, it would list every ligand, which is what makes reading formalcharge
    too early visible. charge_sources is kept for programmatic use (asserted
    via to_dict below) but is a whole-system total under a per-residue report
    and names internal rule identities a reader cannot be expected to know,
    so it is deliberately not rendered."""
    rep = _review()
    text = str(rep)
    assert "charges from" not in text
    assert "table 38" not in text
    assert "formalcharge 9" not in text
    assert "BEN" in text
    assert "no table entry and no assigned formal charge" in text

    data = rep.to_dict()
    assert data["charge_sources"]["table"] == 38
    assert data["charge_sources"]["formalcharge"] == 9


def test_report_omits_the_unclassified_line_when_there_is_nothing_to_say():
    rep = _review()
    rep.unclassified = []
    text = str(rep)
    assert "no table entry" not in text


def test_report_shows_why_each_residue_was_flagged():
    """A residue in the report because a metal coordinates it means something
    different from one there because its pKa is near the pH. The reader cannot
    tell them apart unless the report says so."""
    rep = _review()
    rep.residues[0].flagged_by = (FLAG_PKA_MARGIN, FLAG_METAL_CONTACT)
    rep.residues[1].flagged_by = (FLAG_PKA_MARGIN,)
    text = str(rep)
    assert "flagged: pKa margin, metal contact" in text
    assert "flagged: pKa margin" in text


def test_report_renders_a_metal_only_flag():
    """A confidently-predicted pKa far from the pH still belongs in the report
    when a metal contradicts it, so "metal contact" must stand alone as a
    reason."""
    rep = _review()
    rep.residues[0].pKa = 10.20
    rep.residues[0].margin = 2.80
    rep.residues[0].flagged_by = (FLAG_METAL_CONTACT,)
    text = str(rep)
    assert "flagged: metal contact" in text
    assert "10.20" in text
    assert "+2.80 above pH" in text


def test_report_records_every_parameter_that_shaped_it():
    """to_dict must carry all four radii and the margin, so a serialized report
    can be reproduced and checked. A parameter that changed which residues
    appear but is absent from the record makes the report unauditable."""
    data = _review().to_dict()
    for key in (
        "pH",
        "margin",
        "contact_radius",
        "charge_radius",
        "metal_radius",
        "ligand_charge_radius",
    ):
        assert key in data, f"{key} missing from to_dict()"
    assert data["metal_radius"] == 2.6
    assert data["ligand_charge_radius"] == 4.0
    assert "metal 2.6 A" in str(_review())
    assert "ligand 4.0 A" in str(_review())


def test_report_renders_a_deposited_chain_that_differs():
    rep = _review()
    rep.residues[0].chain = "C"
    rep.residues[0].deposited_chain = "AP"
    text = str(rep)
    assert "AP" in text


def test_report_handles_a_missing_buried_value():
    rep = _review()
    rep.residues[0].buried = float("nan")
    text = str(rep)
    assert "n/a" in text


def test_report_with_no_flagged_residues_says_so():
    rep = _review()
    rep.residues = []
    text = str(rep)
    assert "0 of 43" in text


def test_to_dict_is_json_serializable():
    import json

    data = _review().to_dict()
    text = json.dumps(data)
    back = json.loads(text)
    assert back["pH"] == 7.4
    assert len(back["residues"]) == 2
    assert back["residues"][0]["contacts"][0]["distance"] == 2.67


def test_report_gets_number_agreement_right_in_both_directions():
    """The report is read by a person and forwarded to a colleague, so "1
    residues carry" is a real defect rather than a cosmetic one. Both the
    singular and plural branches need pinning: fixing one and leaving the
    other is how this bug survived the first pass."""
    rep = _review()

    rep.unclassified = [("BEN", 300, "", "A")]
    one = str(rep)
    assert "1 residue carries no table entry" in one
    assert "1 residues" not in one

    rep.unclassified = [("BEN", 300, "", "A"), ("NAG", 401, "", "A")]
    many = str(rep)
    assert "2 residues carry no table entry" in many
    assert "2 residue carries" not in many


def test_report_does_not_claim_a_magnitude_for_an_ion():
    """chargedGroups reports sign and identity for ions, never a magnitude,
    because no ionic-charge table exists in the library. The renderer must
    not turn that sign marker into a magnitude claim: a zinc is +2 and a
    molybdate is 2-, so "metal cation +1" is a factual error in a report a
    chemist acts on."""
    from moleculekit.tools.protonation_review import ChargeContact

    rep = _review()
    rep.residues[0].charges = [
        ChargeContact("ZN", 1001, "", "A", "A", "metal cation", 1, 2.04,
                      source="ion", sign_only=True),
        ChargeContact("ASP", 102, "", "A", "A", "carboxylate", -1, 2.67,
                      source="table"),
    ]
    text = str(rep)
    assert "metal cation" in text
    assert "metal cation +1" not in text, "an ion must carry no magnitude"
    assert "carboxylate -1" in text, "a known magnitude must still be shown"


def test_a_formal_charge_is_rendered_once():
    """The label is an identity and the renderer supplies the magnitude, so
    a formal charge must not appear twice on the same line. Every other
    label in the module is a bare identity for this reason."""
    from moleculekit.tools.protonation_review import ChargeContact

    rep = _review()
    rep.residues[0].charges = [
        ChargeContact("BEN", 1, "", "A", "A", "formal charge", 1, 2.92,
                      source="formalcharge"),
        ChargeContact("LIG", 2, "", "A", "A", "formal charge", -2, 3.40,
                      source="formalcharge"),
    ]
    text = str(rep)
    assert "formal charge +1" in text
    assert "formal charge +1 +1" not in text
    assert "formal charge -2" in text
    assert "formal charge -2 -2" not in text


def test_report_columns_align():
    """The contact and charge tables are scanned by eye, so the distance
    column has to line up regardless of how long the atom names or labels
    are. Padding only part of a composed field is what broke this."""
    from moleculekit.tools.protonation_review import Contact, ChargeContact

    rep = _review()
    rep.residues[0].contacts = [
        Contact("PHE", 589, "", "A", "A", "C", "N", 2.94),
        Contact("ASP", 590, "", "A", "A", "ND1", "OD2", 2.80),
    ]
    rep.residues[0].charges = [
        ChargeContact("ZN", 1001, "", "A", "A", "metal cation", 1, 2.04,
                      source="ion", sign_only=True),
        ChargeContact("ARG", 717, "", "A", "A", "guanidinium", 1, 6.94,
                      source="table"),
    ]
    lines = str(rep).split("\n")

    def distance_column(needle):
        line = next(ln for ln in lines if needle in ln)
        return line.index(f"{float(line.split()[-1]):.2f}")

    assert distance_column("C-N") == distance_column("ND1-OD2")
    assert distance_column("metal cation") == distance_column("guanidinium")


import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def prepared_3ptb():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    pmol, _, df = systemPrepare(
        mol, pH=7.4, return_details=True, _logger_level="ERROR"
    )
    return pmol, df


def test_review_flags_only_residues_within_the_margin(prepared_3ptb):
    """3PTB carries a bound Ca2+ (resid 480, the trypsin calcium loop).
    Measured on this fixture: GLU 70 (OE1, 2.37 A) and GLU 80 (OE2, 2.40 A)
    both coordinate it inside the default metal_radius of 2.6 A, so both are
    flagged by "metal contact" alone even though their margins (1.30 and
    5.20) sit outside the 1.0 window. That is the metal rule working as
    designed, not
    a defect in the margin filter, so only the margin-only residues are held
    to the margin here."""
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb
    rep = reviewProtonation(pmol, df, pH=7.4, margin=1.0)
    assert rep.pH == 7.4 and rep.margin == 1.0
    for r in rep.residues:
        if r.flagged_by == (FLAG_PKA_MARGIN,):
            assert abs(r.pKa - 7.4) <= 1.0
        else:
            assert FLAG_METAL_CONTACT in r.flagged_by
    # Metal-flagged residues sort ahead of margin-only ones, and each group
    # is internally sorted by how close the call is.
    metal_margins = [r.margin for r in rep.residues if FLAG_METAL_CONTACT in r.flagged_by]
    plain_margins = [r.margin for r in rep.residues if r.flagged_by == (FLAG_PKA_MARGIN,)]
    assert metal_margins == sorted(metal_margins)
    assert plain_margins == sorted(plain_margins)


def test_review_widening_the_margin_can_only_add(prepared_3ptb):
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb
    narrow = reviewProtonation(pmol, df, pH=7.4, margin=1.0)
    wide = reviewProtonation(pmol, df, pH=7.4, margin=2.0)
    key = lambda r: (r.chain, r.resid, r.insertion)
    assert {key(r) for r in narrow.residues} <= {key(r) for r in wide.residues}


def test_review_reports_contacts_within_the_contact_radius(prepared_3ptb):
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb
    rep = reviewProtonation(pmol, df, pH=7.4, margin=2.0, contact_radius=4.0)
    seen = [c for r in rep.residues for c in r.contacts]
    assert seen, "3PTB at margin 2.0 must flag something with a neighbour"
    for c in seen:
        assert c.distance <= 4.0
        assert c.resname not in ("HOH", "WAT"), "water must be excluded"
    for r in rep.residues:
        dists = [c.distance for c in r.contacts]
        assert dists == sorted(dists)


def test_review_excludes_sequence_neighbours(prepared_3ptb):
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb
    on = reviewProtonation(pmol, df, pH=7.4, margin=2.0, exclude_adjacent=True)
    off = reviewProtonation(pmol, df, pH=7.4, margin=2.0, exclude_adjacent=False)
    for r in on.residues:
        for c in r.contacts:
            assert not (c.chain == r.chain and abs(c.resid - r.resid) <= 1)
    n_on = sum(len(r.contacts) for r in on.residues)
    n_off = sum(len(r.contacts) for r in off.residues)
    assert n_off > n_on


def test_review_radii_are_independent(prepared_3ptb):
    """Collapsing the two radii into one loses the distinction the two scans
    exist to draw."""
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb
    rep = reviewProtonation(
        pmol, df, pH=7.4, margin=2.0, contact_radius=4.0, charge_radius=8.0
    )
    for r in rep.residues:
        for c in r.contacts:
            assert c.distance <= 4.0
        for g in r.charges:
            assert g.distance <= 8.0
    far = [g for r in rep.residues for g in r.charges if g.distance > 4.0]
    assert far, "a charge beyond the contact radius must still be reported"


def test_review_measures_to_the_charge_carrying_atoms(prepared_3ptb):
    """3PTB's benzamidine: the ring edge is closer than the amidinium, and it
    is the amidinium that would shift a pKa."""
    from moleculekit.distance import cdist
    from moleculekit.tools.protonation_review import reviewProtonation

    shared, df = prepared_3ptb
    # Copy: the fixture is module-scoped and this test assigns a charge, which
    # would otherwise leak into every test that runs after it.
    pmol = shared.copy()
    # Give the benzamidine its amidinium charge, as templating would
    ns = pmol.atomselect("resname BEN and name N1 N2", indexes=True)
    if len(ns) == 0:
        pytest.skip("prepared 3ptb fixture has no BEN amidinium nitrogens")
    pmol.formalcharge[ns[0]] = 1

    rep = reviewProtonation(pmol, df, pH=7.4, margin=2.0, charge_radius=20.0)
    ben = [g for r in rep.residues for g in r.charges if g.resname == "BEN"]
    assert ben, "the charged benzamidine must be reported"
    for g in ben:
        r = [
            rr
            for rr in rep.residues
            if any(gg is g for gg in rr.charges)
        ][0]
        own = pmol.atomselect(
            (pmol.resid == r.resid)
            & (pmol.chain == r.chain)
            & (pmol.insertion == r.insertion)
            & (pmol.element != "H"),
            indexes=True,
        )
        to_charge = cdist(
            pmol.coords[own, :, 0], pmol.coords[[ns[0]], :, 0]
        ).min()
        assert abs(g.distance - to_charge) < 1e-3


def test_review_surfaces_a_metal_that_contradicts_the_call():
    """1R1J, the sharpest case in the whole design.

    The Zn coordinates three protein residues, and NOT ONE of them is reachable
    by a pKa margin. Measured on this fixture at pH 7.4:

        HIP 587  NE2-Zn 2.04 A   predicted pKa 10.20   margin 2.80
        HID 583  NE2-Zn 1.97 A   predicted pKa  4.58   margin 2.82
        GLU 646  OE1-Zn 2.01 A   predicted pKa -6.80   margin 14.20

    So this test is what proves the metal rule earns its place: at the default
    margin of 1.0 these residues appear only because a metal coordinates them.
    If it fails, do NOT widen the margin to make it pass. Widening is exactly
    the failure this rule exists to prevent.
    """
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare
    from moleculekit.tools.protonation_review import reviewProtonation

    mol = Molecule(os.path.join(curr_dir, "pdb", "1r1j.pdb"))
    mol.remove("water", _logger=False)
    pmol, _, df = systemPrepare(
        mol, pH=7.4, return_details=True, _logger_level="ERROR"
    )
    rep = reviewProtonation(pmol, df, pH=7.4, margin=1.0)

    by_resid = {r.resid: r for r in rep.residues}
    for resid in (587, 583, 646):
        assert resid in by_resid, (
            f"residue {resid} coordinates the Zn and must be reported even "
            f"though its pKa margin is far outside 1.0"
        )
        r = by_resid[resid]
        assert FLAG_METAL_CONTACT in r.flagged_by
        assert FLAG_PKA_MARGIN not in r.flagged_by, (
            f"residue {resid} has margin {r.margin:.2f}, which is outside 1.0, "
            f"so 'pKa margin' must not be given as a reason"
        )

    # And the metal itself is reported as context, at coordination distance
    his587 = by_resid[587]
    metals = [g for g in his587.charges if g.label == "metal cation"]
    assert metals, "the coordinating Zn must appear in the charge scan"
    assert min(g.distance for g in metals) < 2.6

    # Metal-flagged residues sort ahead of margin-flagged ones
    first_margin_only = next(
        (i for i, r in enumerate(rep.residues) if r.flagged_by == (FLAG_PKA_MARGIN,)),
        len(rep.residues),
    )
    last_metal = max(
        i for i, r in enumerate(rep.residues) if FLAG_METAL_CONTACT in r.flagged_by
    )
    assert last_metal < first_margin_only


def test_review_metal_rule_can_be_switched_off():
    """metal_radius=0 reduces the selection to the pure margin filter, which is
    what shows the rule is doing the work rather than the fixture."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare
    from moleculekit.tools.protonation_review import reviewProtonation

    mol = Molecule(os.path.join(curr_dir, "pdb", "1r1j.pdb"))
    mol.remove("water", _logger=False)
    pmol, _, df = systemPrepare(
        mol, pH=7.4, return_details=True, _logger_level="ERROR"
    )
    with_metal = reviewProtonation(pmol, df, pH=7.4, margin=1.0)
    without = reviewProtonation(pmol, df, pH=7.4, margin=1.0, metal_radius=0.0)

    assert {r.resid for r in without.residues} < {r.resid for r in with_metal.residues}
    for resid in (587, 583, 646):
        assert resid not in {r.resid for r in without.residues}
    for r in without.residues:
        assert r.flagged_by == (FLAG_PKA_MARGIN,)


def test_review_accepts_a_details_csv_path(prepared_3ptb, tmp_path):
    """The two input paths must produce the same report, insertion codes
    included. An empty insertion is "" in the DataFrame systemPrepare returns
    and NaN once it has been through a CSV, and reading that difference wrongly
    is what mismatches a residue between the table and the molecule. 3PTB has no
    insertion codes of its own, so one is given here: without it the branch that
    is the reason the code exists is never exercised."""
    from moleculekit.tools.protonation_review import reviewProtonation

    shared, df0 = prepared_3ptb
    csv = tmp_path / "details.csv"
    df0.to_csv(csv, index=False)
    from_df = reviewProtonation(shared, df0, pH=7.4, margin=1.0)
    from_path = reviewProtonation(shared, str(csv), pH=7.4, margin=1.0)
    assert len(from_df.residues) == len(from_path.residues)
    assert str(from_df) == str(from_path)

    # Copy: the fixture is module-scoped and this mutates both molecule and table
    pmol = shared.copy()
    df = df0.copy()
    # HIS 57 chain A is flagged at margin 1.0 (pKa 7.44)
    res = (pmol.chain == "A") & (pmol.resid == 57)
    assert res.sum() > 0
    pmol.insertion[res] = "A"
    row = (df.chain == "A") & (df.resid == 57)
    assert row.sum() == 1
    df.loc[row, "insertion"] = "A"

    csv2 = tmp_path / "details_insertion.csv"
    df.to_csv(csv2, index=False)
    ins_df = reviewProtonation(pmol, df, pH=7.4, margin=1.0)
    ins_path = reviewProtonation(pmol, str(csv2), pH=7.4, margin=1.0)

    assert [r.insertion for r in ins_df.residues].count("A") == 1, (
        "the inserted residue must be reported with its insertion code"
    )
    assert str(ins_df) == str(ins_path)
    assert [(r.resname, r.resid, r.insertion, r.chain) for r in ins_df.residues] == [
        (r.resname, r.resid, r.insertion, r.chain) for r in ins_path.residues
    ]


def test_review_separates_two_residues_sharing_a_chain_resid_key(prepared_3ptb):
    """The subject set is the reviewed residue, not every residue that happens
    to share (chain, resid, insertion). Two segments carrying the same key are
    silently unioned into one subject without the segid, and the contacts and
    charges then reported belong to two different residues at once. chain_map
    avoids exactly this ambiguity by keying on all four fields, so the mask
    must not resolve it by guessing either."""
    import numpy as np
    from moleculekit.tools.protonation_review import _residue_mask

    shared, _ = prepared_3ptb
    pmol = shared.copy()
    his = (pmol.chain == "A") & (pmol.resid == 57)
    ben = pmol.resname == "BEN"
    assert his.sum() > 0 and ben.sum() > 0
    # Give the ligand the histidine's (chain, resid, insertion) in another segid
    pmol.chain[ben] = "A"
    pmol.resid[ben] = 57
    pmol.segid[ben] = "Z9"

    without_segid = _residue_mask(pmol, "A", 57, "")
    assert without_segid.sum() == his.sum() + ben.sum(), "the union is reachable"

    with_segid = _residue_mask(pmol, "A", 57, "", str(pmol.segid[np.where(his)[0][0]]))
    assert with_segid.sum() == his.sum()
    assert not (with_segid & ben).any()


def test_review_renders_deposited_chains_from_a_chain_map():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment
    from moleculekit.tools.preparation import systemPrepare
    from moleculekit.tools.protonation_review import reviewProtonation

    mol0 = Molecule(os.path.join(curr_dir, "pdb", "1a25.pdb"))
    mol0.remove("water", _logger=False)
    mol, cmap = autoSegment(
        mol0, fields=("segid", "chain"), return_chain_map=True, _logger=False
    )
    pmol, _, df = systemPrepare(
        mol, pH=7.4, return_details=True, _logger_level="ERROR"
    )
    rep = reviewProtonation(pmol, df, pH=7.4, margin=1.0, chain_map=cmap)
    assert rep.residues, "1a25 must flag something at margin 1.0"
    for r in rep.residues:
        assert r.deposited_chain in ("A", "B")


def test_review_accepts_a_plain_dict_chain_map(prepared_3ptb):
    """A dict in the same shape autoSegment returns: one entry per residue,
    keyed by chain:resid:insertion:segid."""
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb
    chain_map = {
        f"{pmol.chain[i]}:{pmol.resid[i]}:{pmol.insertion[i]}:{pmol.segid[i]}": "ZZ"
        for i in range(pmol.numAtoms)
    }
    rep = reviewProtonation(pmol, df, pH=7.4, margin=2.0, chain_map=chain_map)
    assert rep.residues, "the fixture must flag something at margin 2.0"
    for r in rep.residues:
        assert r.deposited_chain == "ZZ"


def test_review_metal_rule_fires_on_a_charmm_ion_name(prepared_3ptb):
    """The metal rule reached only element-symbol residue names, so it silently
    never fired for the CHARMM ion names (CAL, CES, POT, SOD) or the
    oxidation-state PDB codes (CU1, MN3). The unreported charge was disclosed
    via the unclassified line; the unreported SELECTION was disclosed nowhere,
    and a residue whose predicted pKa a coordinating cation contradicts was
    simply absent from the report.

    Same geometry as test_review_flags_only_residues_within_the_margin: 3PTB's
    bound cation at resid 480, coordinated by GLU 70 (2.37 A) and GLU 80
    (2.40 A), both far outside the 1.0 margin. Only the residue name changes.
    """
    from moleculekit.tools.protonation_review import reviewProtonation
    from moleculekit.residues import CATIONIC_ION_RESIDUE_NAMES

    shared, df = prepared_3ptb
    for resn in sorted(CATIONIC_ION_RESIDUE_NAMES):
        pmol = shared.copy()
        ion = pmol.atomselect("resname CA and not protein", indexes=True)
        assert len(ion) == 1, "prepared 3ptb must carry its single cation"
        pmol.resname[ion] = resn

        rep = reviewProtonation(pmol, df, pH=7.4, margin=1.0)
        flagged = {r.resid for r in rep.residues if FLAG_METAL_CONTACT in r.flagged_by}
        assert flagged == {70, 80}, (
            f"{resn}: the metal rule found {sorted(flagged)}, so a coordinating "
            f"cation named {resn} does not reach a titratable residue"
        )
        for resid in (70, 80):
            r = next(rr for rr in rep.residues if rr.resid == resid)
            assert FLAG_PKA_MARGIN not in r.flagged_by, (
                f"{resn}: GLU {resid} has margin {r.margin:.2f}, outside 1.0, so "
                f"only the metal rule can have kept it"
            )
            near = [g for g in r.charges if g.resname == resn]
            assert near and min(g.distance for g in near) < 2.6


def test_review_raises_when_the_chain_map_describes_another_molecule(prepared_3ptb):
    """A map covering none of the molecule's chains gave every deposited_chain
    as None with no log line, so the report was indistinguishable from one built
    with no map at all. The caller explicitly asked for deposited names, so
    proceeding answers a different question than the one asked."""
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb
    ok = reviewProtonation(pmol, df, pH=7.4, margin=1.0)
    assert ok.residues, "the fixture must flag something for this to mean anything"

    with pytest.raises(RuntimeError, match="does not describe this molecule"):
        reviewProtonation(pmol, df, pH=7.4, margin=1.0, chain_map={"Q": "ZZ"})


def test_review_chain_map_from_a_json_file_resolves_per_residue(tmp_path):
    """The review step often runs in a fresh session reading its inputs off
    disk, which is the path a serialized map exists for. 1A25's six calcium
    merge into one chain: both the dict and the JSON file it was written to
    must still name each calcium its own deposited chain, A A A B B B, never
    a merged label."""
    import json
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment
    from moleculekit.tools.preparation import systemPrepare
    from moleculekit.tools.protonation_review import reviewProtonation

    mol0 = Molecule(os.path.join(curr_dir, "pdb", "1a25.pdb"))
    mol0.remove("water", _logger=False)
    mol, cmap = autoSegment(
        mol0, fields=("segid", "chain"), return_chain_map=True, _logger=False
    )
    pmol, _, df = systemPrepare(
        mol, pH=7.4, return_details=True, _logger_level="ERROR"
    )

    ca = pmol.atomselect("resname CA and not protein", indexes=True)
    assert len(ca) == 6
    assert len({str(pmol.chain[i]) for i in ca}) == 1, "the calcium must be merged"

    fn = tmp_path / "chainmap.json"
    fn.write_text(json.dumps(cmap))

    from_obj = reviewProtonation(pmol, df, pH=7.4, margin=1.0, chain_map=cmap)
    from_json = reviewProtonation(pmol, df, pH=7.4, margin=1.0, chain_map=str(fn))
    assert str(from_obj) == str(from_json)

    # The calcium appear in the charge scan, each naming its own deposited chain
    ca_deps = [
        g.deposited_chain
        for r in from_json.residues
        for g in r.charges
        if g.resname == "CA"
    ]
    assert ca_deps, "the calcium must be within the charge radius of something"
    assert set(ca_deps) == {"A", "B"}
    assert not [d for d in ca_deps if d is None or "+" in d]

    # And the map itself, read back from that file, names all six directly
    back = json.loads(fn.read_text())
    named = [
        back[f"{pmol.chain[i]}:{pmol.resid[i]}:{pmol.insertion[i]}:{pmol.segid[i]}"]
        for i in ca
    ]
    assert named == ["A", "A", "A", "B", "B", "B"]


def test_review_counts_charge_provenance_and_unclassified(prepared_3ptb):
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb
    rep = reviewProtonation(pmol, df, pH=7.4, margin=1.0)
    assert rep.charge_sources.get("table", 0) > 0
    assert rep.n_titratable > 0
    # BEN was never templated in this fixture, so it must be visible as
    # unclassified rather than silently contributing no charge.
    assert any(u[0] == "BEN" for u in rep.unclassified)
    assert "BEN" in str(rep)


@pytest.fixture(scope="module")
def prepared_3ptb_ben_charged():
    """3PTB with the benzamidine templated to an RCSB-style SMILES, giving its
    amidinium a real formal charge of +1 rather than leaving it unclassified.
    This is what a caller who ran templateResidueFromSmiles before
    systemPrepare would pass in, and is the fixture rule 3 needs: the plain
    ``prepared_3ptb`` fixture never templates BEN, so it has no formalcharge
    group to test the rule against."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    mol.remove("element H", _logger=False)
    mol.templateResidueFromSmiles(
        "resname BEN", "[NH2+]=C(N)c1ccccc1", addHs=True, _logger=False
    )
    pmol, _, df = systemPrepare(
        mol, pH=7.4, return_details=True, _logger_level="ERROR"
    )
    return pmol, df


def test_review_ligand_charge_rule_flags_asp189(prepared_3ptb_ben_charged):
    """Rule 3, measured: Asp189's carboxylate sits 2.9 A from the templated
    benzamidine's amidinium (a formal charge of +1). Predicted pKa 4.95,
    margin 2.45, is outside the default margin of 1.0, and no metal
    coordinates it, so rules 1 and 2 do not reach it; rule 3 does."""
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb_ben_charged
    rep = reviewProtonation(pmol, df, pH=7.4)
    by_key = {(r.resname, r.resid): r for r in rep.residues}
    assert ("ASP", 189) in by_key
    asp189 = by_key[("ASP", 189)]
    assert FLAG_LIGAND_CHARGE in asp189.flagged_by
    assert FLAG_PKA_MARGIN not in asp189.flagged_by


def test_review_ligand_charge_radius_zero_disables_the_rule(prepared_3ptb_ben_charged):
    """ligand_charge_radius=0.0 must remove exactly Asp189, which is what shows
    the rule, and not the fixture, does the work."""
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb_ben_charged
    with_rule = reviewProtonation(pmol, df, pH=7.4)
    without_rule = reviewProtonation(pmol, df, pH=7.4, ligand_charge_radius=0.0)
    added = {(r.resname, r.resid) for r in with_rule.residues} - {
        (r.resname, r.resid) for r in without_rule.residues
    }
    assert added == {("ASP", 189)}


def test_review_ben_is_not_flagged_by_its_own_charge(prepared_3ptb_ben_charged):
    """Defect A: BEN's own heavy atoms sit trivially within ligand_charge_radius
    of its own amidinium (distance 0), and rule 3 as first written flagged BEN
    by its own charge. A residue's own charge is not context for its own
    decision, the same guard the charge-contact scan already applies.

    Margin is widened just far enough to surface BEN (its margin is 6.41)
    via rule 1, so its flagged_by can be inspected directly. BEN's absence
    from the default-margin report would prove nothing, since it has no
    other evidence once its own charge is correctly excluded."""
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb_ben_charged
    rep = reviewProtonation(pmol, df, pH=7.4, margin=7.0)
    ben = [r for r in rep.residues if r.resname == "BEN"]
    assert ben, "BEN's pKa margin (6.41) must be inside the widened window"
    assert ben[0].flagged_by == (FLAG_PKA_MARGIN,)
    assert FLAG_LIGAND_CHARGE not in ben[0].flagged_by


def test_review_metal_residue_is_not_flagged_by_its_own_charge():
    """Defect A, rule 2's side of the same hole: a metal residue with a
    predicted pKa would coordinate itself at distance zero, since its own
    (single) heavy atom trivially sits within metal_radius of its own charge.
    1R1J's Zn never surfaced this because PROPKA gives it no pKa (NaN), which
    masked the hole rather than proving its absence, so an artificial one
    exercises the guard directly here."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare
    from moleculekit.tools.protonation_review import reviewProtonation

    mol = Molecule(os.path.join(curr_dir, "pdb", "1r1j.pdb"))
    mol.remove("water", _logger=False)
    pmol, _, df = systemPrepare(
        mol, pH=7.4, return_details=True, _logger_level="ERROR"
    )
    df = df.copy()
    zn_mask = df.resname == "ZN"
    assert zn_mask.any(), "1r1j fixture must carry a Zn row in the details table"
    zn_resid = int(df.loc[zn_mask, "resid"].iloc[0])
    # Far outside margin=1.0 of pH 7.4, so only a self-coordination bug could
    # surface this row.
    df.loc[zn_mask, "pKa"] = 20.0

    rep = reviewProtonation(pmol, df, pH=7.4, margin=1.0)
    assert not any(
        r.resname == "ZN" and r.resid == zn_resid for r in rep.residues
    ), "a metal residue must not be flagged by coordinating its own charge"


def test_review_rule_ordering_metal_then_ligand_then_margin(prepared_3ptb_ben_charged):
    """Metal-flagged residues (Glu70, Glu80, via the bound Ca2+) sort ahead of
    the ligand-flagged one (Asp189, via BEN), which sorts ahead of the
    margin-only ones (His57, Tyr39): most decisive evidence first."""
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb_ben_charged
    rep = reviewProtonation(pmol, df, pH=7.4, margin=1.0)

    def rank(flagged_by):
        if FLAG_METAL_CONTACT in flagged_by:
            return 0
        if FLAG_LIGAND_CHARGE in flagged_by:
            return 1
        return 2

    ranks = [rank(r.flagged_by) for r in rep.residues]
    assert ranks == sorted(ranks), "metal, then ligand, then margin was violated"
    assert set(ranks) == {0, 1, 2}, "the ordering claim is vacuous unless all three occur"


@pytest.fixture(scope="module")
def prepared_3ptb_rcsb():
    """3PTB fetched fresh from RCSB, as the how-to guide's recipe does, rather
    than read from the repo's own ``tests/pdb/3ptb.pdb``.

    This is the fixture that actually carries defect B. Verified: freshly
    fetched, 12 disulfide-bonded CYS reach PROPKA still named CYS and come
    back with its not-titrated sentinel, exactly 99.99, giving 52 rows with a
    non-null pKa where only 40 (spanning 2.20 to 13.81, nowhere near 90) are
    real predictions. ``tests/pdb/3ptb.pdb`` does not reproduce this: reading
    it, those same 12 residues are already resolved to CYX before PROPKA
    runs and PROPKA never emits a row for them at all, so they read NaN
    instead of 99.99 and ``notna()`` already excludes them. Both are
    upstream preparation behaviour, not something this module controls; what
    matters here is that the sentinel is real and reachable, and this
    fixture is what reaches it.
    """
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare

    mol = Molecule("3PTB")
    mol.remove("water", _logger=False)
    pmol, _, df = systemPrepare(
        mol, pH=7.4, return_details=True, _logger_level="ERROR"
    )
    return pmol, df


def test_review_sentinel_pka_is_dropped_and_disclosed(prepared_3ptb_rcsb):
    """Defect B, measured directly: n_titratable must read 40, not the 52 rows
    the pKa column carries a non-null value for, and the 12 dropped rows must
    be disclosed rather than silently absorbed into a smaller denominator."""
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb_rcsb
    assert df.pKa.notna().sum() == 52, "3PTB fetched fresh must carry 52 non-null rows"
    real = df.pKa[df.pKa.notna() & (df.pKa.abs() < 90)]
    sentinel = df.pKa[df.pKa.notna() & (df.pKa.abs() >= 90)]
    assert len(real) == 40 and real.abs().max() < 90, "no real prediction approaches 90"
    assert sentinel.unique().tolist() == [99.99], "99.99 must be the only sentinel value"

    rep = reviewProtonation(pmol, df, pH=7.4, margin=1.0)
    assert rep.n_titratable == 40
    assert rep.no_usable_pka == 12
    assert not any(r.pKa == 99.99 for r in rep.residues), (
        "a sentinel pKa must never appear in the report"
    )
    assert (
        "12 residues carry no usable pKa prediction and were not considered"
        in str(rep)
    )


def test_review_sentinel_pka_never_selected_at_any_radius(prepared_3ptb_rcsb):
    """The sentinel rows are dropped before any rule runs, so widening
    metal_radius or ligand_charge_radius enough to reach nearly everything
    still cannot select one: the guard is a row filter, not a distance
    check."""
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb_rcsb
    rep = reviewProtonation(
        pmol, df, pH=7.4, margin=1.0, metal_radius=100.0, ligand_charge_radius=100.0
    )
    assert rep.no_usable_pka == 12
    assert not any(r.pKa == 99.99 for r in rep.residues)


def test_review_sentinel_pka_line_is_singular_for_one(prepared_3ptb):
    """Correct singular/plural agreement, and the abs() in the threshold: a
    negative sentinel is dropped exactly like the positive one PROPKA
    actually emits."""
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb
    cyx = df.resname == "CYX"
    one = df[cyx].index[:1]

    sentinel_df = df.copy()
    sentinel_df.loc[one, "pKa"] = -99.99

    rep = reviewProtonation(pmol, sentinel_df, pH=7.4, margin=1.0)
    assert rep.no_usable_pka == 1
    assert (
        "1 residue carries no usable pKa prediction and was not considered"
        in str(rep)
    )


def test_review_sentinel_line_absent_when_nothing_dropped(prepared_3ptb):
    from moleculekit.tools.protonation_review import reviewProtonation

    pmol, df = prepared_3ptb
    rep = reviewProtonation(pmol, df, pH=7.4, margin=1.0)
    assert rep.no_usable_pka == 0
    assert "no usable pKa prediction" not in str(rep)


def test_describe_environment_on_a_residue():
    """The environment scan is not about protonation, so it must work on any
    residue without a details table or a pH."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "1r1j.pdb"))
    mol.remove("water", _logger=False)
    pmol, _ = systemPrepare(mol, pH=7.4, _logger_level="ERROR")

    env = describeEnvironment(pmol, 'chain "A" and resid 587')
    assert env.contacts, "HIS 587 coordinates a zinc, so it has contacts"
    zn = [c for c in env.contacts if c.resname == "ZN"]
    assert len(zn) == 1
    assert abs(zn[0].distance - 2.04) < 0.05
    metals = [g for g in env.charges if g.label == "metal cation"]
    assert metals and min(g.distance for g in metals) < 2.6
    for c in env.contacts:
        assert c.distance <= 4.0
    for g in env.charges:
        assert g.distance <= 8.0


def test_describe_environment_matches_what_the_review_reports():
    """The load-bearing test for this task. reviewProtonation must delegate to
    describeEnvironment rather than keeping its own copy of the scans: two
    implementations of the same measurement drift apart, and a divergence here
    would mean the report and the standalone call disagree about the same
    residue."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare
    from moleculekit.tools.protonation_review import (
        describeEnvironment,
        reviewProtonation,
    )

    mol = Molecule(os.path.join(curr_dir, "pdb", "1r1j.pdb"))
    mol.remove("water", _logger=False)
    pmol, _, df = systemPrepare(
        mol, pH=7.4, return_details=True, _logger_level="ERROR"
    )
    rep = reviewProtonation(pmol, df, pH=7.4)
    assert rep.residues, "1r1j must flag something"

    for r in rep.residues[:3]:
        mask = (
            (pmol.chain == r.chain)
            & (pmol.resid == r.resid)
            & (pmol.insertion == r.insertion)
        )
        env = describeEnvironment(pmol, mask)
        assert [
            (c.resname, c.resid, c.chain, c.own_atom, c.other_atom, round(c.distance, 4))
            for c in env.contacts
        ] == [
            (c.resname, c.resid, c.chain, c.own_atom, c.other_atom, round(c.distance, 4))
            for c in r.contacts
        ], f"contacts disagree for {r.resname} {r.resid}"
        assert [
            (g.resname, g.resid, g.chain, g.label, round(g.distance, 4))
            for g in env.charges
        ] == [
            (g.resname, g.resid, g.chain, g.label, round(g.distance, 4))
            for g in r.charges
        ], f"charges disagree for {r.resname} {r.resid}"


def test_describe_environment_on_a_ligand():
    """A ligand is not a titratable residue, so this is the case the old
    private scans could not serve."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    env = describeEnvironment(mol, mol.resname == "BEN")
    assert env.contacts, "the benzamidine sits in the trypsin pocket"
    assert not [c for c in env.contacts if c.resname == "BEN"], (
        "the subject must not appear in its own environment"
    )


def test_describe_environment_accepts_a_multi_residue_selection():
    """Any selection means any selection: a binding site, not just a residue."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "1r1j.pdb"))
    mol.remove("water", _logger=False)
    env = describeEnvironment(mol, "resname ZN or (resname OIR)")
    assert env.contacts
    for c in env.contacts:
        assert c.resname not in ("ZN", "OIR"), "subject residues are excluded"


def test_unclassified_line_is_scoped_to_the_subjects_radius():
    """MEASURED on 3PTB: the CA to BEN minimum distance is 22.61 A, against a
    charge radius of 8.0, so BEN could not have contributed a charge CA's
    own environment would ever have shown. Reporting it under CA's heading
    named a residue almost three times the radius away. ASP 189 sits 2.87 A
    from BEN, well inside the radius, so its own report must still mention
    it. Both the rendered line and the ``unclassified`` field are checked, so
    the object and the text cannot disagree."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)

    far = describeEnvironment(mol, "resname CA")
    assert not [u for u in far.unclassified if u[0] == "BEN"]
    assert "BEN" not in str(far)

    near = describeEnvironment(mol, "resid 189")
    assert any(u[0] == "BEN" for u in near.unclassified)
    assert "BEN" in str(near)


def test_the_subject_never_appears_in_its_own_environment():
    """The subject-exclusion invariant, pinned in one place. It was applied to
    the charge scan but not to the selection rules or the unclassified list,
    which is how a ligand came to be reported as its own environment and
    flagged by its own charge."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    # Single-residue subject, untemplated: BEN carries no formal charge, so
    # chargedGroups cannot classify it and it lands in the whole-molecule
    # unclassified list, exercising that path specifically.
    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    env = describeEnvironment(mol, "resname BEN")
    assert not [c for c in env.contacts if c.resname == "BEN"]
    assert not [g for g in env.charges if g.resname == "BEN"]
    assert not [u for u in env.unclassified if u[0] == "BEN"]
    assert "no table entry" not in str(env), (
        "BEN is the only unclassified residue in 3PTB, so once it is "
        "correctly excluded from its own report nothing is left to list"
    )

    # Single-residue subject, templated: BEN now carries a real formal
    # charge, exercising the charge-scan exclusion instead of the
    # unclassified one.
    charged = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    charged.remove("water", _logger=False)
    charged.remove("element H", _logger=False)
    charged.templateResidueFromSmiles(
        "resname BEN", "[NH2+]=C(N)c1ccccc1", addHs=True, _logger=False
    )
    env_t = describeEnvironment(charged, "resname BEN")
    assert not [c for c in env_t.contacts if c.resname == "BEN"]
    assert not [g for g in env_t.charges if g.resname == "BEN"]

    # Multi-residue subject: BEN together with a real protein residue, so an
    # exclusion that only reached the first residue in the selection would
    # leave the other exposed.
    multi = describeEnvironment(mol, "resname BEN or (chain A and resid 57)")
    assert not [c for c in multi.contacts if c.resname == "BEN"]
    assert not [c for c in multi.contacts if c.chain == "A" and c.resid == 57]
    assert not [g for g in multi.charges if g.resname == "BEN"]
    assert not [g for g in multi.charges if g.chain == "A" and g.resid == 57]
    assert not [u for u in multi.unclassified if u[0] == "BEN"]


def test_polymer_backbone_atoms_takes_the_short_path_through_proline():
    """Proline's CD bonds back to its own N, closing a five-membered ring, so
    the residue's internal graph has two routes from N to C: N-CA-C (length
    2) and N-CD-CG-CB-CA-C (length 5, through the ring). The backbone must be
    the shorter one; a shortest-path implementation that wandered into the
    ring here would be wrong, not just imprecise. Built as a small synthetic
    molecule (the codebase's established pattern, see _residue_mask's own
    test) rather than hunted for in a fixture, since a real PDB rarely
    carries the ring-closing CD-N bond explicitly."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import _polymer_backbone_atoms
    import numpy as np

    mol = Molecule().empty(8)
    #        0    1    2     3     4     5     6    7
    #      prevC  N    CA    CB    CG    CD    C   nextN
    mol.name = np.array(
        ["C", "N", "CA", "CB", "CG", "CD", "C", "N"], dtype=object
    )
    mol.chain = np.array(["A"] * 8, dtype=object)
    mol.resid = np.array([1, 2, 2, 2, 2, 2, 2, 3])
    mol.bonds = np.array(
        [
            [0, 1],  # prev residue's C to proline's N (sequence link)
            [1, 2],  # N-CA
            [2, 3],  # CA-CB
            [3, 4],  # CB-CG
            [4, 5],  # CG-CD
            [5, 1],  # CD-N: closes the ring
            [2, 6],  # CA-C
            [6, 7],  # proline's C to next residue's N (sequence link)
        ],
        dtype=np.uint32,
    )

    polymer_keys = {("A", 1), ("A", 2), ("A", 3)}
    backbone_by_name = np.isin(mol.name, ["N", "CA", "C"])
    backbone = _polymer_backbone_atoms(mol, "A", 2, polymer_keys, backbone_by_name)
    assert backbone == {1, 2, 6}, "the path must be N-CA-C, not through the ring"


def test_describe_environment_excludes_sequence_neighbours_of_any_subject_residue():
    """exclude_adjacent generalises: with a multi-residue subject it skips a
    partner adjacent to ANY subject residue, not just the first one."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    sel = 'protein and resid 57 100'
    on = describeEnvironment(mol, sel, exclude_adjacent=True)
    off = describeEnvironment(mol, sel, exclude_adjacent=False)

    subjects = {(57,), (100,)}
    for c in on.contacts:
        assert not any(abs(c.resid - r[0]) <= 1 for r in subjects if c.chain == "A")
    assert len(off.contacts) >= len(on.contacts)


def test_exclude_adjacent_does_not_apply_to_a_ligand():
    """resid arithmetic encodes backbone adjacency, which is meaningless for
    a non-polymer residue. A ligand numbered next to a protein residue must
    not lose that contact."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    ben = mol.atomselect("resname BEN", indexes=True)
    # Renumber the ligand into the protein's range so the adjacency rule
    # would fire if it were applied to a non-polymer residue.
    target = 190
    mol.resid[ben] = target
    env = describeEnvironment(mol, mol.resname == "BEN", exclude_adjacent=True)
    neighbours = [
        c for c in env.contacts
        if c.chain == "A" and c.resid in (target - 1, target + 1)
    ]
    assert neighbours, (
        "a protein residue at resid +/-1 of a LIGAND is a real contact, "
        "not a bonded sequence neighbour"
    )


def test_exclude_adjacent_still_applies_to_a_modified_amino_acid():
    """The mirror case, and the reason the rule is backbone-based rather than
    resname-based: SEP is a phosphoserine IN the peptide chain, absent from
    PROTEIN_RESIDUE_NAMES_WITH_VARIANTS, so a resname gate would wrongly stop
    excluding its real bonded neighbours. The rule is pair-level, so this
    only asserts the backbone-backbone pairs are gone; 1kdx happens to carry
    no sidechain contact to 132 or 134 within range, so none survive here
    (see test_adjacent_sidechain_contacts_survive_the_backbone_exclusion for
    the case where one does)."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "1kdx.pdb"))
    mol.remove("water", _logger=False)
    env = describeEnvironment(
        mol, 'chain "B" and resid 133', exclude_adjacent=True
    )
    assert env.backbone_links_suppressed == 2, "both peptide bonds are dropped"
    assert not [
        c for c in env.contacts if c.chain == "B" and c.resid in (132, 134)
    ], "SEP 133 is in the chain, so 132 and 134 are bonded neighbours"


def test_adjacent_sidechain_contacts_survive_the_backbone_exclusion():
    """The peptide bond is constant and uninformative, but an i+1 sidechain
    reaching the subject is real chemistry. Dropping the whole residue
    discarded both, which is why the rule is now pair-level.

    3PTB chain A: Cys220's backbone carbonyl O reaches Gln221's sidechain
    amide NE2 at 2.92 A, while both of Cys220's peptide bonds (to Gly219 and
    to Ala221's alternate location) are backbone-backbone and are dropped."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    env = describeEnvironment(mol, 'chain "A" and resid 220', exclude_adjacent=True)
    assert env.backbone_links_suppressed == 2

    survivors = [c for c in env.contacts if c.chain == "A" and c.resid == 221]
    assert survivors, "Cys220's backbone O to Gln221's sidechain NE2 must survive"
    assert survivors[0].own_atom == "O" and survivors[0].other_atom == "NE2"
    assert abs(survivors[0].distance - 2.92) < 0.01


def test_suppressed_backbone_links_are_disclosed():
    """A silent omission is the bug this module exists to prevent, so the
    count has to appear in the report and on the object, and only when
    something was actually suppressed."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)

    on = describeEnvironment(mol, 'chain "A" and resid 220', exclude_adjacent=True)
    assert on.backbone_links_suppressed == 2
    assert "2 backbone links to sequence neighbours suppressed" in str(on)

    off = describeEnvironment(mol, 'chain "A" and resid 220', exclude_adjacent=False)
    assert off.backbone_links_suppressed == 0
    assert "suppressed" not in str(off)


def test_adjacent_residue_charges_are_always_reported():
    """exclude_adjacent suppresses backbone covalent geometry in the contact
    table. It must never touch the charge table: a neighbouring carboxylate
    shifts a pKa regardless of being sequence-adjacent, and a free terminal
    charge sits on backbone atoms, so applying the rule there would hide a
    real charge.

    3PTB chain A: Asn79's neighbour Glu80 carries a carboxylate 3.44 A away,
    reported identically with exclude_adjacent True and False."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)

    on = describeEnvironment(mol, 'chain "A" and resid 79', exclude_adjacent=True)
    off = describeEnvironment(mol, 'chain "A" and resid 79', exclude_adjacent=False)

    on_adjacent = [g for g in on.charges if g.chain == "A" and g.resid == 80]
    off_adjacent = [g for g in off.charges if g.chain == "A" and g.resid == 80]
    assert on_adjacent, "Glu80's carboxylate must be reported even though it is adjacent"
    assert on_adjacent == off_adjacent
    assert abs(on_adjacent[0].distance - 3.44) < 0.01


def test_max_contacts_and_max_charges_default_to_no_limit():
    """A cap is a silent truncation unless disclosed, so the default must be
    None: every contact and charge within radius, uncapped."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    env = describeEnvironment(mol, 'chain "A" and resid 220')
    assert len(env.contacts) == 8
    assert len(env.charges) == 2
    assert env.contacts_truncated == 0
    assert env.charges_truncated == 0
    assert "not shown" not in str(env)


def test_a_truncating_cap_is_disclosed():
    """When a cap is supplied and it actually truncates, the count dropped is
    disclosed in the rendered table and on the object, with correct singular
    and plural, and never when a cap did not truncate anything."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    uncapped = describeEnvironment(mol, 'chain "A" and resid 220')
    n_contacts, n_charges = len(uncapped.contacts), len(uncapped.charges)

    # A cap that does not reach the true count truncates nothing.
    loose = describeEnvironment(
        mol, 'chain "A" and resid 220', max_contacts=n_contacts, max_charges=n_charges
    )
    assert loose.contacts_truncated == 0 and loose.charges_truncated == 0
    assert "not shown" not in str(loose)

    # Singular: one contact and one charge cut off.
    singular = describeEnvironment(
        mol,
        'chain "A" and resid 220',
        max_contacts=n_contacts - 1,
        max_charges=n_charges - 1,
    )
    assert singular.contacts_truncated == 1
    assert singular.charges_truncated == 1
    text = str(singular)
    assert "1 further contact within 4.0 A not shown" in text
    assert "1 further charge within 8.0 A not shown" in text

    # Plural: several of each cut off.
    plural = describeEnvironment(
        mol, 'chain "A" and resid 220', max_contacts=3, max_charges=1
    )
    assert plural.contacts_truncated == n_contacts - 3
    assert plural.charges_truncated == n_charges - 1
    text = str(plural)
    assert f"{n_contacts - 3} further contacts within 4.0 A not shown" in text
    assert f"{n_charges - 1} further charge" in text


def test_describe_environment_reuses_precomputed_charged_groups():
    """reviewProtonation scans many residues, so it must be able to compute the
    charged groups once and pass them in rather than rescanning per residue."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.charged_groups import chargedGroups
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    precomputed = chargedGroups(mol)
    a = describeEnvironment(mol, 'protein and resid 57')
    b = describeEnvironment(mol, 'protein and resid 57', charged_groups=precomputed)
    assert [(g.resname, g.resid, round(g.distance, 4)) for g in a.charges] == [
        (g.resname, g.resid, round(g.distance, 4)) for g in b.charges
    ]


def test_describe_environment_renders_its_own_tables():
    """print(env) has to be readable on its own, with the same aligned columns
    the report uses."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "1r1j.pdb"))
    mol.remove("water", _logger=False)
    env = describeEnvironment(mol, 'chain "A" and resid 587')
    text = str(env)
    assert "contacts (<= 4.0 A)" in text
    assert "charges (<= 8.0 A)" in text

    def distance_column(needle, distance):
        # Keyed on the known distance value rather than the line's last
        # token: a row whose pair is bonded now carries a trailing
        # annotation (e.g. HIS 587's NE2-ZN contact, read straight from
        # 1r1j's recorded connectivity), so the last token is not always the
        # distance any more.
        line = next(ln for ln in text.split("\n") if needle in ln)
        return line.index(f"{distance:.2f}")

    contact_lines = [c for c in env.contacts]
    if len(contact_lines) >= 2:
        first = f"{contact_lines[0].own_atom}-{contact_lines[0].other_atom}"
        second = f"{contact_lines[1].own_atom}-{contact_lines[1].other_atom}"
        assert distance_column(first, contact_lines[0].distance) == distance_column(
            second, contact_lines[1].distance
        )


def test_describe_environment_to_dict_is_json_serializable():
    import json
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    env = describeEnvironment(mol, 'protein and resid 57')
    data = env.to_dict()
    back = json.loads(json.dumps(data))
    assert back["contact_radius"] == 4.0
    assert back["charge_radius"] == 8.0
    assert "contacts" in back and "charges" in back


def test_contacts_annotate_calcium_coordination_when_bonds_are_guessed():
    """MEASURED on 3PTB: guessing bonds (mol.bonds = mol._getBonds()) gives
    3126 bonds, of which exactly four touch the calcium, all bondtype "mc",
    and they are precisely the four contacts CA reports: Glu70 OE1, Asn72 O,
    Val75 O, Glu80 OE2. Read from the data, never geometrically detected."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    mol.bonds = mol._getBonds()

    env = describeEnvironment(mol, "resname CA")
    assert len(env.contacts) == 4
    for c in env.contacts:
        assert c.bondtype == "mc"
    # Glu70 and Glu80 are also reported in the charge table, measured to the
    # same carboxylate oxygen the calcium is bonded to, so the annotation
    # appears there too: 4 contact rows plus 2 charge rows.
    metal_charges = [g for g in env.charges if g.bondtype == "mc"]
    assert {g.resname for g in metal_charges} == {"GLU"}
    assert len(metal_charges) == 2
    assert str(env).count("metal coordination") == 6


def test_contacts_carry_no_annotation_when_bonds_are_stripped():
    """An absent annotation means no bond is RECORDED, not that no bond
    exists: with mol.bonds emptied out, the same four coordination contacts
    are still reported (the geometry has not changed), but none is
    annotated."""
    import numpy as np
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    mol.bonds = np.empty((0, 2), dtype=np.uint32)
    mol.bondtype = np.empty((0,), dtype=object)

    env = describeEnvironment(mol, "resname CA")
    assert len(env.contacts) == 4
    for c in env.contacts:
        assert c.bondtype is None
    assert "coordination" not in str(env)


def test_a_close_but_unbonded_contact_carries_no_annotation():
    """Asp189's backbone O reaches Val17's backbone N at 2.77 A, a real
    hydrogen-bond-range contact, but the two atoms carry no recorded bond:
    being close is not being bonded, and the annotation must not claim
    otherwise."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.protonation_review import describeEnvironment

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    mol.remove("water", _logger=False)
    mol.bonds = mol._getBonds()

    env = describeEnvironment(mol, "resid 189")
    val17 = [c for c in env.contacts if c.resname == "VAL" and c.resid == 17]
    assert val17, "Val17's backbone O-N contact to Asp189 must still be reported"
    assert val17[0].own_atom == "O" and val17[0].other_atom == "N"
    assert val17[0].bondtype is None


def test_bond_annotation_does_not_move_the_distance_column():
    """The annotation is appended AFTER the distance for exactly this
    reason: a bonded row must line up with an unannotated one, in both the
    contact and the charge table."""
    from moleculekit.tools.protonation_review import Contact, ChargeContact

    rep = _review()
    rep.residues[0].contacts = [
        Contact("PHE", 589, "", "A", "A", "C", "N", 2.94),
        Contact("ASP", 590, "", "A", "A", "ND1", "OD2", 2.80, bondtype="mc"),
    ]
    rep.residues[0].charges = [
        ChargeContact("ZN", 1001, "", "A", "A", "metal cation", 1, 2.04,
                      source="ion", sign_only=True, bondtype="mc"),
        ChargeContact("ARG", 717, "", "A", "A", "guanidinium", 1, 6.94,
                      source="table"),
    ]
    text = str(rep)
    assert "metal coordination" in text
    lines = text.split("\n")

    contact_idx = [ln.index("2.94") for ln in lines if "C-N" in ln] + [
        ln.index("2.80") for ln in lines if "ND1-OD2" in ln
    ]
    assert len(set(contact_idx)) == 1, "an annotated contact row moved the distance column"

    charge_idx = [ln.index("2.04") for ln in lines if "metal cation" in ln] + [
        ln.index("6.94") for ln in lines if "guanidinium" in ln
    ]
    assert len(set(charge_idx)) == 1, "an annotated charge row moved the distance column"


@pytest.mark.parametrize(
    "bondtype,word",
    [
        ("1", "single"),
        ("2", "double"),
        ("3", "triple"),
        ("ar", "aromatic"),
        ("mc", "metal coordination"),
    ],
)
def test_bond_words_cover_every_type_the_brief_names(bondtype, word):
    from moleculekit.tools.protonation_review import _BOND_TYPE_WORDS

    assert _BOND_TYPE_WORDS.get(bondtype, bondtype) == word


def test_the_protonation_report_rendering_is_unchanged_by_the_refactor():
    """The report's exact text is pinned by column-index tests and by a docs
    page holding byte-identical generated output, so extracting the shared
    renderer must not move a single character."""
    rep = _review()
    text = str(rep)
    lines = text.split("\n")

    contact_idx = [
        ln.index("2.67") for ln in lines if "ND1-OD2" in ln
    ] + [ln.index("2.96") for ln in lines if "NE2-OG" in ln]
    assert len(set(contact_idx)) == 1, "contact distance column moved"

    assert lines[0].startswith(
        "PROTONATION TO CONFIRM (pH 7.4, margin 1.0, metal 2.6 A, ligand 4.0 A)"
    )
    assert "    contacts (<= 4.0 A)" in text, "report table indent changed"
    assert "      ASP   102 A" in text, "report row indent changed"


# Prepared-output fingerprints captured on the merge base, before any of this
# work landed. This suite's whole point is that nothing here touches a build
# path; these two values are what say so rather than assert it.
_PREPARED_FINGERPRINTS = {
    "3ptb": (3231, "3c84f87c9d7bb713d50aa27c215939a882884384a8bf850ccc5f5565bf572e95"),
    "1r1j": (11160, "d35ab4f56750d261f01d37e93f2b53687ede2129a8bd15c84ab398d27b89e458"),
}


@pytest.mark.parametrize("pid", sorted(_PREPARED_FINGERPRINTS))
def test_preparation_output_is_unchanged(pid):
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare
    import hashlib

    mol = Molecule(os.path.join(curr_dir, "pdb", f"{pid}.pdb"))
    mol.remove("water", _logger=False)
    pmol, _ = systemPrepare(mol, pH=7.4, _logger_level="ERROR")

    h = hashlib.sha256()
    for fld in ("name", "resname", "resid", "insertion", "chain", "segid", "element"):
        h.update(np.asarray(getattr(pmol, fld)).astype(str).tobytes())
    h.update(np.round(pmol.coords, 3).tobytes())

    want_atoms, want_hash = _PREPARED_FINGERPRINTS[pid]
    assert int(pmol.numAtoms) == int(want_atoms)
    assert h.hexdigest() == want_hash, (
        "preparation output moved. Nothing in the protonation-review work "
        "should touch a build path, so this is a real regression, not a "
        "fingerprint to update."
    )
