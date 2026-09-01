import os
import numpy as np
import pytest
from moleculekit.molecule import Molecule
from moleculekit.tools.modelling import (
    _align_full_to_observed,
    detectBackboneBreaks,
    detectSplicedClashes,
    detectSequenceGaps,
    prepareGapModellingInput,
    spliceMissingResidues,
)
from moleculekit.tools.modelling import _pair_donor_chains, _sequence_identity
from moleculekit.tools.modelling import _fit_donor_to_orig, _apply_fit

curr_dir = os.path.dirname(os.path.abspath(__file__))


def test_fit_donor_to_orig_recovers_known_rigid_transform():
    rng = np.random.default_rng(0)
    orig = rng.normal(size=(8, 3))
    # a known rotation (90 deg about z) + translation applied to make the "donor"
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([5.0, -3.0, 2.0])
    donor = orig @ R.T + t
    fit, rmsd = _fit_donor_to_orig(donor, orig)
    recovered = _apply_fit(donor, fit)
    assert np.allclose(recovered, orig, atol=1e-6)
    assert rmsd < 1e-6   # a genuine rigid image fits with ~zero residual


def test_align_full_to_observed_marks_internal_gap():
    full = "ACDEFGHIK"
    observed = "ACDEHIK"  # missing F,G (positions 5-6)
    af, ao = _align_full_to_observed(full, observed)
    assert af == "ACDEFGHIK"
    assert ao == "ACDE--HIK"


def _chain_mol(resnames, resids, chain="A", segid="P", insertions=None):
    # moleculekit's "protein" atomselect needs a full N-CA-C-O backbone with
    # guessed bonds (a lone CA per residue is not recognized as protein).
    names = ["N", "CA", "C", "O"] * len(resnames)
    m = Molecule().empty(len(names))
    m.name[:] = names
    m.resname[:] = np.repeat(list(resnames), 4)
    m.resid[:] = np.repeat(list(resids), 4)
    if insertions is not None:
        m.insertion[:] = np.repeat(list(insertions), 4)
    m.chain[:] = chain
    m.segid[:] = segid
    m.record[:] = "ATOM"
    m.element[:] = [n[0] for n in names]
    coords = np.zeros((len(names), 3), dtype=np.float32)
    coords[:, 0] = np.arange(len(names)) * 1.45
    m.coords = coords.reshape(len(names), 3, 1)
    m.guessBonds()
    return m


def _bonded_chain_mol(resnames, resids, chain="A", segid="P", shift=None):
    """Like :func:`_chain_mol` but with realistic peptide-bond geometry: residues
    whose resids are consecutive come out bonded (C-N ~1.33 A) while a resid jump
    leaves a genuine backbone break. ``_chain_mol``'s backbone is stretched between
    every residue, so it cannot distinguish "residues are missing here" from "the
    backbone is continuous here" - which is exactly what gap placement depends on.

    The residues are threaded along a smooth curve whose two frequencies are
    incommensurate, so no shift of the chain along it superposes back onto itself
    (a straight or helical backbone would, letting a mis-paired alignment fit with a
    deceptively low RMSD)."""
    t = np.linspace(0, 100, 100001)
    curve = np.stack([t, 3.0 * np.sin(0.7 * t), 2.0 * np.cos(0.31 * t)], axis=1)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])

    def at(s):
        return np.stack([np.interp(s, arc, curve[:, k]) for k in range(3)])

    # N-CA 1.45, CA-C 1.52, C-N(next) 1.33 -> 4.30 A of backbone per residue, so a
    # resid jump of k leaves k-1 residues' worth of unbridged backbone behind
    coords, names = [], []
    s = 0.0
    for k, resid in enumerate(resids):
        if k:
            s += 4.30 * (resid - resids[k - 1])
        n, ca, c = at(s), at(s + 1.45), at(s + 2.97)
        tangent = at(s + 2.98) - at(s + 2.96)
        tangent /= np.linalg.norm(tangent)
        perp = np.cross(tangent, [0.0, 0.0, 1.0])
        o = c + 1.23 * perp / np.linalg.norm(perp)
        coords += [n, ca, c, o]
        names += ["N", "CA", "C", "O"]

    m = Molecule().empty(len(names))
    m.name[:] = names
    m.resname[:] = np.repeat(list(resnames), 4)
    m.resid[:] = np.repeat(list(resids), 4)
    m.chain[:] = chain
    m.segid[:] = segid
    m.record[:] = "ATOM"
    m.element[:] = [n[0] for n in names]
    m.coords = np.array(coords, dtype=np.float32).reshape(len(names), 3, 1)
    if shift is not None:
        for resid, vec in shift.items():
            m.coords[m.resid == resid, :, 0] += np.array(vec, dtype=np.float32)
    m.guessBonds()
    return m


# A chain whose missing residues sit in TWO nearby holes. BLOSUM62's -11 gap-open
# makes one merged gap cheaper than two real ones, so a pure-sequence alignment
# slides the observed PRO/ALA onto the wrong donor residues:
#     donor     K K K P P P A E W W W
#     observed  K K K P - - - A W W W   <- one gap, ALA paired with GLU
# while the deposited backbone is only broken after resid 3 and resid 7:
#     observed  K K K - - P A - W W W   <- two gaps, every residue paired correctly
_MERGE_FULL_SEQ = "KKKPPPAEWWW"
_MERGE_FULL_RESNAMES = ["LYS", "LYS", "LYS", "PRO", "PRO", "PRO",
                        "ALA", "GLU", "TRP", "TRP", "TRP"]
_MERGE_OBS_RESNAMES = ["LYS", "LYS", "LYS", "PRO", "ALA", "TRP", "TRP", "TRP"]
_MERGE_OBS_RESIDS = [1, 2, 3, 6, 7, 9, 10, 11]


def test_bonded_chain_mol_has_real_bonds_and_real_breaks():
    # the fixture itself must hold up: bonded where resids are consecutive, broken
    # where they jump, otherwise the gap-placement tests below prove nothing
    m = _bonded_chain_mol(_MERGE_OBS_RESNAMES, _MERGE_OBS_RESIDS)
    _, idx = m.getSequence(dict_key="chain", return_idx=True, sel="protein")

    def cn(k):
        c = idx["A"][k][m.name[idx["A"][k]] == "C"][0]
        n = idx["A"][k + 1][m.name[idx["A"][k + 1]] == "N"][0]
        return float(np.linalg.norm(m.coords[c, :, 0] - m.coords[n, :, 0]))

    bonded = [cn(k) for k in (0, 1, 3, 5, 6)]   # resids 1-2, 2-3, 6-7, 9-10, 10-11
    broken = [cn(k) for k in (2, 4)]            # resids 3->6 and 7->9
    assert max(bonded) < 1.6, bonded
    assert min(broken) > 2.0, broken


def test_align_full_to_observed_confines_gaps_to_backbone_breaks():
    # unconstrained, the aligner merges the two holes into one and mis-pairs
    af, ao = _align_full_to_observed(_MERGE_FULL_SEQ, "KKKPAWWW")
    assert af == _MERGE_FULL_SEQ
    assert ao == "KKKP---AWWW"
    # told where the backbone is actually broken, it splits them correctly
    af, ao = _align_full_to_observed(_MERGE_FULL_SEQ, "KKKPAWWW", breaks={3, 5})
    assert af == _MERGE_FULL_SEQ
    assert ao == "KKK--PA-WWW"


def test_detect_sequence_gaps_splits_holes_at_backbone_breaks():
    m = _bonded_chain_mol(_MERGE_OBS_RESNAMES, _MERGE_OBS_RESIDS)
    gaps, skipped, _ = detectSequenceGaps(m, {"A": _MERGE_FULL_SEQ})
    assert skipped == []
    internal = [
        (g["after_resid"], g["before_resid"], g["missing_seq"])
        for g in gaps
        if not g["is_terminal"]
    ]
    # the two holes are reported where the backbone is broken, NOT merged into a
    # single "PPA" run hanging off resid 6
    assert internal == [(3, 6, "PP"), (7, 9, "E")]


def test_splice_places_runs_at_backbone_breaks():
    # donor and original share coordinates for the co-observed residues, so a
    # correctly paired alignment superposes exactly and every hole must fill
    donor = _bonded_chain_mol(_MERGE_FULL_RESNAMES, list(range(1, 12)), chain="A")
    gapped = _bonded_chain_mol(_MERGE_OBS_RESNAMES, _MERGE_OBS_RESIDS, chain="A")

    spliced, new_mask = spliceMissingResidues(gapped, donor, {"A": "A"})

    ca = spliced.name == "CA"
    assert [int(r) for r in spliced.resid[ca]] == list(range(1, 12))
    assert [str(n) for n in spliced.resname[ca]] == _MERGE_FULL_RESNAMES
    assert [str(i) for i in spliced.insertion[ca]] == [""] * 11
    # only the two holes are new; the deposited residues are kept as they were
    assert [int(r) for r in spliced.resid[new_mask & ca]] == [4, 5, 8]


def test_splice_fills_runs_separated_by_a_single_grafted_residue():
    # Two holes separated by ONE co-observed residue (resid 6), whose deposited
    # position is displaced enough that closing the first hole's C-side junction
    # requires grafting resid 6 from the donor. That leaves the first run's C-side
    # abutting the SECOND run, whose residues are still in the donor's own frame:
    # measuring a junction against them compares frames, not atoms.
    full = ["MET", "LYS", "THR", "TRP", "ASP", "GLU",
            "TYR", "ARG", "LEU", "ASN", "GLY"]
    donor = _bonded_chain_mol(full, list(range(1, 12)), chain="A")
    # a donor arrives in its own arbitrary frame, as a prediction does
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    donor.coords[:, :, 0] = donor.coords[:, :, 0] @ rot.T + [50.0, -20.0, 10.0]

    gapped = _bonded_chain_mol(
        ["MET", "LYS", "THR", "GLU", "LEU", "ASN", "GLY"],
        [1, 2, 3, 6, 9, 10, 11],
        chain="A",
        shift={6: (0.0, -2.0, 0.0)},
    )

    spliced, new_mask = spliceMissingResidues(gapped, donor, {"A": "A"})

    ca = spliced.name == "CA"
    assert [int(r) for r in spliced.resid[ca]] == list(range(1, 12))
    assert [str(n) for n in spliced.resname[ca]] == full
    assert _worst_bonded_junction(spliced) < 1.6


def test_sequence_identity_full_match_and_mismatch():
    # observed AGS is an exact sub-sequence of full AGHIS -> identity 1.0
    assert _sequence_identity("AGHIS", "AG--S") == 1.0
    # one mismatch out of three observed -> 2/3
    assert abs(_sequence_identity("AGHIS", "AX--S") - (2 / 3)) < 1e-9


def test_pair_donor_chains_ignores_labels():
    # same sequence, deliberately different chain labels/segids on the donor
    orig = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5], chain="A", segid="P")
    donor = _chain_mol(["ALA", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5],
                       chain="Z", segid="9")
    pairing, unpaired = _pair_donor_chains(orig, donor)
    assert pairing == {"A": "Z"}
    assert unpaired == []


def test_pair_donor_chains_below_threshold_is_unpaired():
    orig = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 3], chain="A")
    donor = _chain_mol(["TRP", "TRP", "TRP"], [1, 2, 3], chain="B")  # nothing in common
    pairing, unpaired = _pair_donor_chains(orig, donor, min_identity=0.95)
    assert pairing == {}
    assert unpaired == ["A"]


def test_pair_donor_chains_override_resolves_segid():
    # aceboltz-shaped donor: chain "A", label lives in segid "0"
    orig = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5], chain="A", segid="P")
    donor = _chain_mol(["ALA", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5],
                       chain="A", segid="0")
    pairing, unpaired = _pair_donor_chains(orig, donor, chain_map={"0": "A"})
    assert pairing == {"A": "A"}


def test_pair_donor_chains_warns_on_unknown_override_target(caplog):
    # chain_map targets original chain "Z", which the structure does not have: the
    # override is silently unusable, so it must warn (and auto-pairing still runs).
    orig = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5], chain="A", segid="P")
    donor = _chain_mol(["ALA", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5],
                       chain="A", segid="0")
    with caplog.at_level("WARNING"):
        pairing, unpaired = _pair_donor_chains(orig, donor, chain_map={"0": "Z"})
    assert any("no such protein chain" in r.message for r in caplog.records)
    assert pairing == {"A": "A"}   # real chain A still auto-pairs by sequence


def test_detect_sequence_gaps_internal():
    # observed ALA GLY SER (resid 1,2,5) missing 2 residues (resid 3,4) -> internal gap
    m = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5])
    full = {"A": "AGHIS"}  # A G [H I] S  -> observed A G S, missing HI
    gaps, skipped, _ = detectSequenceGaps(m, full)
    assert skipped == []
    assert len(gaps) == 1
    g = gaps[0]
    assert g["chain"] == "A"
    assert g["missing_seq"] == "HI"
    assert g["after_resid"] == 2
    assert g["before_resid"] == 5
    assert g["is_terminal"] is False


def test_detect_sequence_gaps_terminal():
    m = _chain_mol(["GLY", "SER"], [3, 4])  # observed G S at resid 3,4
    full = {"A": "AAGS"}  # missing A A at the N-terminus
    gaps, skipped, _ = detectSequenceGaps(m, full)
    assert len(gaps) == 1
    assert gaps[0]["missing_seq"] == "AA"
    assert gaps[0]["after_resid"] is None
    assert gaps[0]["is_terminal"] is True


def test_prepare_gap_modelling_input(tmp_path):
    m = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5])  # observed AGS
    sequences = {"A": "AGHIS"}
    gaps = [
        {
            "chain": "A",
            "after_resid": 2,
            "before_resid": 5,
            "missing_seq": "HI",
            "is_terminal": False,
        }
    ]
    fasta, template, chain_map = prepareGapModellingInput(
        m, sequences, gaps, str(tmp_path)
    )
    # FASTA: chain A desired sequence = observed + selected loop = AGHIS
    recs = [l.strip() for l in open(fasta) if l.strip() and not l.startswith(">")]
    assert recs == ["AGHIS"]
    assert chain_map == {"0": "A"}
    assert os.path.exists(template)


def test_splice_round_trip_reconstructs():
    # full ALA-GLY-HIS-ILE-SER; the gapped input keeps its deposited numbering
    # 10,11,14 (HIS,ILE missing at 12,13) so the test proves preservation, not
    # renumber-from-1.
    predicted = _chain_mol(["ALA", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5])
    gapped = _chain_mol(["ALA", "GLY", "SER"], [10, 11, 14])
    spliced, new_mask = spliceMissingResidues(gapped, predicted, {"A": "A"}, graft_flanks=0)

    # chain A reconstructed as ALA-GLY-HIS-ILE-SER with ORIGINAL numbering kept and
    # the two new residues filling the natural integer gap (12, 13).
    ca = (spliced.name == "CA") & (spliced.chain == "A")
    assert list(spliced.resname[ca]) == ["ALA", "GLY", "HIS", "ILE", "SER"]
    assert list(spliced.resid[ca]) == [10, 11, 12, 13, 14]
    # exactly the two missing residues were inserted (4 atoms each -> 2 residues).
    assert set(spliced.resname[new_mask]) == {"HIS", "ILE"}
    assert sorted(np.unique(spliced.resid[new_mask])) == [12, 13]


def test_splice_tight_gap_uses_insertion_codes():
    # observed ALA-GLY-SER numbered consecutively 1,2,3 but the full sequence has
    # HIS,ILE between GLY(2) and SER(3): no integer room, so insertion codes.
    predicted = _chain_mol(["ALA", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5])
    gapped = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 3])
    spliced, new_mask = spliceMissingResidues(gapped, predicted, {"A": "A"}, graft_flanks=0)

    ca = (spliced.name == "CA") & (spliced.chain == "A")
    assert list(spliced.resname[ca]) == ["ALA", "GLY", "HIS", "ILE", "SER"]
    assert list(spliced.resid[ca]) == [1, 2, 2, 2, 3]        # anchored on resid 2
    assert list(spliced.insertion[ca]) == ["", "", "A", "B", ""]
    assert set(spliced.resname[new_mask]) == {"HIS", "ILE"}


def test_splice_uses_integer_room_after_insertion_coded_flank():
    # the residue before the gap carries an insertion code (2A) but there IS integer
    # room (3,4,5) before the next observed residue at 6, so the integers are used.
    predicted = _chain_mol(
        ["ALA", "GLY", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5, 6]
    )
    gapped = _chain_mol(
        ["ALA", "GLY", "GLY", "SER"], [1, 2, 2, 6], insertions=["", "", "A", ""]
    )
    spliced, new_mask = spliceMissingResidues(
        gapped, predicted, {"A": "A"}, graft_flanks=0
    )

    ca = (spliced.name == "CA") & (spliced.chain == "A")
    assert list(spliced.resname[ca]) == ["ALA", "GLY", "GLY", "HIS", "ILE", "SER"]
    assert list(spliced.resid[ca]) == [1, 2, 2, 3, 4, 6]
    assert list(spliced.insertion[ca]) == ["", "", "A", "", "", ""]
    assert set(spliced.resname[new_mask]) == {"HIS", "ILE"}


def test_splice_insertion_codes_continue_after_coded_flank():
    # no integer room after 2A (the next observed residue is 3), so insertion codes
    # are used - and they must continue at B, not restart at A and duplicate 2A.
    predicted = _chain_mol(
        ["ALA", "GLY", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5, 6]
    )
    gapped = _chain_mol(
        ["ALA", "GLY", "GLY", "SER"], [1, 2, 2, 3], insertions=["", "", "A", ""]
    )
    spliced, new_mask = spliceMissingResidues(
        gapped, predicted, {"A": "A"}, graft_flanks=0
    )

    ca = (spliced.name == "CA") & (spliced.chain == "A")
    assert list(spliced.resname[ca]) == ["ALA", "GLY", "GLY", "HIS", "ILE", "SER"]
    assert list(spliced.resid[ca]) == [1, 2, 2, 2, 2, 3]
    assert list(spliced.insertion[ca]) == ["", "", "A", "B", "C", ""]
    assert set(spliced.resname[new_mask]) == {"HIS", "ILE"}


def _long_loop_donor():
    # 30 missing residues between GLY and SER: more than the 26 insertion codes.
    loop = ["HIS", "ILE"] * 15
    return _chain_mol(["ALA", "GLY"] + loop + ["SER"], list(range(1, 34)))


def test_splice_overflowing_run_shifts_following_residues():
    # observed 1,2,3 leaves no integer room and 30 residues do not fit in insertion
    # codes, so the residues following the run shift up to open integer room.
    gapped = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 3])
    spliced, new_mask = spliceMissingResidues(
        gapped, _long_loop_donor(), {"A": "A"}, graft_flanks=0
    )

    ca = (spliced.name == "CA") & (spliced.chain == "A")
    # run numbered 3..32 as plain integers; the trailing SER moved 3 -> 33
    assert list(spliced.resid[ca]) == [1, 2] + list(range(3, 33)) + [33]
    assert set(spliced.insertion[ca]) == {""}  # integers only, no insertion codes
    assert sorted(np.unique(spliced.resid[new_mask])) == list(range(3, 33))


def test_splice_shift_warns_naming_the_renumbered_residues(caplog):
    gapped = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 3])
    with caplog.at_level("WARNING"):
        spliceMissingResidues(gapped, _long_loop_donor(), {"A": "A"}, graft_flanks=0)
    # the one moved original (SER 3 -> 33) is named with its deposited number
    assert any(
        "renumbered" in r.message and "3 -> 33 (+30)" in r.message
        for r in caplog.records
    )


def test_splice_two_overflowing_runs_accumulate_shifts():
    # two 30-residue runs in one chain: the second run's shift stacks on the first.
    loop = ["HIS", "ILE"] * 15
    predicted = _chain_mol(
        ["ALA", "GLY"] + loop + ["TRP"] + loop + ["SER"], list(range(1, 65))
    )
    gapped = _chain_mol(["ALA", "GLY", "TRP", "SER"], [1, 2, 3, 4])
    spliced, _ = spliceMissingResidues(gapped, predicted, {"A": "A"}, graft_flanks=0)

    ca = (spliced.name == "CA") & (spliced.chain == "A")
    # run 1 -> 3..32 moves TRP 3->33 and SER 4->34; run 2 -> 34..63 moves SER again
    assert list(spliced.resid[ca]) == (
        [1, 2] + list(range(3, 33)) + [33] + list(range(34, 64)) + [64]
    )
    assert set(spliced.insertion[ca]) == {""}


def test_splice_shift_preserves_room_of_later_gap():
    # a shifted chain must not hand extra room to a later gap: the 2-residue gap that
    # follows the shifted run still has zero integer room, so it still uses codes.
    # The TRP block between the two gaps is 3 residues wide so the aligner cannot
    # merge them by mismatching a lone anchor residue.
    loop = ["HIS", "ILE"] * 15
    predicted = _chain_mol(
        ["ALA"] + loop + ["TRP", "TRP", "TRP", "PHE", "PHE", "SER"],
        list(range(1, 38)),
    )
    gapped = _chain_mol(["ALA", "TRP", "TRP", "TRP", "SER"], [1, 2, 3, 4, 5])
    spliced, _ = spliceMissingResidues(gapped, predicted, {"A": "A"}, graft_flanks=0)

    ca = (spliced.name == "CA") & (spliced.chain == "A")
    # run 1 -> 2..31 moves TRP 2,3,4 -> 32,33,34 and SER 5 -> 35; the 2-residue run
    # then still has zero room between 34 and 35, so it gets 34A, 34B
    assert list(spliced.resid[ca]) == (
        [1] + list(range(2, 32)) + [32, 33, 34, 34, 34, 35]
    )
    assert list(spliced.insertion[ca])[-4:] == ["", "A", "B", ""]


def test_splice_shift_moves_colliding_same_chain_ligand():
    # a metal sharing the protein's chain sits exactly where the shifted numbering
    # lands (resid 33), so it is moved above the chain maximum instead of duplicating.
    gapped = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 3])
    zn = Molecule().empty(1)
    zn.name[:] = ["ZN"]
    zn.resname[:] = ["ZN"]
    zn.resid[:] = [33]
    zn.chain[:] = "A"
    zn.segid[:] = "I"
    zn.record[:] = "HETATM"
    zn.element[:] = ["Zn"]
    zn.coords = np.array([50.0, 0.0, 0.0], dtype=np.float32).reshape(1, 3, 1)
    gapped.append(zn)

    spliced, new_mask = spliceMissingResidues(
        gapped, _long_loop_donor(), {"A": "A"}, graft_flanks=0
    )

    zn_sel = spliced.resname == "ZN"
    assert zn_sel.sum() == 1
    assert not bool(new_mask[zn_sel][0])
    assert float(spliced.coords[zn_sel][0, 0, 0]) == 50.0  # coordinates untouched
    assert int(spliced.resid[zn_sel][0]) == 34  # chain max was 33, so it goes to 34
    # the protein numbering is what it would have been without the ligand
    ca = (spliced.name == "CA") & (spliced.chain == "A")
    assert list(spliced.resid[ca]) == [1, 2] + list(range(3, 33)) + [33]


def test_splice_numbering_fills_integer_gap():
    # graft_flanks=0 (insert-all, no junction check) numbers the inserted run in the
    # natural integer gap between the flanking original residues. (Junction geometry
    # is covered by the real-data tests; this pins the numbering.)
    gapped = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5])
    donor = _chain_mol(["ALA", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5])
    spliced, new_mask = spliceMissingResidues(gapped, donor, graft_flanks=0)
    # loop inserted between GLY(2) and SER(5) -> new residues fill the gap at 3,4
    assert set(spliced.resname[new_mask]) == {"HIS", "ILE"}
    assert sorted(np.unique(spliced.resid[new_mask])) == [3, 4]


def test_splice_c_terminal_tail_numbers_upward():
    # observed ALA-GLY (1,2); full ALA-GLY-HIS-ILE adds a C-terminal tail.
    predicted = _chain_mol(["ALA", "GLY", "HIS", "ILE"], [1, 2, 3, 4])
    gapped = _chain_mol(["ALA", "GLY"], [1, 2])
    spliced, new_mask = spliceMissingResidues(gapped, predicted, {"A": "A"}, graft_flanks=0)

    ca = (spliced.name == "CA") & (spliced.chain == "A")
    assert list(spliced.resname[ca]) == ["ALA", "GLY", "HIS", "ILE"]
    assert list(spliced.resid[ca]) == [1, 2, 3, 4]           # 3,4 count up from 2
    assert set(spliced.resname[new_mask]) == {"HIS", "ILE"}


def test_splice_n_terminal_tail_numbers_up_to_first():
    # observed GLY-SER at 3,4; full ALA-ALA-GLY-SER adds an N-terminal tail.
    predicted = _chain_mol(["ALA", "ALA", "GLY", "SER"], [1, 2, 3, 4])
    gapped = _chain_mol(["GLY", "SER"], [3, 4])
    spliced, new_mask = spliceMissingResidues(gapped, predicted, {"A": "A"}, graft_flanks=0)

    ca = (spliced.name == "CA") & (spliced.chain == "A")
    assert list(spliced.resname[ca]) == ["ALA", "ALA", "GLY", "SER"]
    assert list(spliced.resid[ca]) == [1, 2, 3, 4]           # 1,2 count up to 3
    assert int(new_mask.sum()) == 8                          # 2 new residues x 4 atoms


def test_splice_preserves_non_protein():
    predicted = _chain_mol(["ALA", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5])
    gapped = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5])
    zn = Molecule().empty(1)
    zn.name[:] = ["ZN"]
    zn.resname[:] = ["ZN"]
    zn.resid[:] = [1]
    zn.chain[:] = "B"
    zn.segid[:] = "I"
    zn.record[:] = "HETATM"
    zn.element[:] = ["Zn"]
    zn.coords = np.array([99.0, 0.0, 0.0], dtype=np.float32).reshape(1, 3, 1)
    gapped.append(zn)

    spliced, new_mask = spliceMissingResidues(gapped, predicted, {"A": "A"}, graft_flanks=0)

    # the ZN ion (a non-modelled chain) is preserved, not marked new, coords intact.
    zn_sel = spliced.resname == "ZN"
    assert zn_sel.sum() == 1
    assert not bool(new_mask[zn_sel][0])
    assert float(spliced.coords[zn_sel][0, 0, 0]) == 99.0
    # chain A is still fully reconstructed alongside it.
    ca = (spliced.name == "CA") & (spliced.chain == "A")
    assert list(spliced.resname[ca]) == ["ALA", "GLY", "HIS", "ILE", "SER"]


def test_detect_modelled_clashes_reports_ion_overlap():
    # one modelled protein CA at origin, a ZN ion 1.0 A away, a distant water
    m = Molecule().empty(3)
    m.name[:] = ["CA", "ZN", "O"]
    m.resname[:] = ["HIS", "ZN", "HOH"]
    m.resid[:] = [3, 900, 901]
    m.chain[:] = ["A", "B", "C"]
    m.segid[:] = ["P", "I", "W"]
    m.record[:] = ["ATOM", "HETATM", "HETATM"]
    m.element[:] = ["C", "Zn", "O"]
    m.coords = np.array([[0, 0, 0], [1.0, 0, 0], [50, 0, 0]], dtype=np.float32).reshape(3, 3, 1)
    new_mask = np.array([True, False, False])
    clashes = detectSplicedClashes(m, new_mask, cutoff=2.0)
    assert len(clashes) == 1
    assert clashes[0]["target_resname"] == "ZN"
    assert clashes[0]["new_resid"] == 3
    assert abs(clashes[0]["min_distance"] - 1.0) < 1e-3


def test_detect_modelled_clashes_clean():
    m = Molecule().empty(2)
    m.name[:] = ["CA", "ZN"]; m.resname[:] = ["HIS", "ZN"]
    m.resid[:] = [3, 900]; m.chain[:] = ["A", "B"]; m.segid[:] = ["P", "I"]
    m.record[:] = ["ATOM", "HETATM"]; m.element[:] = ["C", "Zn"]
    m.coords = np.array([[0, 0, 0], [50, 0, 0]], dtype=np.float32).reshape(2, 3, 1)
    assert detectSplicedClashes(m, np.array([True, False]), cutoff=2.0) == []


def test_splice_preserves_same_chain_ligand():
    # a ligand/metal sharing the protein's chain id (the common PDB arrangement)
    # must survive the splice - not just one on a separate chain.
    predicted = _chain_mol(["ALA", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5])
    gapped = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5])
    zn = Molecule().empty(1)
    zn.name[:] = ["ZN"]
    zn.resname[:] = ["ZN"]
    zn.resid[:] = [900]
    zn.chain[:] = "A"
    zn.segid[:] = "A"
    zn.record[:] = "HETATM"
    zn.element[:] = ["Zn"]
    zn.coords = np.array([50.0, 0.0, 0.0], dtype=np.float32).reshape(1, 3, 1)
    gapped.append(zn)

    spliced, new_mask = spliceMissingResidues(gapped, predicted, {"A": "A"}, graft_flanks=0)

    zn_sel = spliced.resname == "ZN"
    assert zn_sel.sum() == 1                          # ligand kept, not dropped
    assert not bool(new_mask[zn_sel][0])              # not marked as newly modelled
    assert float(spliced.coords[zn_sel][0, 0, 0]) == 50.0  # coordinates untouched
    # the protein chain is still fully reconstructed alongside the ligand
    ca = (spliced.name == "CA") & (spliced.chain == "A")
    assert list(spliced.resname[ca]) == ["ALA", "GLY", "HIS", "ILE", "SER"]


def test_detect_sequence_gaps_skips_ncaa_chain():
    # NLE (norleucine) is a protein residue but not canonical, so the whole chain
    # is skipped and reported, never gap-modelled.
    m = _chain_mol(["ALA", "GLY", "NLE"], [1, 2, 3])
    gaps, skipped, _ = detectSequenceGaps(m, {"A": "AGXAA"})
    assert skipped == ["A"]
    assert gaps == []


def test_prepare_gap_modelling_input_raises_on_unlocatable_gap(tmp_path):
    m = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5])
    sequences = {"A": "AGHIS"}
    # a selected gap whose (after_resid, missing_seq) is not in the alignment
    bogus = [{"chain": "A", "after_resid": 99, "before_resid": 100,
              "missing_seq": "WW", "is_terminal": False}]
    with pytest.raises(RuntimeError):
        prepareGapModellingInput(m, sequences, bogus, str(tmp_path))


# Real case: EGFR kinase (1M17) with erlotinib (AQ4) bound.
#
# `1m17_gapped.pdb.gz` is the deposited crystal (protein chain A 672-995 with the
# 965-976 loop `LPSPTDSNFYRA` missing, plus erlotinib and 20 crystallographic
# waters, all on chain A). `1m17_full_sequence.txt` is the full deposited sequence.
# `1m17_model.pdb.gz` is a real aceboltz `gapmodel` output (chain A, the full
# construct: N-His-tag `GSHMAS`, the observed sequence with the loop, C-tail `QQG`).


def _full_sequence():
    with open(os.path.join(curr_dir, "test_modelling", "1m17_full_sequence.txt")) as fh:
        return fh.read().strip()


def test_1m17_real_case_detects_gaps():
    gapped = Molecule(os.path.join(curr_dir, "test_modelling", "1m17_gapped.pdb.gz"))
    gaps, skipped, _ = detectSequenceGaps(gapped, {"A": _full_sequence()})
    assert skipped == []

    internal = [g for g in gaps if not g["is_terminal"]]
    assert len(internal) == 1
    g = internal[0]
    assert g["chain"] == "A"
    assert g["after_resid"] == 964
    assert g["before_resid"] == 977
    assert g["missing_seq"] == "LPSPTDSNFYRA"

    # the two terminal tails are detected and classified separately
    terminal = [g for g in gaps if g["is_terminal"]]
    assert {g["missing_seq"] for g in terminal} == {"GSHMAS", "QQG"}


def test_1m17_real_case_splice_preserves_ligand_and_waters():
    gapped = Molecule(os.path.join(curr_dir, "test_modelling", "1m17_gapped.pdb.gz"))
    model = Molecule(os.path.join(curr_dir, "test_modelling", "1m17_model.pdb.gz"))

    assert int((gapped.resname == "AQ4").sum()) == 29   # erlotinib
    assert int(gapped.atomselect("water").sum()) == 20  # crystallographic waters

    spliced, new_mask = spliceMissingResidues(gapped, model, {"A": "A"})

    # the internal loop is inserted with the ORIGINAL author numbering 965-976
    new_ca = new_mask & (spliced.name == "CA")
    loop = [(int(r), str(n)) for r, n in zip(spliced.resid[new_ca], spliced.resname[new_ca])
            if 965 <= int(r) <= 976]
    assert [r for r, _ in loop] == list(range(965, 977))
    assert [n for _, n in loop] == [
        "LEU", "PRO", "SER", "PRO", "THR", "ASP",
        "SER", "ASN", "PHE", "TYR", "ARG", "ALA",
    ]  # LPSPTDSNFYRA

    # the loop is covalently continuous with the flanking protein: the junction
    # peptide bonds are real (~1.33 A), NOT the stretched bonds that get capped as a
    # chain break. graft_flanks=1 takes the remodelled 964/977 flanks from the model.
    prot = spliced.atomselect("protein")

    def cn(r1, r2):
        c = spliced.coords[(spliced.resid == r1) & (spliced.name == "C") & prot, :, 0]
        n = spliced.coords[(spliced.resid == r2) & (spliced.name == "N") & prot, :, 0]
        return float(np.linalg.norm(c[0] - n[0]))

    assert cn(964, 965) < 1.6   # N-junction of the loop
    assert cn(976, 977) < 1.6   # C-junction of the loop

    # erlotinib and every crystallographic water survive untouched, unmarked
    assert int((spliced.resname == "AQ4").sum()) == 29
    assert int(spliced.atomselect("water").sum()) == 20
    assert not new_mask[spliced.resname == "AQ4"].any()
    assert not new_mask[spliced.atomselect("water")].any()

    # newly inserted residues carry the protein's own chain and segid
    assert set(spliced.chain[new_mask]) == set(spliced.chain[prot & ~new_mask])
    assert set(spliced.segid[new_mask]) == set(spliced.segid[prot & ~new_mask])

    # a correctly modelled loop must not clash with the bound drug (water clashes,
    # if any, would be reported for the caller to drop those waters)
    clashes = detectSplicedClashes(spliced, new_mask)
    assert not any(c["target_resname"] == "AQ4" for c in clashes)


def _apply_rigid(mol, R, t):
    mol = mol.copy()
    mol.coords[:, :, 0] = mol.coords[:, :, 0] @ R.T + t
    return mol


def test_splice_superposes_donor_from_different_frame():
    # Placing the donor in a foreign coordinate frame must not move the spliced
    # loop: local superposition brings it back into the original's frame, so the
    # junction peptide bonds stay physical.
    gapped = Molecule(os.path.join(curr_dir, "test_modelling", "1m17_gapped.pdb.gz"))
    model = Molecule(os.path.join(curr_dir, "test_modelling", "1m17_model.pdb.gz"))

    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moved = _apply_rigid(model, R, np.array([40.0, -25.0, 10.0]))

    spliced, new_mask = spliceMissingResidues(gapped, moved)
    prot = spliced.atomselect("protein")

    def cn(r1, r2):
        c = spliced.coords[(spliced.resid == r1) & (spliced.name == "C") & prot, :, 0]
        n = spliced.coords[(spliced.resid == r2) & (spliced.name == "N") & prot, :, 0]
        return float(np.linalg.norm(c[0] - n[0]))

    # The junctions that bridge the donor loop to the RETAINED original backbone
    # are grafted-flank <-> first-kept-original: 963(orig)-964(grafted donor) on
    # the N side and 977(grafted donor)-978(orig) on the C side. (964-965 and
    # 976-977 are donor-internal and stay physical under any rigid frame, so they
    # would NOT detect a missing superposition.) These are physical only if the
    # local fit put the donor into the original's frame — without it the donor sits
    # 40+ A away and both bonds blow up.
    assert cn(963, 964) < 1.6
    assert cn(977, 978) < 1.6
    assert int((new_mask & (spliced.name == "CA")).sum()) == 21

    # One-sided terminal-tail fits must bridge the donor frame to the original too:
    # N-tail grafted 672 -> kept-original 673, and kept-original 994 -> C-tail
    # grafted 995. (671-672 and 995-996 are donor-internal and stay physical under
    # any frame, so they would not detect a missing tail superposition.)
    assert cn(672, 673) < 1.6   # N-terminal tail (GSHMAS) junction
    assert cn(994, 995) < 1.6   # C-terminal tail (QQG) junction


def test_1m17_splice_auto_pairs_without_chain_map(tmp_path):
    # No chain_map: the donor chain is paired to original chain A purely by
    # sequence, so the "0"-in-segid vs "A"-in-chain mismatch is irrelevant.
    gapped = Molecule(os.path.join(curr_dir, "test_modelling", "1m17_gapped.pdb.gz"))
    model = Molecule(os.path.join(curr_dir, "test_modelling", "1m17_model.pdb.gz"))
    spliced, new_mask = spliceMissingResidues(gapped, model)   # chain_map defaults to None
    new_ca = new_mask & (spliced.name == "CA")
    assert int(new_ca.sum()) == 21                       # GSHMAS(6)+loop(12)+QQG(3)
    assert int((spliced.resname == "AQ4").sum()) == 29
    assert int(spliced.atomselect("water").sum()) == 20


def test_splice_skips_unmatched_chain_but_fills_sibling(caplog):
    # Two original chains: A that the donor matches (its gap is filled) and B that
    # the donor cannot match (left untouched + reported). Proves an unmatched chain
    # is skipped and warned WHILE a sibling chain is still filled.
    chainA = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5], chain="A", segid="PA")
    chainB = _chain_mol(["TRP", "TRP", "TRP"], [1, 2, 3], chain="B", segid="PB")
    gapped = chainA.copy()
    gapped.append(chainB)
    donor = _chain_mol(["ALA", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5],
                       chain="X", segid="9")

    with caplog.at_level("WARNING"):
        spliced, new_mask = spliceMissingResidues(gapped, donor, graft_flanks=0)

    # chain A filled (HIS, ILE inserted); chain B carried over untouched
    a_ca = (spliced.chain == "A") & (spliced.name == "CA")
    assert list(spliced.resname[a_ca]) == ["ALA", "GLY", "HIS", "ILE", "SER"]
    assert set(spliced.resname[new_mask]) == {"HIS", "ILE"}
    b_ca = (spliced.chain == "B") & (spliced.name == "CA")
    assert list(spliced.resname[b_ca]) == ["TRP", "TRP", "TRP"]
    assert not new_mask[spliced.chain == "B"].any()
    # the unmatched chain is reported (never silently dropped)
    assert any("No donor chain matched" in r.message and "'B'" in r.message
               for r in caplog.records)


def test_splice_donor_monomer_fills_original_dimer():
    # A single-chain donor fills the same gap in BOTH chains of a homodimer: a
    # donor chain may legitimately pair with multiple original chains.
    chainA = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5], chain="A", segid="PA")
    chainB = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5], chain="B", segid="PB")
    gapped = chainA.copy()
    gapped.append(chainB)
    donor = _chain_mol(["ALA", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5],
                       chain="X", segid="9")

    spliced, new_mask = spliceMissingResidues(gapped, donor, graft_flanks=0)

    for ch in ("A", "B"):
        ca = (spliced.chain == ch) & (spliced.name == "CA")
        assert list(spliced.resname[ca]) == ["ALA", "GLY", "HIS", "ILE", "SER"], ch
    # HIS + ILE inserted in each of the two chains -> 4 new CA total
    assert int((new_mask & (spliced.name == "CA")).sum()) == 4


# A donor spanning resids 1-11 with eleven distinct residue types, so the
# alignment has only one sensible pairing. Targets are built by DELETING residues
# from this donor rather than by building a second molecule: _bonded_chain_mol
# threads residues along a curve starting at the first resid in its list, so an
# independently built target that starts at a different resid would sit at a
# different point on the curve and the co-observed residues would not superpose.
# Filtering keeps every kept residue at exactly the donor's coordinates.
_TAIL_FULL = ["MET", "LYS", "THR", "TRP", "ASP", "GLU",
              "TYR", "ARG", "LEU", "ASN", "GLY"]


def _tail_donor():
    return _bonded_chain_mol(_TAIL_FULL, list(range(1, 12)), chain="A")


def _drop_resids(mol, resids):
    """The donor minus `resids`, i.e. a target missing exactly those residues."""
    m = mol.copy()
    m.filter(~np.isin(m.resid, list(resids)), _logger=False)
    m.guessBonds()
    return m


def _ca_resids(mol, mask=None):
    ca = mol.name == "CA"
    if mask is not None:
        ca = ca & mask
    return [int(r) for r in mol.resid[ca]]


def test_splice_extends_both_termini_when_no_gaps_given():
    # Baseline: with gaps=None a donor observed over a wider range than the target
    # extends it past BOTH termini, on top of filling the internal hole. This is
    # the 1BL6-into-1KV2 shape (donor 4-354 into target 5-352 yields 4-354).
    donor = _tail_donor()
    target = _drop_resids(donor, [1, 2, 6, 11])   # missing N-tail, resid 6, C-tail

    spliced, new_mask = spliceMissingResidues(target, donor, {"A": "A"})

    assert _ca_resids(spliced) == list(range(1, 12))
    assert [str(n) for n in spliced.resname[spliced.name == "CA"]] == _TAIL_FULL
    assert _ca_resids(spliced, new_mask) == [1, 2, 6, 11]


def test_splice_gaps_from_backbone_breaks_fills_interior_only():
    # THE case the parameter exists for: detectBackboneBreaks cannot report a
    # terminal tail (it only compares residues adjacent in file order), so passing
    # its output fills the interior hole and leaves both termini exactly as
    # deposited. Asserting the first and last resid is what distinguishes this from
    # "the filter dropped everything".
    donor = _tail_donor()
    target = _drop_resids(donor, [1, 2, 6, 11])

    breaks = detectBackboneBreaks(target)
    assert [(b["after_resid"], b["before_resid"]) for b in breaks] == [(5, 7)]

    spliced, new_mask = spliceMissingResidues(target, donor, {"A": "A"}, gaps=breaks)

    resids = _ca_resids(spliced)
    assert resids == list(range(3, 11))       # 3-10: interior closed, tails absent
    assert resids[0] == 3 and resids[-1] == 10
    assert _ca_resids(spliced, new_mask) == [6]


def test_splice_does_not_extend_a_target_with_no_backbone_break():
    # The 2CBA-into-1CA2 case stated directly: the target has nothing missing in
    # the middle, so detectBackboneBreaks returns [] and the donor's wider span
    # must not touch it.
    donor = _tail_donor()
    target = _drop_resids(donor, [1, 2, 11])   # tails only, backbone continuous

    breaks = detectBackboneBreaks(target)
    assert breaks == []

    spliced, new_mask = spliceMissingResidues(target, donor, {"A": "A"}, gaps=breaks)

    assert _ca_resids(spliced) == list(range(3, 11))
    assert int(new_mask.sum()) == 0


def test_splice_gaps_selects_one_of_two_interior_holes():
    donor = _tail_donor()
    target = _drop_resids(donor, [4, 5, 8])    # two interior holes, no tails missing

    breaks = detectBackboneBreaks(target)
    assert [(b["after_resid"], b["before_resid"]) for b in breaks] == [(3, 6), (7, 9)]

    chosen = [b for b in breaks if b["after_resid"] == 3]
    spliced, new_mask = spliceMissingResidues(target, donor, {"A": "A"}, gaps=chosen)

    # the first hole is closed, the second is still open
    assert _ca_resids(spliced) == [1, 2, 3, 4, 5, 6, 7, 9, 10, 11]
    assert _ca_resids(spliced, new_mask) == [4, 5]
    assert [(b["after_resid"], b["before_resid"])
            for b in detectBackboneBreaks(spliced)] == [(7, 9)]


def test_splice_empty_gaps_inserts_nothing():
    # An empty list is a real request ("fill none of them"), not a missing
    # argument: the same target fills three runs under gaps=None.
    donor = _tail_donor()
    target = _drop_resids(donor, [1, 2, 6, 11])

    spliced, new_mask = spliceMissingResidues(target, donor, {"A": "A"}, gaps=[])

    assert _ca_resids(spliced) == [3, 4, 5, 7, 8, 9, 10]
    assert int(new_mask.sum()) == 0


def test_splice_gaps_selects_one_terminus_independently():
    # The two termini are separately addressable: asking for the N-terminal run by
    # name fills it and leaves the C-terminal one alone.
    donor = _tail_donor()
    target = _drop_resids(donor, [1, 2, 11])

    gaps = [{"chain": "A", "after_resid": None, "before_resid": 3}]
    spliced, new_mask = spliceMissingResidues(target, donor, {"A": "A"}, gaps=gaps)

    assert _ca_resids(spliced) == list(range(1, 11))   # 1-10: N-tail back, no 11
    assert _ca_resids(spliced, new_mask) == [1, 2]


def test_splice_warns_for_a_requested_gap_it_cannot_fill(caplog):
    # Resids 5 and 6 are both present and covalently continuous, so there is no run
    # to insert between them. Asking for one must not pass silently: a mistyped
    # resid would otherwise look like a successful splice that filled nothing.
    donor = _tail_donor()
    target = _drop_resids(donor, [1, 2, 11])

    gaps = [{"chain": "A", "after_resid": 5, "before_resid": 6}]
    with caplog.at_level("WARNING"):
        spliced, new_mask = spliceMissingResidues(target, donor, {"A": "A"}, gaps=gaps)

    assert int(new_mask.sum()) == 0
    assert _ca_resids(spliced) == list(range(3, 11))
    assert any("was not filled" in r.message and "'A'" in r.message
               and "5" in r.message for r in caplog.records)


def test_splice_does_not_warn_when_every_requested_gap_is_filled(caplog):
    # The warning must not fire for the runs that DID fill, otherwise it is noise
    # on every correct call.
    donor = _tail_donor()
    target = _drop_resids(donor, [1, 2, 6, 11])

    with caplog.at_level("WARNING"):
        spliceMissingResidues(target, donor, {"A": "A"},
                              gaps=detectBackboneBreaks(target))

    assert not any("was not filled" in r.message for r in caplog.records)


def test_splice_gaps_filter_is_scoped_to_the_named_chain():
    # A homodimer whose two chains carry the identical (after_resid, before_resid)
    # break, requesting the gap in chain A only. The run key is (chain, after_resid,
    # before_resid); without the `chain` component a regression would still pass
    # every other test here, since none of them uses more than one chain with
    # `gaps=`, and would fill chain B too.
    donor = _tail_donor()
    chainA = _drop_resids(donor, [6])
    chainA.chain[:] = "A"
    chainB = _drop_resids(donor, [6])
    chainB.chain[:] = "B"
    chainB.segid[:] = "Q"
    target = chainA.copy()
    target.append(chainB)

    gaps = [{"chain": "A", "after_resid": 5, "before_resid": 7}]
    spliced, new_mask = spliceMissingResidues(target, donor, gaps=gaps)

    assert _ca_resids(spliced, spliced.chain == "A") == list(range(1, 12))
    assert _ca_resids(spliced, spliced.chain == "B") == [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
    assert int(new_mask[spliced.chain == "A"].sum()) > 0
    assert int(new_mask[spliced.chain == "B"].sum()) == 0


def test_splice_gaps_selects_c_terminal_tail_independently():
    # Only the N-terminal direction is exercised by
    # test_splice_gaps_selects_one_terminus_independently; a regression that mixed
    # up which side a None marks (e.g. treating before_resid=None as N-terminal)
    # would still pass that test while getting this one backwards.
    donor = _tail_donor()
    target = _drop_resids(donor, [1, 2, 11])

    gaps = [{"chain": "A", "after_resid": 10, "before_resid": None}]
    spliced, new_mask = spliceMissingResidues(target, donor, {"A": "A"}, gaps=gaps)

    assert _ca_resids(spliced) == list(range(3, 12))   # 3-11: C-tail back, no 1/2
    assert _ca_resids(spliced, new_mask) == [11]


def test_splice_warns_for_unmatched_terminal_request_without_saying_none(caplog):
    # FIX 4's path: an N-terminal request against a target whose N-terminus is
    # already intact has no run to match. The message must describe the request in
    # words (chain's N-terminus), not print `None` for the missing after_resid.
    donor = _tail_donor()
    target = _drop_resids(donor, [6, 11])   # N-terminus intact, nothing to request

    gaps = [{"chain": "A", "after_resid": None, "before_resid": 1}]
    with caplog.at_level("WARNING"):
        spliced, new_mask = spliceMissingResidues(target, donor, {"A": "A"}, gaps=gaps)

    assert int(new_mask.sum()) == 0
    warnings = [r.message for r in caplog.records if "was not filled" in r.message]
    assert len(warnings) == 1
    assert "None" not in warnings[0]
    assert "N-terminus" in warnings[0] and "resid 1" in warnings[0]


def test_splice_gaps_from_detect_sequence_gaps_fills_interior_only():
    # The docstring explicitly promises detectSequenceGaps' output works directly
    # as `gaps=`; nothing else here exercises that path (only detectBackboneBreaks
    # is used elsewhere). Filtering to the non-terminal entries and passing the
    # rest must fill only the interior hole, leaving both termini as deposited.
    donor = _tail_donor()
    full_seq = donor.getSequence()["A"]
    target = _drop_resids(donor, [1, 2, 6, 11])

    gaps, skipped, _ = detectSequenceGaps(target, {"A": full_seq})
    assert skipped == []
    interior = [g for g in gaps if not g["is_terminal"]]

    spliced, new_mask = spliceMissingResidues(target, donor, {"A": "A"}, gaps=interior)

    assert _ca_resids(spliced) == list(range(3, 11))   # 3-10: interior closed, tails absent
    assert _ca_resids(spliced, new_mask) == [6]


def _worst_bonded_junction(sp):
    """Max backbone C(i)->N(i+1) distance over consecutive residues that are
    actually bonded (contiguous numbering). A numbering jump marks an unfilled
    (skipped) gap - not a peptide bond - so it is excluded."""
    prot = sp.atomselect("protein")
    worst = 0.0
    for ch in np.unique(sp.chain[prot]):
        csel = prot & (sp.chain == ch)
        order, seen = [], set()
        for a in np.where(csel)[0]:
            k = (int(sp.resid[a]), str(sp.insertion[a]))
            if k not in seen:
                seen.add(k)
                order.append(k)

        def bb(r, ins, name):
            s = csel & (sp.resid == r) & (sp.insertion == ins) & (sp.name == name)
            return sp.coords[s, :, 0][0] if s.sum() else None

        for (r1, i1), (r2, i2) in zip(order, order[1:]):
            if r2 - r1 > 1:
                continue
            c, natom = bb(r1, i1, "C"), bb(r2, i2, "N")
            if c is not None and natom is not None:
                worst = max(worst, float(np.linalg.norm(c - natom)))
    return worst


def test_splice_5vq2_from_4dso_crosscrystal():
    # Real cross-crystal splice: KRAS. 5VQ2 (a homodimer, chains A+B, GTP/Mg-bound,
    # three unresolved regions per chain) is filled from 4DSO - the same protein in a
    # more complete crystal (one chain). Both 5VQ2 chains pair to 4DSO by sequence
    # (~0.97 id). Structures load from tests/pdb/{5vq2,4dso}.bcif.gz via
    # LOCAL_PDB_REPO (conftest / CI).
    #
    # This exercises the graft_flanks workflow. At graft_flanks=1 the two loops whose
    # flanks already agree between the crystals splice cleanly, but the flexible
    # switch-II / C-terminal region diverges too far for a 1-residue graft, so it is
    # SKIPPED with a warning. Re-running on the RESULT with graft_flanks=2 grafts
    # deeper and fills it. We assert only backbone JUNCTION distances - sidechain
    # clashes introduced by the deeper graft are expected and left for downstream
    # relaxation.
    import logging

    def _has(sp, ch, resid):
        return bool(((sp.chain == ch) & (sp.resid == resid) & (sp.name == "CA")).any())

    def _capture():
        msgs = []

        class _C(logging.Handler):
            def emit(self, record):
                msgs.append(record.getMessage())

        h = _C()
        h.setLevel(logging.WARNING)
        logging.getLogger("moleculekit.tools.modelling").addHandler(h)
        return msgs, h

    m5 = Molecule("5VQ2")
    m4 = Molecule("4DSO")

    # --- Part 1: graft_flanks=1 (minimal graft) ---
    msgs, h = _capture()
    try:
        sp1, nm1 = spliceMissingResidues(m5, m4, graft_flanks=1)
    finally:
        logging.getLogger("moleculekit.tools.modelling").removeHandler(h)

    # Two loops per chain splice cleanly (34-36 and switch-II 59-68 = 13 residues);
    # the divergent C-terminal region (after resid 168) is skipped and warned.
    assert {ch: int((nm1 & (sp1.chain == ch) & (sp1.name == "CA")).sum())
            for ch in ("A", "B")} == {"A": 13, "B": 13}
    assert _has(sp1, "A", 35) and _has(sp1, "A", 64)     # both clean loops filled
    assert not _has(sp1, "A", 174)                       # C-terminal region skipped
    for ch in ("A", "B"):
        assert any(f"chain {ch} after resid 168" in m for m in msgs)
    # every bonded junction among the spliced loops is physical
    assert _worst_bonded_junction(sp1) < 1.6

    # --- Part 2: re-run on the result with a larger graft_flanks to fill the rest ---
    msgs2, h2 = _capture()
    try:
        sp2, nm2 = spliceMissingResidues(sp1, m4, graft_flanks=2)
    finally:
        logging.getLogger("moleculekit.tools.modelling").removeHandler(h2)

    # The deeper graft closes the previously-skipped region: it fills now, with no
    # skip warnings, and every chain is full length (181 residues) and continuous.
    assert _has(sp2, "A", 174) and _has(sp2, "B", 174)
    assert not any("Skipping" in m for m in msgs2)
    assert _worst_bonded_junction(sp2) < 1.6
    for ch in ("A", "B"):
        n = len({int(r) for r in sp2.resid[sp2.atomselect("protein") & (sp2.chain == ch)]})
        assert n == 181, (ch, n)

    # Report the fit quality: co-observed CA RMSD, donor -> original chain A.
    s5, i5 = m5.getSequence(dict_key="chain", return_idx=True, sel="protein", _logger=False)
    s4, i4 = m4.getSequence(dict_key="chain", return_idx=True, sel="protein", _logger=False)

    def _ca(mol, atomlist):
        a = atomlist[mol.name[atomlist] == "CA"]
        return mol.coords[a[0], :, 0] if len(a) else None

    aln4, aln5 = _align_full_to_observed(s4["A"], s5["A"])
    p4 = p5 = 0
    donor_ca, orig_ca = [], []
    for c4, c5 in zip(aln4, aln5):
        if c4 != "-" and c5 != "-":
            d, o = _ca(m4, i4["A"][p4]), _ca(m5, i5["A"][p5])
            if d is not None and o is not None:
                donor_ca.append(d)
                orig_ca.append(o)
        if c4 != "-":
            p4 += 1
        if c5 != "-":
            p5 += 1
    _, rmsd = _fit_donor_to_orig(np.array(donor_ca), np.array(orig_ca))
    print(
        f"\n5VQ2<-4DSO KRAS: gf=1 filled 13/chain (2 loops), C-terminal region "
        f"skipped+warned; re-run gf=2 filled the rest (junctions "
        f"<{_worst_bonded_junction(sp2):.2f} A). co-observed CA fit RMSD {rmsd:.2f} A."
    )

    # Cofactors preserved throughout: both GTP molecules (2 x 32 atoms) + both Mg.
    assert int((sp2.resname == "GTP").sum()) == 64
    assert int((sp2.resname == "MG").sum()) == 2


def test_detect_sequence_gaps_reports_mismatches():
    # Bonded backbone, equal lengths: the observed GLY 2 aligns to a THR in the
    # reference, which is a point mutation, not a missing residue.
    m = _bonded_chain_mol(["ALA", "GLY", "SER"], [1, 2, 3])
    gaps, skipped, mismatches = detectSequenceGaps(m, {"A": "ATS"})
    assert gaps == []
    assert skipped == []
    assert mismatches == [
        {
            "chain": "A",
            "resid": 2,
            "insertion": "",
            "reference": "T",
            "observed": "G",
            "ref_index": 1,
        }
    ]


def test_detect_sequence_gaps_mismatch_ref_index_spans_gap():
    # A mismatch downstream of a gap must carry the reference index INCLUDING the
    # missing residues, so patching sequences[chain][ref_index] hits the residue
    # the mismatch is actually about.
    m = _bonded_chain_mol(["ALA", "GLY", "SER"], [1, 2, 6])
    gaps, _, mismatches = detectSequenceGaps(m, {"A": "AGHIKT"})
    assert len(gaps) == 1 and gaps[0]["missing_seq"] == "HIK"
    assert [
        (x["resid"], x["reference"], x["observed"], x["ref_index"]) for x in mismatches
    ] == [(6, "T", "S", 5)]
    # the index is usable as-is to graft the observed residue onto the reference
    ref = list("AGHIKT")
    ref[mismatches[0]["ref_index"]] = mismatches[0]["observed"]
    assert "".join(ref) == "AGHIKS"


def test_detect_sequence_gaps_mismatch_carries_insertion_code():
    m = _bonded_chain_mol(["ALA", "GLY", "SER"], [1, 2, 3])
    m.insertion[m.resid == 2] = "A"
    _, _, mismatches = detectSequenceGaps(m, {"A": "ATS"})
    assert [(x["resid"], x["insertion"]) for x in mismatches] == [(2, "A")]


def test_detect_sequence_gaps_ignores_unknown_residue_mismatches():
    # "X" on either side is an unknown / modified residue, not a mutation.
    m = _bonded_chain_mol(["ALA", "GLY", "SER"], [1, 2, 3])
    _, _, mismatches = detectSequenceGaps(m, {"A": "AXS"})
    assert mismatches == []


def test_detect_sequence_gaps_no_mismatches_when_identical():
    m = _bonded_chain_mol(["ALA", "GLY", "SER"], [1, 2, 6])
    gaps, _, mismatches = detectSequenceGaps(m, {"A": "AGHIKS"})
    assert len(gaps) == 1
    assert mismatches == []


def test_detect_backbone_breaks_finds_the_break():
    m = _bonded_chain_mol(["ALA", "GLY", "SER", "THR"], [1, 2, 6, 7])
    breaks = detectBackboneBreaks(m)
    assert len(breaks) == 1
    b = breaks[0]
    assert (b["segid"], b["chain"], b["after_resid"], b["before_resid"]) == (
        "P",
        "A",
        2,
        6,
    )
    assert b["distance"] > 2.0


def test_detect_backbone_breaks_none_on_continuous_chain():
    m = _bonded_chain_mol(["ALA", "GLY", "SER"], [1, 2, 3])
    assert detectBackboneBreaks(m) == []


def test_detect_backbone_breaks_ignores_chain_and_segment_boundaries():
    # Two chains sharing one (empty) segid: the junction between them is not a
    # break, however far apart they sit.
    a = _bonded_chain_mol(["ALA", "GLY"], [1, 2], chain="A", segid="")
    b = _bonded_chain_mol(["SER", "THR"], [1, 2], chain="B", segid="")
    m = a.copy()
    m.append(b)
    assert detectBackboneBreaks(m) == []
    # Same for one chain id split across two segids.
    c = _bonded_chain_mol(["ALA", "GLY"], [1, 2], chain="A", segid="P1")
    d = _bonded_chain_mol(["SER", "THR"], [3, 4], chain="A", segid="P2")
    m2 = c.copy()
    m2.append(d)
    assert detectBackboneBreaks(m2) == []


def test_detect_backbone_breaks_tolerance_is_configurable():
    m = _bonded_chain_mol(["ALA", "GLY", "SER"], [1, 2, 3])
    # a peptide C-N is ~1.33 A, so a tolerance below it turns every junction into
    # a break - proves the threshold is what decides, not the bonds
    assert len(detectBackboneBreaks(m, tol=1.0)) == 2


def test_detect_backbone_breaks_1m17_real_case():
    gapped = Molecule(os.path.join(curr_dir, "test_modelling", "1m17_gapped.pdb.gz"))
    breaks = detectBackboneBreaks(gapped)
    # the single internal gap of the deposited structure, and nothing else
    assert [(b["after_resid"], b["before_resid"]) for b in breaks] == [(964, 977)]


def test_detect_backbone_breaks_reports_missing_backbone_atom():
    # A residue with no N at all: continuity cannot be established from
    # coordinates, so it is a break with distance None rather than a silent pass.
    m = _bonded_chain_mol(["ALA", "GLY", "SER"], [1, 2, 3])
    m.remove("resid 2 and name N", _logger=False)
    breaks = detectBackboneBreaks(m)
    assert [(b["after_resid"], b["before_resid"], b["distance"]) for b in breaks] == [
        (1, 2, None)
    ]


# --- a partly filled gap must not read as a filled one ------------------------
# A run is keyed by the resids flanking it, so a donor carrying one residue of a
# four-residue gap matches exactly as one carrying all four. Without the count,
# partial coverage is indistinguishable from complete coverage.

def _slots(pattern, start=1):
    """`pattern` is a string of 'o' (original) and 'n' (new) residues."""
    slots, resid = [], start
    for ch in pattern:
        if ch == "o":
            slots.append({"new": False, "resid": resid, "insertion": ""})
            resid += 1
        else:
            slots.append({"new": True})
    return slots


def test_a_matched_run_reports_how_many_residues_it_supplied():
    from moleculekit.tools.modelling import _select_requested_runs

    # one new residue between deposited 1 and 2
    _, matched = _select_requested_runs(_slots("ono"), "A", {("A", 1, 2)})
    assert matched == {("A", 1, 2): 1}

    # three new residues in the same slot
    _, matched = _select_requested_runs(_slots("onnno"), "A", {("A", 1, 2)})
    assert matched == {("A", 1, 2): 3}


def test_an_unrequested_run_is_still_dropped():
    from moleculekit.tools.modelling import _select_requested_runs

    slots, matched = _select_requested_runs(_slots("onno"), "A", {("A", 9, 9)})
    assert matched == {}
    assert all(not s["new"] for s in slots)


def test_a_terminal_run_is_keyed_with_none_on_the_open_side():
    from moleculekit.tools.modelling import _select_requested_runs

    _, matched = _select_requested_runs(_slots("nno"), "A", {("A", None, 1)})
    assert matched == {("A", None, 1): 2}


def test_splice_warns_when_a_donor_fills_only_part_of_a_gap(caplog):
    """The donor carries residue 5 but not 4, so the run between deposited 3 and 6
    is one residue where the caller asked for two. Matching is keyed on those two
    flanking resids, so without this warning the gap reads as filled.
    """
    import logging

    full = _bonded_chain_mol(_MERGE_FULL_RESNAMES, list(range(1, 12)), chain="A")
    # donor missing residue 4: it can supply 5, and nothing for 4
    keep = [r for r in range(1, 12) if r != 4]
    donor = _bonded_chain_mol(
        [_MERGE_FULL_RESNAMES[r - 1] for r in keep], keep, chain="A"
    )
    gapped = _bonded_chain_mol(_MERGE_OBS_RESNAMES, _MERGE_OBS_RESIDS, chain="A")

    gaps = [{"chain": "A", "after_resid": 3, "before_resid": 6, "missing_seq": "PP"}]
    with caplog.at_level(logging.WARNING, logger="moleculekit.tools.modelling"):
        spliceMissingResidues(gapped, donor, {"A": "A"}, gaps=gaps)

    partly = [r for r in caplog.messages if "only partly filled" in r]
    assert partly, caplog.messages
    assert "supplied 1 of 2" in partly[0]
    assert "between resid 3 and 6" in partly[0]
    assert full is not None  # the fully-covered donor is the contrast, below


def test_splice_stays_quiet_when_the_donor_covers_the_whole_gap(caplog):
    import logging

    donor = _bonded_chain_mol(_MERGE_FULL_RESNAMES, list(range(1, 12)), chain="A")
    gapped = _bonded_chain_mol(_MERGE_OBS_RESNAMES, _MERGE_OBS_RESIDS, chain="A")

    gaps = [{"chain": "A", "after_resid": 3, "before_resid": 6, "missing_seq": "PP"}]
    with caplog.at_level(logging.WARNING, logger="moleculekit.tools.modelling"):
        spliceMissingResidues(gapped, donor, {"A": "A"}, gaps=gaps)

    assert not [r for r in caplog.messages if "partly filled" in r], caplog.messages


# --- a residue modelled short of its backbone is present, not missing ---------
# `atomselect("protein")` asks for all four of N, CA, C and O, so a residue with
# any one of them absent is not protein to it and drops out of the sequence
# derived from it. The alignment then reports a residue missing that is sitting
# in the file bonded to both neighbours, and the termini and caps that follow
# name an end in the middle of a chain, which no build can apply.


def test_a_residue_missing_its_carbonyl_is_not_a_gap():
    m = _bonded_chain_mol(["ALA", "GLY", "LEU", "SER", "VAL"], [1, 2, 3, 4, 5])
    # a real deposition: one atom short
    m.remove("resid 3 and name O", _logger=False)

    gaps, skipped, mismatches = detectSequenceGaps(m, {"A": "AGLSV"})

    assert gaps == [], gaps
    assert skipped == []
    assert mismatches == []


def test_that_residue_still_reaches_the_sequence_with_its_own_letter():
    from moleculekit.tools.modelling import _observed_sequence

    m = _bonded_chain_mol(["ALA", "GLY", "LEU", "SER", "VAL"], [1, 2, 3, 4, 5])
    m.remove("resid 3 and name O", _logger=False)

    seqs, idxs = _observed_sequence(m)

    assert seqs["A"] == "AGLSV"
    assert len(idxs["A"]) == 5


def test_a_real_gap_is_still_reported():
    """The fix must not swallow the case it has to keep finding."""
    m = _bonded_chain_mol(["ALA", "GLY", "SER", "VAL"], [1, 2, 5, 6])

    gaps, _, _ = detectSequenceGaps(m, {"A": "AGLLSV"})

    assert len(gaps) == 1, gaps
    assert (gaps[0]["after_resid"], gaps[0]["before_resid"]) == (2, 5)
    assert gaps[0]["missing_seq"] == "LL"


def test_an_unbonded_amino_acid_is_not_part_of_the_chain():
    """A free glycine in the solvent is a component, not a residue of the chain.

    It carries a polymer resname, so identity alone would take it; the peptide
    bond is what decides.
    """
    from moleculekit.tools.modelling import _observed_sequence

    m = _bonded_chain_mol(["ALA", "GLY", "LEU"], [1, 2, 3])
    free = _bonded_chain_mol(["GLY"], [99], chain="A", segid="P")
    free.moveBy([60.0, 60.0, 60.0])
    # incomplete, so the selection rejects it too
    free.remove("name O", _logger=False)
    m.append(free)

    seqs, _ = _observed_sequence(m)

    assert seqs["A"] == "AGL", seqs


def test_the_classifier_takes_the_short_residue_and_skips_the_free_one():
    """The same rule as a public function, keyed per residue."""
    from moleculekit.tools.backbone import residuePolymerStatus

    m = _bonded_chain_mol(["ALA", "GLY", "LEU"], [1, 2, 3])
    m.remove("resid 2 and name O", _logger=False)
    free = _bonded_chain_mol(["GLY"], [99], chain="A", segid="P")
    free.moveBy([60.0, 60.0, 60.0])
    free.remove("name O", _logger=False)
    m.append(free)

    resids = [
        key[2] for status, key, _atoms in residuePolymerStatus(m)
        if status == "protein"
    ]

    assert resids == [1, 2, 3]


def _polymer_keys(mol, polymer):
    """Keys of the residues `polymer` claims, via the shared classifier."""
    from moleculekit.tools.backbone import residuePolymerStatus

    wanted = ("protein", "nucleic") if polymer == "both" else (polymer,)
    return [
        key for status, key, _atoms in residuePolymerStatus(mol) if status in wanted
    ]


def test_the_classifier_walks_a_nucleic_chain():
    from moleculekit.tools.backbone import residuePolymerStatus

    m = Molecule(os.path.join(curr_dir, "pdb", "1bna.pdb"))

    assert len(_polymer_keys(m, "nucleic")) == 24
    assert _polymer_keys(m, "protein") == []

    # The yielded indices index the molecule, not positions within a selection:
    # read relatively these would land on whatever sits at the front of the file.
    for status, key, atoms in residuePolymerStatus(m):
        assert str(m.chain[atoms[0]]) == key[1]
        assert int(m.resid[atoms[0]]) == key[2]
        if status == "nucleic":
            assert str(m.resname[atoms[0]]) in ("DA", "DC", "DG", "DT")


def test_a_nucleotide_short_of_its_backbone_stays_in_the_chain():
    """The nucleic half of the rule, on the selection's own threshold.

    ``atomselect("nucleic")`` wants four connected backbone atoms out of a list of
    about ten, so a nucleotide has to lose most of its backbone to fall out. Strip
    one that far and the phosphodiester to its neighbour is what puts it back.
    """
    m = Molecule(os.path.join(curr_dir, "pdb", "1bna.pdb"))
    m.remove("chain A and resid 5 and name P OP1 OP2 O5' C5'", _logger=False)
    stripped = (m.chain == "A") & (m.resid == 5)

    assert not m.atomselect("nucleic")[stripped].any()

    keys = _polymer_keys(m, "nucleic")
    assert ("A", 5) in [(key[1], key[2]) for key in keys]
    assert len(keys) == 24


def test_both_walks_a_protein_rna_complex():
    path = os.path.join(
        curr_dir, "test_systemprepare", "test-rna-protein-complex", "3WBM.pdb"
    )
    m = Molecule(path)

    protein = set(_polymer_keys(m, "protein"))
    nucleic = set(_polymer_keys(m, "nucleic"))
    both = set(_polymer_keys(m, "both"))

    assert protein and nucleic
    assert not (protein & nucleic)
    assert both == protein | nucleic

    # and the same at mask level, which is how a caller combines polymers
    from moleculekit.tools.backbone import chainResidueMask

    assert np.array_equal(
        chainResidueMask(m, "both"),
        chainResidueMask(m, "protein") | chainResidueMask(m, "nucleic"),
    )


def test_an_unknown_polymer_is_refused():
    from moleculekit.tools.backbone import chainResidueMask

    m = _bonded_chain_mol(["ALA", "GLY"], [1, 2])

    with pytest.raises(ValueError):
        chainResidueMask(m, "lipid")


def test_a_terminus_the_build_will_remove_is_not_reported():
    """A half-modelled chain end is not a terminus the build has.

    The rule admits a residue carrying only its backbone N, which is right for a
    gap: the tail starts after it. `check_backbone` then removes it, one heavy
    atom against a threshold of four, so proposing a cap there asks for a cap on a
    residue that will not exist. An internal gap edge is a different matter and
    stays, being no terminal to `check_backbone`.
    """
    from moleculekit.tools.backbone import check_backbone, removable_broken_terminal

    m = _bonded_chain_mol(["ALA", "GLY", "LEU", "SER", "VAL"], [1, 2, 3, 4, 5])
    m.remove("resid 5 and not name N", _logger=False)
    end = m.getResidues(return_idx=True)[1][-1]

    assert removable_broken_terminal(m, end, "C")
    # and the build agrees: the residue is gone after check_backbone
    check_backbone(m)
    assert 5 not in [int(x) for x in m.resid]


def test_a_whole_terminus_is_kept():
    from moleculekit.tools.backbone import removable_broken_terminal

    m = _bonded_chain_mol(["ALA", "GLY", "LEU"], [1, 2, 3])
    groups = m.getResidues(return_idx=True)[1]

    assert not removable_broken_terminal(m, groups[-1], "C")
    assert not removable_broken_terminal(m, groups[0], "N")


def test_a_cap_is_never_trimmed_as_a_broken_terminus():
    """The resname gate is load-bearing, not incidental.

    NH2 carries one heavy atom and ACE three, both under the threshold, and both
    are missing backbone atoms -- the arithmetic alone would drop them. They are
    refused because they are not protein resnames, the same gate `check_backbone`
    uses, which is why the two agree. They also never reach the trim: the observed
    sequence is built over protein residues and a cap is not one.
    """
    from moleculekit.tools.backbone import (
        _observed_sequence,
        removable_broken_terminal,
    )
    from moleculekit.residues import CAP_RESIDUE_NAMES

    m = _bonded_chain_mol(["ALA", "GLY", "LEU"], [1, 2, 3])
    cap = _bonded_chain_mol(["ALA"], [4], chain="A", segid="P")
    cap.resname[:] = "NME"
    cap.filter("name N CA", _logger=False)
    m.append(cap)

    groups = m.getResidues(return_idx=True)[1]
    assert str(m.resname[groups[-1][0]]) in CAP_RESIDUE_NAMES
    assert not removable_broken_terminal(m, groups[-1], "C")

    _, idxs = _observed_sequence(m)
    walked = [str(m.resname[g[0]]) for g in idxs.get("A", [])]
    assert "NME" not in walked, walked
