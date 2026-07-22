import os
import numpy as np
import pytest
from moleculekit.molecule import Molecule
from moleculekit.tools.modelling import (
    _align_full_to_observed,
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


def _chain_mol(resnames, resids, chain="A", segid="P"):
    # moleculekit's "protein" atomselect needs a full N-CA-C-O backbone with
    # guessed bonds (a lone CA per residue is not recognized as protein).
    names = ["N", "CA", "C", "O"] * len(resnames)
    m = Molecule().empty(len(names))
    m.name[:] = names
    m.resname[:] = np.repeat(list(resnames), 4)
    m.resid[:] = np.repeat(list(resids), 4)
    m.chain[:] = chain
    m.segid[:] = segid
    m.record[:] = "ATOM"
    m.element[:] = [n[0] for n in names]
    coords = np.zeros((len(names), 3), dtype=np.float32)
    coords[:, 0] = np.arange(len(names)) * 1.45
    m.coords = coords.reshape(len(names), 3, 1)
    m.guessBonds()
    return m


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
    gaps, skipped = detectSequenceGaps(m, full)
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
    gaps, skipped = detectSequenceGaps(m, full)
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
    gaps, skipped = detectSequenceGaps(m, {"A": "AGXAA"})
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
    gaps, skipped = detectSequenceGaps(gapped, {"A": _full_sequence()})
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
