import os
import numpy as np
import pytest
from moleculekit.molecule import Molecule
from moleculekit.tools.modelling import (
    _align_full_to_observed,
    detectModelledClashes,
    detectSequenceGaps,
    prepareGapModellingInput,
    spliceModelledResidues,
)

curr_dir = os.path.dirname(os.path.abspath(__file__))


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
    spliced, new_mask = spliceModelledResidues(gapped, predicted, {"A": "A"})

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
    spliced, new_mask = spliceModelledResidues(gapped, predicted, {"A": "A"})

    ca = (spliced.name == "CA") & (spliced.chain == "A")
    assert list(spliced.resname[ca]) == ["ALA", "GLY", "HIS", "ILE", "SER"]
    assert list(spliced.resid[ca]) == [1, 2, 2, 2, 3]        # anchored on resid 2
    assert list(spliced.insertion[ca]) == ["", "", "A", "B", ""]
    assert set(spliced.resname[new_mask]) == {"HIS", "ILE"}


def test_splice_grafts_flanking_residue_from_model():
    # a loop modeller moves the residues flanking a gap to close the loop. graft_flanks=1
    # must adopt the model's flank (keeping the original numbering) so the junction stays
    # continuous; graft_flanks=0 keeps the original coordinates.
    gapped = _chain_mol(["ALA", "GLY", "SER"], [1, 2, 5])
    predicted = _chain_mol(["ALA", "GLY", "HIS", "ILE", "SER"], [1, 2, 3, 4, 5])
    # displace the model's -1 flank (GLY, resid 2) by 10 A so it differs from the crystal
    predicted.coords[predicted.resid == 2, 0, 0] += 10.0

    def flank_x(m, resid):
        return float(m.coords[(m.resid == resid) & (m.name == "CA"), 0, 0][0])

    sp1, new1 = spliceModelledResidues(gapped, predicted, {"A": "A"}, graft_flanks=1)
    # the flank keeps its original number (2), is NOT marked new, but took the model coords
    assert not new1[sp1.resid == 2].any()
    assert abs(flank_x(sp1, 2) - flank_x(predicted, 2)) < 1e-4

    sp0, _ = spliceModelledResidues(gapped, predicted, {"A": "A"}, graft_flanks=0)
    assert abs(flank_x(sp0, 2) - flank_x(gapped, 2)) < 1e-4  # original coords kept


def test_splice_c_terminal_tail_numbers_upward():
    # observed ALA-GLY (1,2); full ALA-GLY-HIS-ILE adds a C-terminal tail.
    predicted = _chain_mol(["ALA", "GLY", "HIS", "ILE"], [1, 2, 3, 4])
    gapped = _chain_mol(["ALA", "GLY"], [1, 2])
    spliced, new_mask = spliceModelledResidues(gapped, predicted, {"A": "A"})

    ca = (spliced.name == "CA") & (spliced.chain == "A")
    assert list(spliced.resname[ca]) == ["ALA", "GLY", "HIS", "ILE"]
    assert list(spliced.resid[ca]) == [1, 2, 3, 4]           # 3,4 count up from 2
    assert set(spliced.resname[new_mask]) == {"HIS", "ILE"}


def test_splice_n_terminal_tail_numbers_up_to_first():
    # observed GLY-SER at 3,4; full ALA-ALA-GLY-SER adds an N-terminal tail.
    predicted = _chain_mol(["ALA", "ALA", "GLY", "SER"], [1, 2, 3, 4])
    gapped = _chain_mol(["GLY", "SER"], [3, 4])
    spliced, new_mask = spliceModelledResidues(gapped, predicted, {"A": "A"})

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

    spliced, new_mask = spliceModelledResidues(gapped, predicted, {"A": "A"})

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
    clashes = detectModelledClashes(m, new_mask, cutoff=2.0)
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
    assert detectModelledClashes(m, np.array([True, False]), cutoff=2.0) == []


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

    spliced, new_mask = spliceModelledResidues(gapped, predicted, {"A": "A"})

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

    spliced, new_mask = spliceModelledResidues(gapped, model, {"A": "A"})

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
    clashes = detectModelledClashes(spliced, new_mask)
    assert not any(c["target_resname"] == "AQ4" for c in clashes)


def test_1m17_splice_accepts_workflow_chain_map(tmp_path):
    # Regression: the splice must accept the chain_map that
    # prepareGapModellingInput actually produces. Its keys are FASTA record
    # indices ("0","1",...), not chain letters, so chain_map is {"0": "A"} here.
    # aceboltz's model.pdb carries that index in the SEGID column (its minimize()
    # step round-trips through OpenMM's PDB writer, which relabels the chainID
    # column by index -> "A" and preserves the real chain id in segid). Resolving
    # the predicted residues by chain alone therefore looked up chain "0", found
    # nothing, and raised "no protein sequence"; the predicted chain must be
    # matched by segid (falling back to chain).
    gapped = Molecule(os.path.join(curr_dir, "test_modelling", "1m17_gapped.pdb.gz"))
    model = Molecule(os.path.join(curr_dir, "test_modelling", "1m17_model.pdb.gz"))
    sequences = {"A": _full_sequence()}

    gaps, _ = detectSequenceGaps(gapped, sequences)
    _, _, chain_map = prepareGapModellingInput(gapped, sequences, gaps, str(tmp_path))
    # the workflow keys the predicted chain by its FASTA index, and the model
    # file carries that same index in segid (with chain relabeled to "A").
    assert chain_map == {"0": "A"}
    assert set(model.segid) == {"0"}
    assert set(model.chain) == {"A"}

    # must not raise, and must fill all three gaps: GSHMAS (6) + the 965-976 loop
    # LPSPTDSNFYRA (12) + QQG (3) = 21 residues.
    spliced, new_mask = spliceModelledResidues(gapped, model, chain_map)
    new_ca = new_mask & (spliced.name == "CA")
    assert int(new_ca.sum()) == 21
    # ligand and crystallographic waters survive the splice untouched
    assert int((spliced.resname == "AQ4").sum()) == 29
    assert int(spliced.atomselect("water").sum()) == 20
