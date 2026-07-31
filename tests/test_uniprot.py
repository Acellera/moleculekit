import os

import pytest
from moleculekit.molecule import Molecule
from moleculekit.uniprot import (
    trimPrecursorSequences,
    uniprotSearch,
    uniprotSequence,
)

curr_dir = os.path.dirname(os.path.abspath(__file__))

# Bovine cationic trypsin: 246 residues of which the first 23 are the signal
# peptide (MKTFIFLALLGAAVAF) plus the activation peptide (PVDDDDK). 3PTB contains
# the mature chain only, so it is the canonical case for precursor trimming.
TRYPSIN_ACC = "P00760"
TRYPSIN_PRECURSOR_LEN = 246
TRYPSIN_OVERHANG = "MKTFIFLALLGAAVAFPVDDDDK"


def test_uniprot_sequence():
    seq = uniprotSequence(TRYPSIN_ACC)
    assert len(seq) == TRYPSIN_PRECURSOR_LEN
    assert seq.startswith(TRYPSIN_OVERHANG)


def test_uniprot_sequence_bad_accession_raises():
    with pytest.raises(RuntimeError):
        uniprotSequence("NOTANACCESSION")


def test_uniprot_search():
    hits = uniprotSearch('trypsin AND organism_name:"Bos taurus"', size=5)
    assert len(hits) > 0
    hit = next(h for h in hits if h["accession"] == TRYPSIN_ACC)
    assert hit["organism"] == "Bos taurus"
    assert hit["length"] == TRYPSIN_PRECURSOR_LEN
    assert hit["reviewed"] is True
    assert "trypsin" in hit["name"].lower() or "protease" in hit["name"].lower()


def test_trim_precursor_sequences_removes_signal_peptide():
    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    full = uniprotSequence(TRYPSIN_ACC)
    trimmed = trimPrecursorSequences(mol, {"A": full})
    assert trimmed["A"] == full[len(TRYPSIN_OVERHANG) :]
    # the mature chain observed in 3PTB is exactly what is left
    assert trimmed["A"].startswith("IVGG")


def test_trim_precursor_sequences_leaves_the_input_untouched():
    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    sequences = {"A": uniprotSequence(TRYPSIN_ACC)}
    before = dict(sequences)
    trimPrecursorSequences(mol, sequences)
    assert sequences == before


def test_trim_precursor_sequences_only_named_chains():
    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    full = uniprotSequence(TRYPSIN_ACC)
    # chain "A" not selected -> returned as-is, overhang intact
    assert trimPrecursorSequences(mol, {"A": full}, chains=[])["A"] == full


def test_trimmed_sequence_leaves_no_terminal_gaps():
    from moleculekit.tools.modelling import detectSequenceGaps

    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    trimmed = trimPrecursorSequences(mol, {"A": uniprotSequence(TRYPSIN_ACC)})
    gaps, skipped, mismatches = detectSequenceGaps(mol, trimmed)
    # 3PTB is a complete mature chain: after trimming there is nothing missing and
    # nothing mutated relative to the reference.
    assert gaps == []
    assert skipped == []
    assert mismatches == []
