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


def _payload(features, length):
    return {"sequence": {"length": length, "value": "A" * length}, "features": features}


def test_mature_chains_from_chain_features(monkeypatch):
    # P00760: signal 1-17, activation peptide 18-23, mature chain 24-246,
    # plus the two alpha-trypsin autolysis chains.
    from moleculekit import uniprot as up

    feats = [
        {"type": "Signal", "location": {"start": {"value": 1}, "end": {"value": 17}}},
        {"type": "Propeptide", "description": "Activation peptide",
         "location": {"start": {"value": 18}, "end": {"value": 23}}},
        {"type": "Chain", "description": "Serine protease 1",
         "location": {"start": {"value": 24}, "end": {"value": 246}}},
        {"type": "Chain", "description": "Alpha-trypsin chain 1",
         "location": {"start": {"value": 24}, "end": {"value": 148}}},
        {"type": "Chain", "description": "Alpha-trypsin chain 2",
         "location": {"start": {"value": 149}, "end": {"value": 246}}},
    ]
    monkeypatch.setattr(up, "_getUniProtJson", lambda url: _payload(feats, 246))

    spans = up.uniprotMatureChains("P00760")
    assert [(s["start"], s["end"]) for s in spans] == [(24, 148), (24, 246), (149, 246)]
    assert spans[1]["description"] == "Serine protease 1"
    assert all(s["type"] == "Chain" for s in spans)


def test_mature_chains_synthesised_when_no_chain_feature(monkeypatch):
    from moleculekit import uniprot as up

    feats = [
        {"type": "Signal", "location": {"start": {"value": 1}, "end": {"value": 24}}},
    ]
    monkeypatch.setattr(up, "_getUniProtJson", lambda url: _payload(feats, 1210))

    spans = up.uniprotMatureChains("Q99999")
    assert spans == [
        {"start": 25, "end": 1210, "type": "synthesised", "description": "precursor minus Signal"}
    ]


def test_mature_chains_whole_precursor_when_no_features(monkeypatch):
    from moleculekit import uniprot as up

    monkeypatch.setattr(up, "_getUniProtJson", lambda url: _payload([], 100))
    spans = up.uniprotMatureChains("Q00000")
    assert spans == [
        {"start": 1, "end": 100, "type": "synthesised", "description": "precursor"}
    ]


def test_fully_cleaved_precursor_yields_no_span(monkeypatch):
    """P13948 is a 21-residue propeptide and nothing else: no mature chain."""
    from moleculekit import uniprot as up

    feats = [
        {"type": "Propeptide", "location": {"start": {"value": 1}, "end": {"value": 21}}},
    ]
    monkeypatch.setattr(up, "_getUniProtJson", lambda url: _payload(feats, 21))
    assert up.uniprotMatureChains("P13948") == []


def test_signal_plus_propeptide_covering_everything_yields_no_span(monkeypatch):
    """P05223-shaped: signal 1-26 then propeptide 27-136 of a 136-residue entry."""
    from moleculekit import uniprot as up

    feats = [
        {"type": "Signal", "location": {"start": {"value": 1}, "end": {"value": 26}}},
        {"type": "Propeptide", "location": {"start": {"value": 27}, "end": {"value": 136}}},
    ]
    monkeypatch.setattr(up, "_getUniProtJson", lambda url: _payload(feats, 136))
    assert up.uniprotMatureChains("P05223") == []


def test_trailing_propeptide_is_trimmed_from_the_c_terminus(monkeypatch):
    """A C-terminal propeptide is cleaved too, so it is not part of the mature end."""
    from moleculekit import uniprot as up

    feats = [
        {"type": "Signal", "location": {"start": {"value": 1}, "end": {"value": 20}}},
        {"type": "Propeptide", "location": {"start": {"value": 200}, "end": {"value": 230}}},
    ]
    monkeypatch.setattr(up, "_getUniProtJson", lambda url: _payload(feats, 230))
    assert up.uniprotMatureChains("Q11111") == [
        {"start": 21, "end": 199, "type": "synthesised",
         "description": "precursor minus Propeptide, Signal"}
    ]


def test_interior_propeptide_is_not_treated_as_a_leader(monkeypatch):
    """A propeptide touching neither terminus says nothing about either end."""
    from moleculekit import uniprot as up

    feats = [
        {"type": "Propeptide", "location": {"start": {"value": 50}, "end": {"value": 60}}},
    ]
    monkeypatch.setattr(up, "_getUniProtJson", lambda url: _payload(feats, 100))
    assert up.uniprotMatureChains("Q22222") == [
        {"start": 1, "end": 100, "type": "synthesised", "description": "precursor"}
    ]


def test_inverted_parsed_spans_are_dropped(monkeypatch):
    """Malformed Chain features must never reach a caller as a usable span."""
    from moleculekit import uniprot as up

    feats = [
        {"type": "Chain", "description": "bad",
         "location": {"start": {"value": 40}, "end": {"value": 10}}},
        {"type": "Chain", "description": "good",
         "location": {"start": {"value": 1}, "end": {"value": 50}}},
    ]
    monkeypatch.setattr(up, "_getUniProtJson", lambda url: _payload(feats, 50))
    spans = up.uniprotMatureChains("Q33333")
    assert [(s["start"], s["end"]) for s in spans] == [(1, 50)]


def test_trim_reports_leading_offset():
    mol = Molecule(os.path.join(curr_dir, "pdb", "3ptb.pdb"))
    chain = next(iter(mol.getSequence(dict_key="chain", sel="protein", _logger=False)))
    obs = mol.getSequence(dict_key="chain", sel="protein", _logger=False)[chain]
    # A precursor with 5 extra leading and 3 extra trailing residues.
    precursor = "AAAAA" + obs + "CCC"

    trimmed_only = trimPrecursorSequences(mol, {chain: precursor})
    trimmed, offsets = trimPrecursorSequences(
        mol, {chain: precursor}, return_offsets=True
    )

    assert trimmed_only == trimmed          # default return is unchanged
    assert offsets[chain] == 5
    assert trimmed[chain] == obs
