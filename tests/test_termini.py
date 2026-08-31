import os

import pytest
from moleculekit.molecule import Molecule
from moleculekit.tools.modelling import detectSequenceGaps
from moleculekit.tools.termini import CAP_VOCABULARY, detectTermini

curr_dir = os.path.dirname(os.path.abspath(__file__))
PDB_3PTB = os.path.join(curr_dir, "pdb", "3ptb.pdb")

# The mature trypsin chain and its two autolysis products (see P00760).
TRYPSIN_SPANS = [
    {"start": 24, "end": 148, "type": "Chain", "description": "Alpha-trypsin chain 1"},
    {"start": 24, "end": 246, "type": "Chain", "description": "Serine protease 1"},
    {"start": 149, "end": 246, "type": "Chain", "description": "Alpha-trypsin chain 2"},
]


def _mol_and_ref():
    mol = Molecule(PDB_3PTB)
    chain = next(iter(mol.getSequence(dict_key="chain", sel="protein", _logger=False)))
    ref = mol.getSequence(dict_key="chain", sel="protein", _logger=False)[chain]
    return mol, chain, ref


def _meta(chain, **kw):
    base = {
        "source": "pdb_entity",
        "accession": "P00760",
        # entity residue 1 is UniProt 24; the structure covers the whole entity
        "uniprot_refs": [
            {
                "accession": "P00760",
                "aligned_regions": [
                    {"entity_beg_seq_id": 1, "ref_beg_seq_id": 24, "length": 223}
                ],
            }
        ],
        "trim_offset": 0,
    }
    base.update(kw)
    return {chain: base}


def test_flush_ends_matching_mature_chain_are_natural():
    mol, chain, ref = _mol_and_ref()
    gaps, _, _ = detectSequenceGaps(mol, {chain: ref})
    spans = dict(P00760=TRYPSIN_SPANS)

    term = detectTermini(mol, {chain: ref}, gaps, _meta(chain), spans)

    assert [t["end"] for t in term] == ["N", "C"]
    assert all(t["classification"] == "natural" for t in term)
    assert all(t["evidence"] == "uniprot_mature_chain" for t in term)
    assert all(t["proposed_cap"] == "none" for t in term)
    assert "Serine protease 1" in term[0]["matched_feature"]
    # 3PTB's mature N-terminus is Ile16 in the deposited numbering
    assert (term[0]["resid"], term[0]["resname"]) == (16, "ILE")
    assert term[0]["sel"] == 'chain "A" and resid 16'


def test_construct_boundary_is_truncated():
    mol, chain, ref = _mol_and_ref()
    gaps, _, _ = detectSequenceGaps(mol, {chain: ref})
    # Pretend the same coordinates are an internal fragment: entity residue 1 is
    # UniProt 696 and the mature chain runs 25-1210 (the EGFR situation).
    meta = _meta(
        chain,
        accession="P00533",
        uniprot_refs=[
            {
                "accession": "P00533",
                "aligned_regions": [
                    {"entity_beg_seq_id": 1, "ref_beg_seq_id": 696, "length": len(ref)}
                ],
            }
        ],
    )
    spans = {"P00533": [
        {"start": 25, "end": 1210, "type": "Chain", "description": "EGFR"}
    ]}

    term = detectTermini(mol, {chain: ref}, gaps, meta, spans)

    assert [t["classification"] for t in term] == ["truncated", "truncated"]
    assert [t["proposed_cap"] for t in term] == ["ACE", "NME"]
    assert all(t["matched_feature"] is None for t in term)


def test_terminal_gap_is_truncated_without_uniprot():
    mol, chain, ref = _mol_and_ref()
    # Reference extends 4 residues past the structure at both ends.
    padded = "AAAA" + ref + "CCCC"
    gaps, _, _ = detectSequenceGaps(mol, {chain: padded})
    # No accession at all: the terminal gaps alone must decide.
    meta = {chain: {"source": "user", "accession": None, "uniprot_refs": [],
                    "trim_offset": 0}}

    term = detectTermini(mol, {chain: padded}, gaps, meta, {})

    assert [t["classification"] for t in term] == ["truncated", "truncated"]
    assert [t["evidence"] for t in term] == ["terminal_gap", "terminal_gap"]
    assert [t["proposed_cap"] for t in term] == ["ACE", "NME"]


def test_no_evidence_is_unknown_with_no_proposal():
    mol, chain, ref = _mol_and_ref()
    gaps, _, _ = detectSequenceGaps(mol, {chain: ref})
    meta = {chain: {"source": "user", "accession": None, "uniprot_refs": [],
                    "trim_offset": 0}}

    term = detectTermini(mol, {chain: ref}, gaps, meta, {})

    assert [t["classification"] for t in term] == ["unknown", "unknown"]
    assert [t["evidence"] for t in term] == ["flush_no_evidence", "flush_no_evidence"]
    assert all(t["proposed_cap"] is None for t in term)
    assert all(t["cappable"] for t in term)      # unknown is still cappable


def test_user_accession_uses_trim_offset():
    mol, chain, ref = _mol_and_ref()
    gaps, _, _ = detectSequenceGaps(mol, {chain: ref})
    # Reference is the precursor trimmed by 23 leading residues, so reference
    # position 1 is UniProt 24 - the mature chain start.
    meta = {chain: {"source": "uniprot", "accession": "P00760",
                    "uniprot_refs": [], "trim_offset": 23}}

    term = detectTermini(mol, {chain: ref}, gaps, meta, {"P00760": TRYPSIN_SPANS})

    assert [t["classification"] for t in term] == ["natural", "natural"]


def test_chimera_classifies_each_end_against_its_own_accession():
    """2RH1-style fusion: the two ends belong to two different proteins.

    Entity 1..N is laid out as protein A (covered by P07550), a fused T4 lysozyme
    (P00720), then protein A again. Here the structure's N-terminus falls in the
    fusion partner's row and its C-terminus in the receptor's, so each end must be
    judged against the spans of the accession that actually covers it.
    """
    mol, chain, ref = _mol_and_ref()
    gaps, _, _ = detectSequenceGaps(mol, {chain: ref})
    half = len(ref) // 2
    meta = _meta(
        chain,
        accession="P07550",
        uniprot_refs=[
            {   # covers the second half of the entity, ending at the mature C-term
                "accession": "P07550",
                "aligned_regions": [
                    {"entity_beg_seq_id": half + 1, "ref_beg_seq_id": 365,
                     "length": len(ref) - half}
                ],
            },
            {   # covers the first half; entity residue 1 is this protein's residue 2
                "accession": "P00720",
                "aligned_regions": [
                    {"entity_beg_seq_id": 1, "ref_beg_seq_id": 2, "length": half}
                ],
            },
        ],
    )
    spans = {
        "P07550": [{"start": 1, "end": 364 + (len(ref) - half),
                    "type": "Chain", "description": "Beta-2 adrenergic receptor"}],
        "P00720": [{"start": 1, "end": 164, "type": "Chain",
                    "description": "Endolysin"}],
    }

    term = detectTermini(mol, {chain: ref}, gaps, meta, spans)

    # N-term: P00720 position 2, which is not that chain's start (1) -> truncated
    assert term[0]["classification"] == "truncated"
    assert term[0]["accession"] == "P00720"
    # C-term: P07550's mature end -> natural, and the evidence names P07550
    assert term[1]["classification"] == "natural"
    assert term[1]["accession"] == "P07550"
    assert "Beta-2 adrenergic receptor" in term[1]["matched_feature"]


def test_non_canonical_terminus_is_not_cappable():
    mol, chain, ref = _mol_and_ref()
    gaps, _, _ = detectSequenceGaps(mol, {chain: ref})
    first = mol.getSequence(dict_key="chain", return_idx=True, sel="protein",
                            _logger=False)[1][chain][0]
    mol.resname[first] = "XYZ"      # something no force field can cap onto

    term = detectTermini(mol, {chain: ref}, gaps, _meta(chain),
                         {"P00760": TRYPSIN_SPANS})

    assert term[0]["cappable"] is False
    assert term[0]["proposed_cap"] is None
    assert term[0]["sel"] is None
    assert term[1]["cappable"] is True


def test_selection_escalates_when_resid_is_ambiguous():
    mol, chain, ref = _mol_and_ref()
    gaps, _, _ = detectSequenceGaps(mol, {chain: ref})
    idx = mol.getSequence(dict_key="chain", return_idx=True, sel="protein",
                          _logger=False)[1][chain]
    first_atoms, second_atoms = idx[0], idx[1]
    # Make the second residue "16A": now `resid 16` alone matches two residues.
    mol.resid[second_atoms] = int(mol.resid[first_atoms[0]])
    mol.insertion[second_atoms] = "A"

    term = detectTermini(mol, {chain: ref}, gaps, _meta(chain),
                         {"P00760": TRYPSIN_SPANS})

    assert term[0]["sel"] == 'chain "A" and resid 16 and insertion ""'


def test_cap_vocabulary_is_amber_only():
    assert CAP_VOCABULARY == ("none", "ACE", "NME", "NHE")


def test_terminal_gap_outranks_uniprot_evidence():
    """A gap must win even when UniProt would call the same end natural.

    Without the gap branch running first, the C-terminal position is computed as
    len(ref) - the reference's own end - which can match a mature-chain end and
    return `natural` for a demonstrably cut terminus.
    """
    mol, chain, ref = _mol_and_ref()
    padded = ref + "CCCC"                     # 4 unmodelled C-terminal residues
    gaps, _, _ = detectSequenceGaps(mol, {chain: padded})
    meta = _meta(
        chain,
        uniprot_refs=[
            {
                "accession": "P00760",
                "aligned_regions": [
                    {"entity_beg_seq_id": 1, "ref_beg_seq_id": 24, "length": len(padded)}
                ],
            }
        ],
    )
    # A span whose end coincides with the padded reference's last position, so a
    # UniProt-first implementation would answer "natural" for the C-terminus.
    spans = {"P00760": [
        {"start": 24, "end": 24 + len(padded) - 1, "type": "Chain", "description": "whole"}
    ]}

    term = detectTermini(mol, {chain: padded}, gaps, meta, spans)

    assert term[1]["classification"] == "truncated"
    assert term[1]["evidence"] == "terminal_gap"


def test_blank_chain_id_still_yields_a_usable_selection():
    """Structures with no chain identifiers are ordinary, not pathological."""
    mol = Molecule(os.path.join(curr_dir, "test_readers", "dialanine_solute.pdb"))
    mol.filter("protein", _logger=False)
    assert set(mol.chain.tolist()) == {""}     # the condition under test
    obs = mol.getSequence(dict_key="chain", sel="protein", _logger=False)[""]
    gaps, _, _ = detectSequenceGaps(mol, {"": obs})

    term = detectTermini(mol, {"": obs}, gaps, {"": {"source": "user",
                         "accession": None, "uniprot_refs": [], "trim_offset": 0}}, {})

    for t in term:
        # Either it is cappable and nameable, or it is neither - never cappable
        # with nothing to name it by.
        assert (t["sel"] is not None) == t["cappable"]
    assert any(t["sel"] is not None for t in term), "a blank chain must still be selectable"


def test_already_capped_terminus_is_not_offered_another_cap():
    """ACE/NME are invisible to sel="protein", so the residue under the cap must
    be recognised as already capped rather than capped a second time."""
    mol = Molecule(os.path.join(curr_dir, "test_readers", "dialanine_solute.pdb"))
    mol.filter("protein or resname ACE NME NHE", _logger=False)
    assert {"ACE", "NME"} <= set(mol.resname.tolist())   # the condition under test
    chain = ""
    obs = mol.getSequence(dict_key="chain", sel="protein", _logger=False)[chain]
    gaps, _, _ = detectSequenceGaps(mol, {chain: obs})

    term = detectTermini(mol, {chain: obs}, gaps, {chain: {"source": "user",
                         "accession": None, "uniprot_refs": [], "trim_offset": 0}}, {})

    assert [t["cappable"] for t in term] == [False, False]
    assert all(t["proposed_cap"] is None and t["sel"] is None for t in term)


def test_chain_without_gap_analysis_is_never_called_natural():
    """A chain skipped by gap detection has no flushness evidence, so its termini
    must be unknown - never natural on the strength of an assumed alignment."""
    mol, chain, ref = _mol_and_ref()
    # Filter first: the last 20 unique resids of the raw file are all HOH, so
    # removing them without this deletes waters and truncates nothing.
    mol.filter("protein", _logger=False)
    resids = sorted({int(r) for r in mol.resid})
    mol.remove(f"resid {' '.join(str(r) for r in resids[-20:])}", _logger=False)
    mol.set("resname", "ALY", sel=f"resid {resids[5]}")   # not a cappable resname

    # Assert the fixture really built the harm, not just the guard's trigger.
    observed = mol.getSequence(dict_key="chain", sel="protein", _logger=False)[chain]
    assert len(observed) == len(ref) - 20, "the C-terminus must actually be cut"

    gaps, skipped, _ = detectSequenceGaps(mol, {chain: ref})
    assert skipped == [chain] and gaps == []      # the conditions under test

    spans = {"P00760": [
        {"start": 24, "end": 24 + len(ref) - 1, "type": "Chain", "description": "whole"}
    ]}
    term = detectTermini(mol, {chain: ref}, gaps, _meta(chain), spans,
                         skipped_chains=skipped)

    assert [t["classification"] for t in term] == ["unknown", "unknown"]
    assert [t["evidence"] for t in term] == ["no_gap_analysis", "no_gap_analysis"]
    assert all(t["proposed_cap"] is None for t in term)


def test_canonical_chain_is_unaffected_by_the_skip_guard():
    """The guard must not make an ordinary chain unknown."""
    mol, chain, ref = _mol_and_ref()
    gaps, skipped, _ = detectSequenceGaps(mol, {chain: ref})
    assert skipped == []

    term = detectTermini(mol, {chain: ref}, gaps, _meta(chain),
                         {"P00760": TRYPSIN_SPANS}, skipped_chains=skipped)

    assert [t["classification"] for t in term] == ["natural", "natural"]


def test_unmodelled_internal_gap_gives_each_piece_its_own_ends():
    """Leave a gap unmodelled and the builder receives two pieces, not one chain.

    Both ends the gap creates need a cap; the chain's own two ends are still the
    mature chain's and must stay natural.
    """
    mol, chain, ref = _mol_and_ref()
    cut = mol.copy()
    cut.remove("resid 100 to 104", _logger=False)  # reference still has them
    gaps, _, _ = detectSequenceGaps(cut, {chain: ref})

    term = detectTermini(cut, {chain: ref}, gaps, _meta(chain),
                         {"P00760": TRYPSIN_SPANS})

    assert [(t["end"], t["resid"]) for t in term] == [
        ("N", 16), ("C", 99), ("N", 105), ("C", 245)
    ]
    assert [t["classification"] for t in term] == [
        "natural", "truncated", "truncated", "natural"
    ]
    assert [t["evidence"] for t in term][1:3] == ["internal_gap", "internal_gap"]
    assert [t["proposed_cap"] for t in term] == ["none", "NME", "ACE", "none"]


def test_every_unmodelled_gap_adds_a_piece():
    mol, chain, ref = _mol_and_ref()
    cut = mol.copy()
    cut.remove("resid 100 to 104 or resid 180 to 184", _logger=False)
    gaps, _, _ = detectSequenceGaps(cut, {chain: ref})
    assert len(gaps) == 2

    term = detectTermini(cut, {chain: ref}, gaps, _meta(chain),
                         {"P00760": TRYPSIN_SPANS})

    assert [(t["end"], t["resid"]) for t in term] == [
        ("N", 16), ("C", 99), ("N", 105), ("C", 179), ("N", 185), ("C", 245)
    ]
