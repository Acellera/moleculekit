import json
import os
import shutil

import numpy as np
from moleculekit.molecule import Molecule
from moleculekit.tools import survey as sv

curr_dir = os.path.dirname(os.path.abspath(__file__))
PDB_3PTB = os.path.join(curr_dir, "pdb", "3ptb.pdb")


def _observed(mol):
    seqs = mol.getSequence(dict_key="chain", sel="protein", _logger=False)
    return {c: s for c, s in seqs.items() if s}


def _mock_resolved(seqs, source="pdb_entity", identity=1.0, entity_id=None):
    return {
        c: {
            "sequence": s,
            "source": source,
            "identity": identity,
            "entity_id": entity_id,
        }
        for c, s in seqs.items()
    }


# ---------------------------------------------------------------------------
# surveyStructure
# ---------------------------------------------------------------------------


def test_survey_pdbid_writes_files_and_full_report(tmp_path, monkeypatch):
    obs = _observed(Molecule("3ptb"))
    monkeypatch.setattr(
        sv, "resolveFullSequences", lambda m, pdbid=None: _mock_resolved(obs)
    )
    monkeypatch.setattr(sv, "rcsbIsMembraneProtein", lambda p: False)

    outdir = str(tmp_path / "run")
    rep = sv.surveyStructure("3ptb", outdir=outdir)

    assert os.path.exists(os.path.join(outdir, "input.cif"))
    with open(os.path.join(outdir, "sequences.json")) as fh:
        assert json.load(fh) == obs
    with open(os.path.join(outdir, "survey.json")) as fh:
        data = json.load(fh)

    assert rep.pdbid == "3ptb"
    assert rep.membrane is False
    assert rep.candidate_pdbid is None
    assert rep.gaps == [] and data["gaps"] == []
    assert rep.mismatches == []
    assert rep.unresolved == []
    # 3PTB carries benzamidine; the calcium ion is excluded by detection.
    ben = [n for n in rep.nonstandard if n["resname"] == "BEN"]
    assert len(ben) == 1 and ben[0]["type"] == "free ligand"
    text = str(rep)
    assert "membrane" in text and "BEN" in text


def test_survey_file_input_reports_candidate_and_unknown_membrane(
    tmp_path, monkeypatch
):
    src = str(tmp_path / "unknown_structure.pdb")
    shutil.copy(PDB_3PTB, src)
    obs = _observed(Molecule(src))
    monkeypatch.setattr(
        sv,
        "resolveFullSequences",
        lambda m, pdbid=None: _mock_resolved(
            obs, source="sequence_search", identity=1.0, entity_id="3PTB_1"
        ),
    )

    def _no_membrane_lookup(p):
        raise AssertionError("membrane lookup attempted without a PDB id")

    monkeypatch.setattr(sv, "rcsbIsMembraneProtein", _no_membrane_lookup)

    rep = sv.surveyStructure(src, outdir=str(tmp_path / "run"))
    assert rep.pdbid is None
    assert rep.membrane is None
    assert rep.candidate_pdbid == "3PTB"
    assert "3PTB" in str(rep)


def _gapped_1m17():
    # Real EGFR case from the modelling tests: a structure with genuine
    # backbone breaks (internal reference gaps are only representable at real
    # breaks) and its true full sequence.
    gapped = os.path.join(curr_dir, "test_modelling", "1m17_gapped.pdb.gz")
    with open(os.path.join(curr_dir, "test_modelling", "1m17_full_sequence.txt")) as fh:
        return gapped, fh.read().strip()


def test_survey_detects_gaps_and_mismatches(tmp_path, monkeypatch):
    gapped, full = _gapped_1m17()
    obs = _observed(Molecule(gapped))
    chain = next(iter(obs))
    # Mutate the reference at a position that is observed in the structure so
    # exactly one aligned mismatch exists next to the real gaps.
    i = full.find(obs[chain][:10]) + 5
    orig = full[i]
    swap = "A" if orig != "A" else "G"
    reference = full[:i] + swap + full[i + 1 :]
    monkeypatch.setattr(
        sv,
        "resolveFullSequences",
        lambda m, pdbid=None: _mock_resolved({chain: reference}),
    )

    rep = sv.surveyStructure(gapped, outdir=str(tmp_path / "run"))
    assert len(rep.gaps) >= 1
    assert all(g["missing_seq"] for g in rep.gaps)
    assert len(rep.mismatches) == 1
    assert rep.mismatches[0]["observed"] == orig
    assert rep.mismatches[0]["reference"] == swap


def test_survey_keep_mutations_patches_the_reference(tmp_path, monkeypatch):
    gapped, full = _gapped_1m17()
    obs = _observed(Molecule(gapped))
    chain = next(iter(obs))
    i = full.find(obs[chain][:10]) + 5
    orig = full[i]
    swap = "A" if orig != "A" else "G"
    reference = full[:i] + swap + full[i + 1 :]
    monkeypatch.setattr(
        sv,
        "resolveFullSequences",
        lambda m, pdbid=None: _mock_resolved({chain: reference}),
    )

    outdir = str(tmp_path / "run")
    rep = sv.surveyStructure(gapped, outdir=outdir, keep_mutations=True)
    assert rep.mismatches == []
    with open(os.path.join(outdir, "sequences.json")) as fh:
        assert json.load(fh)[chain][i] == orig  # patched back to observed


def test_survey_rerun_reuses_sequences_json(tmp_path, monkeypatch):
    obs = _observed(Molecule("3ptb"))
    calls = []

    def _resolve(m, pdbid=None):
        calls.append(pdbid)
        return _mock_resolved(obs)

    monkeypatch.setattr(sv, "resolveFullSequences", _resolve)
    monkeypatch.setattr(sv, "rcsbIsMembraneProtein", lambda p: False)

    outdir = str(tmp_path / "run")
    sv.surveyStructure("3ptb", outdir=outdir)
    rep2 = sv.surveyStructure("3ptb", outdir=outdir)
    assert len(calls) == 1  # second run reused sequences.json
    chain = next(iter(obs))
    assert rep2.chains[chain]["source"] == "pdb_entity"  # meta survived the rerun


def test_survey_uniprot_accession_merge_and_precursor_trim(tmp_path, monkeypatch):
    obs = _observed(Molecule("3ptb"))
    chain = next(iter(obs))
    s = obs[chain]
    monkeypatch.setattr(sv, "resolveFullSequences", lambda m, pdbid=None: {})
    monkeypatch.setattr(sv, "rcsbIsMembraneProtein", lambda p: False)
    monkeypatch.setattr(sv, "uniprotSequence", lambda acc: "M" * 23 + s)

    outdir = str(tmp_path / "run")
    rep = sv.surveyStructure("3ptb", outdir=outdir)
    assert rep.unresolved == [chain]

    rep = sv.surveyStructure("3ptb", outdir=outdir, sequences={chain: "P00760"})
    assert rep.unresolved == []
    assert rep.chains[chain]["source"] == "uniprot"
    with open(os.path.join(outdir, "sequences.json")) as fh:
        assert json.load(fh)[chain] == s  # precursor overhang trimmed
    assert rep.gaps == []


def test_survey_raw_sequence_is_used_verbatim(tmp_path, monkeypatch):
    obs = _observed(Molecule("3ptb"))
    chain = next(iter(obs))
    s = obs[chain]
    monkeypatch.setattr(sv, "resolveFullSequences", lambda m, pdbid=None: {})
    monkeypatch.setattr(sv, "rcsbIsMembraneProtein", lambda p: False)

    def _no_uniprot(acc):
        raise AssertionError("a raw sequence must not trigger a UniProt fetch")

    monkeypatch.setattr(sv, "uniprotSequence", _no_uniprot)

    rep = sv.surveyStructure(
        "3ptb", outdir=str(tmp_path / "run"), sequences={chain: s}, trim=False
    )
    assert rep.chains[chain]["source"] == "user"
    assert rep.gaps == [] and rep.mismatches == []


# ---------------------------------------------------------------------------
# verifyBuildResult
# ---------------------------------------------------------------------------


def _chain_mol(nres=6, chain="A"):
    # N/CA/C laid along x at peptide-bond spacing, O offset in y, so
    # detectBackboneBreaks sees a continuous backbone.
    names = ["N", "CA", "C", "O"] * nres
    m = Molecule().empty(len(names))
    m.name[:] = names
    m.resname[:] = "ALA"
    m.resid[:] = np.repeat(np.arange(1, nres + 1), 4)
    m.chain[:] = chain
    m.segid[:] = "P0"
    m.record[:] = "ATOM"
    m.element[:] = [n[0] for n in names]
    coords = np.zeros((len(names), 3), dtype=np.float32)
    x = 0.0
    for i, name in enumerate(names):
        if name == "O":
            coords[i] = (x - 1.4, 1.2, 0.0)  # off-axis, next to its C
        else:
            coords[i] = (x, 0.0, 0.0)
            x += 1.4
    m.coords = coords.reshape(len(names), 3, 1)
    m.guessBonds()
    return m


def test_verify_clean_result(tmp_path):
    m = _chain_mol()
    ref = str(tmp_path / "input.pdb")
    out = str(tmp_path / "built.pdb")
    m.write(ref)
    m.write(out)
    rep = sv.verifyBuildResult(ref, out)
    assert rep.clean
    assert rep.breaks == [] and rep.caps == []
    assert rep.residues_in == rep.residues_out == {"A": 6}
    assert "CLEAN" in str(rep)


def test_verify_detects_break_and_lost_residues(tmp_path):
    m = _chain_mol()
    ref = str(tmp_path / "input.pdb")
    out = str(tmp_path / "built.pdb")
    m.write(ref)
    broken = m.copy()
    broken.remove("resid 3", _logger=False)
    broken.write(out)
    rep = sv.verifyBuildResult(ref, out)
    assert not rep.clean
    assert len(rep.breaks) == 1
    assert rep.residues_in == {"A": 6} and rep.residues_out == {"A": 5}
    assert "NOT CLEAN" in str(rep)


def test_verify_lists_caps(tmp_path):
    m = _chain_mol()
    ref = str(tmp_path / "input.pdb")
    out = str(tmp_path / "built.pdb")
    m.write(ref)
    capped = m.copy()
    capped.resname[capped.resid == 1] = "ACE"
    capped.write(out)
    rep = sv.verifyBuildResult(ref, out)
    assert [c["resname"] for c in rep.caps] == ["ACE"]
    assert rep.caps[0]["resid"] == 1
    # Caps are listed for the caller to judge (terminal cap = fine, capped
    # break = not) but do not decide the verdict by themselves: a capped break
    # is already caught by the break and residue-count checks.
    assert "ACE" in str(rep)
