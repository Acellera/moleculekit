import json
from unittest import mock

import numpy as np
import pytest
from moleculekit.molecule import Molecule
from moleculekit.rcsb import (
    rcsbFetchLigandInfo,
    rcsbFetchLigandSmiles,
    rcsbIsMembraneProtein,
    rcsbSequenceSearch,
    resolveFullSequences,
)


def test_fetch_ligand_info_returns_record():
    info = rcsbFetchLigandInfo("BEN")
    assert isinstance(info, dict)
    assert "rcsb_chem_comp_descriptor" in info
    assert info["rcsb_chem_comp_descriptor"]["comp_id"] == "BEN"
    # the full record exposes per-program SMILES variants too
    assert "pdbx_chem_comp_descriptor" in info


def test_fetch_ligand_smiles_stereo_default():
    # benzamidine, stereo SMILES carries the /N=C(\...)/ double-bond geometry
    smi = rcsbFetchLigandSmiles("BEN")
    assert isinstance(smi, str) and len(smi) > 0
    assert "c1ccccc1" in smi


def test_fetch_ligand_smiles_non_stereo():
    smi = rcsbFetchLigandSmiles("BEN", stereo=False)
    assert smi == "[H]N=C(c1ccccc1)N"


def test_fetch_ligand_info_unknown_code_raises():
    with pytest.raises(RuntimeError):
        rcsbFetchLigandInfo("ZZZZ")


def test_lowercase_code_is_accepted():
    smi = rcsbFetchLigandSmiles("ben")
    assert "c1ccccc1" in smi


def test_default_program_is_openeye():
    # explicit OpenEye matches the default (curated rcsb_chem_comp_descriptor)
    assert rcsbFetchLigandSmiles("BEN", program="OpenEye") == rcsbFetchLigandSmiles("BEN")


def test_program_cactvs():
    assert rcsbFetchLigandSmiles("BEN", program="CACTVS") == "NC(=N)c1ccccc1"
    # CACTVS differs from the OpenEye default for this ligand
    assert rcsbFetchLigandSmiles("BEN", program="CACTVS") != rcsbFetchLigandSmiles("BEN")


def test_program_match_is_case_insensitive():
    assert rcsbFetchLigandSmiles("BEN", program="cactvs") == "NC(=N)c1ccccc1"


def test_program_falls_back_to_other_type_when_canonical_absent():
    # ACDLabs only provides a plain SMILES row for BEN (no SMILES_CANONICAL),
    # so a stereo request falls back to it rather than raising.
    assert rcsbFetchLigandSmiles("BEN", program="ACDLabs") == "[N@H]=C(N)c1ccccc1"


def test_unknown_program_raises():
    with pytest.raises(RuntimeError):
        rcsbFetchLigandSmiles("BEN", program="Nonesuch")


def test_rcsb_sequence_search_parses_hits():
    fake = {
        "total_count": 2,
        "result_set": [
            {
                "identifier": "132L_1",
                "score": 1.0,
                "services": [
                    {"nodes": [{"match_context": [{"sequence_identity": 1.0}]}]}
                ],
            },
            {
                "identifier": "193L_1",
                "score": 0.9,
                "services": [
                    {"nodes": [{"match_context": [{"sequence_identity": 0.95}]}]}
                ],
            },
        ],
    }

    class FakeResp:
        def read(self):
            return json.dumps(fake).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with mock.patch("urllib.request.urlopen", return_value=FakeResp()):
        hits = rcsbSequenceSearch("ACDEFGHIKLMNPQRSTVWY")
    assert hits[0]["polymer_entity_id"] == "132L_1"
    assert hits[0]["identity"] == 1.0
    assert hits[1]["identity"] == 0.95


def _tiny_protein():
    # 3 residues ALA-GLY-SER on chain A. moleculekit's "protein" atomselect
    # requires full N-CA-C-O backbone atoms plus guessed bonds between them
    # (a lone CA per residue with no bonds is not recognized as protein), so
    # each residue gets a minimal backbone laid out along a line with
    # bond-length spacing for guessBonds() to connect.
    resnames = ["ALA", "GLY", "SER"]
    names = ["N", "CA", "C", "O"] * len(resnames)
    m = Molecule().empty(len(names))
    m.name[:] = names
    m.resname[:] = np.repeat(resnames, 4)
    m.resid[:] = np.repeat([1, 2, 3], 4)
    m.chain[:] = "A"
    m.segid[:] = "P"
    m.record[:] = "ATOM"
    m.element[:] = [n[0] for n in names]
    coords = np.zeros((len(names), 3), dtype=np.float32)
    coords[:, 0] = np.arange(len(names)) * 1.45
    m.coords = coords.reshape(len(names), 3, 1)
    m.guessBonds()
    return m


def test_resolve_full_sequences_pdbid_uses_entities():
    m = _tiny_protein()
    with mock.patch(
        "moleculekit.rcsb._entity_sequences_for_pdbid",
        return_value={"A": "AGSAA"},  # full = observed AGS + 2 missing
    ):
        res = resolveFullSequences(m, pdbid="XXXX")
    assert res["A"]["sequence"] == "AGSAA"
    assert res["A"]["source"] == "pdb_entity"
    assert res["A"]["identity"] == 1.0


def test_resolve_full_sequences_search_path():
    m = _tiny_protein()
    with mock.patch(
        "moleculekit.rcsb.rcsbSequenceSearch",
        return_value=[{"polymer_entity_id": "132L_1", "identity": 0.98, "score": 1.0}],
    ), mock.patch(
        "moleculekit.rcsb._get_pdb_entity_sequences",
        return_value={"132L_1": "AGSAA"},
    ):
        res = resolveFullSequences(m)
    assert res["A"]["sequence"] == "AGSAA"
    assert res["A"]["source"] == "sequence_search"
    assert res["A"]["identity"] == 0.98


def test_resolve_full_sequences_search_keeps_entity_id():
    # The hit's polymer entity id is the only trace of WHICH deposited entry a
    # file input matches; surveyStructure derives its candidate PDB id from it.
    m = _tiny_protein()
    with mock.patch(
        "moleculekit.rcsb.rcsbSequenceSearch",
        return_value=[{"polymer_entity_id": "132L_1", "identity": 0.98, "score": 1.0}],
    ), mock.patch(
        "moleculekit.rcsb._get_pdb_entity_sequences",
        return_value={"132L_1": "AGSAA"},
    ):
        res = resolveFullSequences(m)
    assert res["A"]["entity_id"] == "132L_1"


def test_resolve_full_sequences_pdbid_entity_id_is_none():
    # With a known pdbid there is no search hit; entity_id is explicitly None
    # so consumers can key on it without hasattr/get dances.
    m = _tiny_protein()
    with mock.patch(
        "moleculekit.rcsb._entity_sequences_for_pdbid",
        return_value={"A": "AGSAA"},
    ):
        res = resolveFullSequences(m, pdbid="XXXX")
    assert res["A"]["entity_id"] is None


def test_is_membrane_protein_true_from_keywords():
    fake = {
        "struct_keywords": {
            "pdbx_keywords": "MEMBRANE PROTEIN",
            "text": "GPCR, integral membrane protein",
        }
    }
    with mock.patch("moleculekit.rcsb._getRCSBjson", return_value=fake):
        assert rcsbIsMembraneProtein("7q5b") is True


def test_is_membrane_protein_false_for_soluble():
    fake = {
        "struct_keywords": {
            "pdbx_keywords": "HYDROLASE",
            "text": "serine protease, trypsin",
        }
    }
    with mock.patch("moleculekit.rcsb._getRCSBjson", return_value=fake):
        assert rcsbIsMembraneProtein("3ptb") is False


def test_is_membrane_protein_missing_keywords_is_false():
    with mock.patch("moleculekit.rcsb._getRCSBjson", return_value={}):
        assert rcsbIsMembraneProtein("1abc") is False
