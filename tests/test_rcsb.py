import json
from unittest import mock

import numpy as np
import pytest
import moleculekit.rcsb as rcsb
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
        # full = observed AGS + 2 missing
        return_value={"A": {"sequence": "AGSAA", "uniprot_refs": []}},
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
    ), mock.patch(
        # the search path also looks up the hit's UniProt cross-references; keep
        # this test off the network
        "moleculekit.rcsb._entity_uniprot_refs",
        return_value={},
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
    ), mock.patch(
        "moleculekit.rcsb._entity_uniprot_refs",
        return_value={},
    ):
        res = resolveFullSequences(m)
    assert res["A"]["entity_id"] == "132L_1"


def test_resolve_full_sequences_pdbid_entity_id_is_none():
    # With a known pdbid there is no search hit; entity_id is explicitly None
    # so consumers can key on it without hasattr/get dances.
    m = _tiny_protein()
    with mock.patch(
        "moleculekit.rcsb._entity_sequences_for_pdbid",
        return_value={"A": {"sequence": "AGSAA", "uniprot_refs": []}},
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


_3PTB_ENTITY_JSON = {
    "data": {"entries": [{"polymer_entities": [{
        "entity_poly": {"pdbx_seq_one_letter_code_can": "IVGGYTCGANTVPYQVSLN"},
        "rcsb_polymer_entity_container_identifiers": {
            "auth_asym_ids": ["A"],
            "reference_sequence_identifiers": [
                {"database_accession": "P00760", "database_name": "UniProt"}
            ],
        },
        "rcsb_polymer_entity_align": [{
            "reference_database_name": "UniProt",
            "reference_database_accession": "P00760",
            "aligned_regions": [
                {"entity_beg_seq_id": 1, "ref_beg_seq_id": 24, "length": 223}
            ],
        }],
    }]}]}
}


class _FakeResp:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_entity_sequences_carry_accession_and_regions(monkeypatch):
    monkeypatch.setattr(
        rcsb.urllib.request, "urlopen", lambda url, timeout=45: _FakeResp(_3PTB_ENTITY_JSON)
    )
    out = rcsb._entity_sequences_for_pdbid("3PTB")
    assert out["A"]["sequence"].startswith("IVGG")
    assert out["A"]["uniprot_refs"] == [
        {
            "accession": "P00760",
            "aligned_regions": [
                {"entity_beg_seq_id": 1, "ref_beg_seq_id": 24, "length": 223}
            ],
        }
    ]


def test_chimera_keeps_every_uniprot_row():
    """2RH1: one entity, beta2AR + a fused T4 lysozyme. Both rows must survive."""
    rows = [
        {
            "reference_database_name": "UniProt",
            "reference_database_accession": "P07550",
            "aligned_regions": [
                {"entity_beg_seq_id": 8, "ref_beg_seq_id": 1, "length": 230},
                {"entity_beg_seq_id": 399, "ref_beg_seq_id": 264, "length": 102},
            ],
        },
        {
            "reference_database_name": "UniProt",
            "reference_database_accession": "P00720",
            "aligned_regions": [
                {"entity_beg_seq_id": 238, "ref_beg_seq_id": 2, "length": 161}
            ],
        },
    ]

    refs = rcsb._uniprot_refs(rows)

    assert [r["accession"] for r in refs] == ["P07550", "P00720"]
    assert len(refs[0]["aligned_regions"]) == 2
    # The primary accession is the one covering the most entity residues (332 vs 161),
    # i.e. the actual protein rather than the fusion partner.
    assert rcsb._primary_accession(refs) == "P07550"


def test_non_uniprot_rows_are_ignored():
    rows = [
        {
            "reference_database_name": "GenBank",
            "reference_database_accession": "AAA12345",
            "aligned_regions": [
                {"entity_beg_seq_id": 1, "ref_beg_seq_id": 1, "length": 10}
            ],
        }
    ]
    assert rcsb._uniprot_refs(rows) == []
    assert rcsb._primary_accession([]) is None


def test_resolve_full_sequences_exposes_accession(monkeypatch):
    from moleculekit.molecule import Molecule

    mol = Molecule("3ptb")
    monkeypatch.setattr(
        rcsb,
        "_entity_sequences_for_pdbid",
        lambda pdbid: {
            "A": {
                "sequence": "IVGG",
                "uniprot_refs": [
                    {
                        "accession": "P00760",
                        "aligned_regions": [
                            {"entity_beg_seq_id": 1, "ref_beg_seq_id": 24, "length": 223}
                        ],
                    }
                ],
            }
        },
    )
    res = rcsb.resolveFullSequences(mol, pdbid="3PTB")
    assert res["A"]["source"] == "pdb_entity"
    assert res["A"]["accession"] == "P00760"
    assert res["A"]["uniprot_refs"][0]["aligned_regions"][0]["ref_beg_seq_id"] == 24


def test_entity_uniprot_refs_by_entity_id(monkeypatch):
    payload = {"data": {"polymer_entities": [{
        "rcsb_id": "132L_1",
        "rcsb_polymer_entity_align": [{
            "reference_database_name": "UniProt",
            "reference_database_accession": "P00698",
            "aligned_regions": [
                {"entity_beg_seq_id": 1, "ref_beg_seq_id": 19, "length": 129}
            ],
        }],
    }]}}
    monkeypatch.setattr(
        rcsb.urllib.request, "urlopen", lambda url, timeout=45: _FakeResp(payload)
    )
    refs = rcsb._entity_uniprot_refs(["132L_1"])
    assert refs["132L_1"][0]["accession"] == "P00698"
    assert refs["132L_1"][0]["aligned_regions"][0]["ref_beg_seq_id"] == 19


def test_resolve_full_sequences_search_path_carries_refs():
    # The search path has no entry id, so the mapping has to come from the hit's
    # own entity. RCSB's search service reports ids in either case, hence the
    # lowercase id here against the uppercase key of the refs lookup.
    m = _tiny_protein()
    entity_refs = [
        {
            "accession": "P00698",
            "aligned_regions": [
                {"entity_beg_seq_id": 1, "ref_beg_seq_id": 19, "length": 129}
            ],
        }
    ]
    with mock.patch(
        "moleculekit.rcsb.rcsbSequenceSearch",
        return_value=[{"polymer_entity_id": "132l_1", "identity": 0.98, "score": 1.0}],
    ), mock.patch(
        "moleculekit.rcsb._get_pdb_entity_sequences",
        return_value={"132L_1": "AGSAA"},
    ), mock.patch(
        "moleculekit.rcsb._entity_uniprot_refs",
        return_value={"132L_1": entity_refs},
    ):
        res = resolveFullSequences(m)
    assert res["A"]["accession"] == "P00698"
    assert res["A"]["uniprot_refs"] == entity_refs


def test_resolve_full_sequences_search_path_survives_refs_failure():
    # A failed cross-reference lookup must not lose the sequence we already have;
    # the chain simply ends up without a mapping.
    m = _tiny_protein()

    def _boom(entity_ids):
        raise RuntimeError("RCSB is down")

    with mock.patch(
        "moleculekit.rcsb.rcsbSequenceSearch",
        return_value=[{"polymer_entity_id": "132L_1", "identity": 0.98, "score": 1.0}],
    ), mock.patch(
        "moleculekit.rcsb._get_pdb_entity_sequences",
        return_value={"132L_1": "AGSAA"},
    ), mock.patch("moleculekit.rcsb._entity_uniprot_refs", _boom):
        res = resolveFullSequences(m)
    assert res["A"]["sequence"] == "AGSAA"
    assert res["A"]["accession"] is None
    assert res["A"]["uniprot_refs"] == []


def test_entity_sequences_fall_back_to_cross_reference_without_alignment():
    # Some entries carry a UniProt cross-reference but no SIFTS alignment rows.
    # The accession is still worth reporting; the regions are simply empty.
    payload = {
        "data": {"entries": [{"polymer_entities": [{
            "entity_poly": {"pdbx_seq_one_letter_code_can": "IVGG"},
            "rcsb_polymer_entity_container_identifiers": {
                "auth_asym_ids": ["A", "B"],
                "reference_sequence_identifiers": [
                    {"database_accession": "AAA12345", "database_name": "GenBank"},
                    {"database_accession": "P00760", "database_name": "UniProt"},
                ],
            },
            "rcsb_polymer_entity_align": None,
        }]}]}
    }
    with mock.patch.object(
        rcsb.urllib.request, "urlopen", lambda url, timeout=45: _FakeResp(payload)
    ):
        out = rcsb._entity_sequences_for_pdbid("3PTB")
    assert out["A"]["uniprot_refs"] == [
        {"accession": "P00760", "aligned_regions": []}
    ]
    # every auth chain of the entity gets the same mapping
    assert out["B"] == out["A"]
