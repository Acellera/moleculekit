import pytest
from moleculekit.rcsb import rcsbFetchLigandInfo, rcsbFetchLigandSmiles


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
