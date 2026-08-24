import numpy as np
import os
import pytest
from os import path


curr_dir = os.path.dirname(os.path.abspath(__file__))


def test_autoSegment2_deprecated_shim():
    import warnings
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment, autoSegment2

    mol = Molecule("3ptb")
    # autoSegment2 must warn and produce the same result as autoSegment for the
    # same selection (it is now a thin forwarding shim).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shim = autoSegment2(
            mol, sel="protein or resname ACE NME", fields=("chain", "segid")
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    ref = autoSegment(
        mol, sel="protein or resname ACE NME", fields=("chain", "segid")
    )
    assert np.array_equal(shim.segid, ref.segid)
    assert np.array_equal(shim.chain, ref.chain)


def test_autoSegment_classify():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import _classify_residues

    mol = Molecule("3ptb")
    cats, _ = _classify_residues(mol, mol.atomselect("all"))
    # 3PTB: one protein chain, one CA ion, one BEN ligand, then waters
    assert "protein" in cats
    assert "ion" in cats  # the calcium
    assert "water" in cats
    assert "other" in cats  # benzamidine
    # No nucleic in 3PTB
    assert "nucleic" not in cats


def test_autoSegment_linked():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import _polymer_linked, _classify_residues

    mol = Molecule("3ptb")
    sel = mol.atomselect("protein")
    cats, residue_idx = _classify_residues(mol, sel)
    # Consecutive protein residues in 3PTB are peptide-bonded -> linked
    linked = [
        _polymer_linked(
            mol,
            residue_idx[i - 1],
            residue_idx[i],
            "protein",
            protein_cutoff=2.0,
            nucleic_cutoff=2.2,
            ca_fallback_cutoff=5.0,
            nucleic_fallback_cutoff=3.2,
        )
        for i in range(1, len(residue_idx))
    ]
    # The vast majority of consecutive residues are linked (allow chain breaks)
    assert np.mean(linked) > 0.95


def _isopeptide_dipeptide():
    """Two residues whose ONLY backbone link is a non-standard isopeptide bond:
    prev (GLU) side-chain CD is bonded to curr (ALA) backbone N, while the
    standard C(prev)-N(curr) and CA-CA distances are both far beyond the cutoffs
    (a microcystin-style gamma-glutamyl link, 1FJM FGA->DAM)."""
    from moleculekit.molecule import Molecule

    names = ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "N", "CA", "C", "O", "CB"]
    elems = ["N", "C", "C", "O", "C", "C", "C", "O", "N", "C", "C", "O", "C"]
    resid = [1] * 8 + [2] * 5
    resname = ["GLU"] * 8 + ["ALA"] * 5
    coords = np.array([
        [0, 0, 0], [1.5, 0, 0], [2.5, 1, 0], [2.5, 2, 0], [1.5, -1.5, 0],
        [1.5, -3.0, 0], [1.5, -4.5, 0], [2.5, -5.0, 0],
        [1.5, -5.83, 0], [2.5, -6.5, 0], [3.5, -6.0, 0], [3.5, -5.0, 0], [2.5, -8.0, 0],
    ], dtype=np.float32)
    mol = Molecule().empty(len(names))
    mol.name[:] = names
    mol.element[:] = elems
    mol.resid[:] = resid
    mol.resname[:] = resname
    mol.chain[:] = "A"
    mol.segid[:] = ""
    mol.record[:] = "ATOM"
    mol.coords = coords.reshape(-1, 3, 1)
    bonds = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (6, 7),
             (8, 9), (9, 10), (10, 11), (9, 12), (6, 8)]  # last = CD(GLU) - N(ALA)
    mol.bonds = np.array(bonds, dtype=np.uint32)
    mol.bondtype = np.array(["1"] * len(bonds), dtype=object)
    return mol


def test_autoSegment_respects_nonstandard_backbone_bond():
    """autoSegment must keep two residues in one segment when they are joined by
    a non-standard backbone bond (isopeptide on curr's N), even though the
    standard C-N / CA-CA distances exceed the cutoffs. Distance-only continuity
    wrongly splits them (1FJM's microcystin FGA->DAM)."""
    from moleculekit.tools.autosegment import _polymer_linked, autoSegment

    mol = _isopeptide_dipeptide()
    linked = _polymer_linked(
        mol, np.where(mol.resid == 1)[0], np.where(mol.resid == 2)[0], "protein",
        protein_cutoff=2.0, nucleic_cutoff=2.2,
        ca_fallback_cutoff=5.0, nucleic_fallback_cutoff=3.2,
    )
    assert linked is True, "isopeptide-linked residues should be backbone-continuous"

    out = autoSegment(mol, fields=("segid",), _logger=False)
    assert len(set(out.segid.tolist())) == 1, (
        f"expected 1 segment, got {sorted(set(out.segid.tolist()))}"
    )


def test_autoSegment_1aud_continuous():
    from os import path
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol = Molecule(path.join(curr_dir, "test_autosegment", "1aud.pdb"))
    # Only the RNA; it has a resid jump 30->33 but a continuous backbone.
    out = autoSegment(mol, sel="nucleic", fields=("segid",))
    rna_segids = set(np.unique(out.segid[out.atomselect("nucleic")]))
    assert len(rna_segids) == 1, f"expected 1 RNA segment, got {rna_segids}"


def test_autoSegment_3ptb_buckets():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol = Molecule("3ptb")
    out = autoSegment(mol, fields=("chain", "segid"))

    # Same index ranges as the existing test_autosegment_detailed
    prot_idx = np.arange(1629)
    ca_idx = np.array([1629])
    ben_idx = np.arange(1630, 1639)
    water_idx = np.arange(1639, 1701)

    # One protein segment over the whole protein
    assert len(set(out.segid[prot_idx])) == 1
    # Calcium ion is its own (ion) segment
    assert len(set(out.segid[ca_idx])) == 1
    # Benzamidine is one "other" segment
    assert len(set(out.segid[ben_idx])) == 1
    # All waters collapse into a single segment
    assert len(set(out.segid[water_idx])) == 1
    # The four groups are four distinct segids
    groups = [
        out.segid[prot_idx][0],
        out.segid[ca_idx][0],
        out.segid[ben_idx][0],
        out.segid[water_idx][0],
    ]
    assert len(set(groups)) == 4


def test_autoSegment_distinct_resids_for_collapsed_ions():
    """Ions distinguished only by chain (same resid in different chains - e.g.
    7BTI's five Mg, each resid 401 in chains A-E) collapse into one ion segment.
    They must come out with distinct resids within that segment, or the
    downstream (resid, insertion, segid) renumbering folds them into one residue
    and the duplicates are silently dropped (only 1 of 5 Mg survives the build)."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol = Molecule().empty(3)
    mol.name[:] = "MG"
    mol.element[:] = "Mg"
    mol.resname[:] = "MG"
    mol.resid[:] = [401, 401, 401]
    mol.chain[:] = ["A", "B", "C"]
    mol.segid[:] = ["A", "B", "C"]
    mol.record[:] = "HETATM"
    mol.coords = np.array(
        [[0, 0, 0], [20, 0, 0], [40, 0, 0]], np.float32
    ).reshape(3, 3, 1)

    out = autoSegment(mol, fields=("segid", "chain"), _logger=False)
    # collapsed into a single ion segment, but three distinct resids within it
    assert len(set(out.segid.tolist())) == 1
    assert len(set(out.resid.tolist())) == 3


def test_autoSegment_system_matrix():
    from os import path
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    expected = {
        "5vbl": 5,   # isopeptide bond -> peptide stays one segment per chain
        "4tot": 15,
        "8qfz": 5,   # scaffolded / bicycle (chain A has a real 21A backbone gap)
        "8qu4": 6,   # stapled peptide (chain A) keeps its ACE + NH2 caps in-chain
        "1r1j": 7,   # covalent sugars -> own "other" segment(s)
        "2kdc": 3,   # membrane protein
        "1bl8": 6,   # ion channel
        "2b5i": 14,
        "1u5u": 5,   # ferroheme cofactor
    }
    for pid, n in expected.items():
        mol = Molecule(path.join(curr_dir, "test_autosegment", f"{pid}.pdb"))
        out = autoSegment(mol, fields=("chain", "segid"))
        got = len(np.unique(out.segid))
        assert got == n, f"{pid}: expected {n} segments, got {got}"


def test_autoSegment_chain_change_splits():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol = Molecule("3ptb")
    prot = mol.atomselect("protein")
    # Artificially relabel the second half of the protein to a different chain.
    prot_idx = np.where(prot)[0]
    half = prot_idx[len(prot_idx) // 2 :]
    mol.chain[half] = "Z"

    out = autoSegment(mol, sel="protein", fields=("segid",))
    # Despite continuous backbone, the chain change must split into >= 2 segments
    assert len(set(out.segid[prot_idx])) >= 2


def test_autoSegment_single_other_segment():
    from os import path
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment, _classify_residues

    # 3gbn has several distinct 'other' molecules (EDO, GOL, a glycan tree, ...)
    mol = Molecule("3gbn")
    cats, residue_idx = _classify_residues(mol, mol.atomselect("all"))
    other_atoms = np.hstack(
        [residue_idx[i] for i, c in enumerate(cats) if c == "other"]
    )

    split = autoSegment(mol, fields=("segid",), single_other_segment=False)
    merged = autoSegment(mol, fields=("segid",), single_other_segment=True)

    # Splitting yields several 'other' segments; merging yields exactly one.
    assert len(set(split.segid[other_atoms])) > 1
    assert len(set(merged.segid[other_atoms])) == 1
    # The single 'other' segment must not collide with polymer/water/ion segments
    non_other = np.ones(mol.numAtoms, dtype=bool)
    non_other[other_atoms] = False
    assert merged.segid[other_atoms][0] not in set(merged.segid[non_other])


def test_autoSegment_gfp_chromophore():
    # 1gfl: the GFP chromophore residue 65 (SER) is missing its backbone O, so
    # the standard "backbone" selection drops it. autoSegment classifies it by
    # N/CA/C presence and keeps each chain continuous through it.
    from os import path
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol = Molecule(path.join(curr_dir, "test_autosegment", "1gfl.pdb"))
    out = autoSegment(mol, fields=("segid",))
    prot = mol.atomselect("protein")
    # One segment per chain (A, B); the chromophore does NOT cause an extra split
    assert len(set(out.segid[prot])) == 2
    for ch in ("A", "B"):
        before = out.segid[(mol.chain == ch) & (mol.resid == 64)]
        after = out.segid[(mol.chain == ch) & (mol.resid == 66)]
        assert before[0] == after[0], f"chromophore split chain {ch}"


def test_autoSegment_missing_carbonyl_continuity():
    # 1hgu: residues 148/151 are missing their carbonyl O (AS2 over-splits there),
    # while 37-39 are a genuine missing loop. autoSegment stays continuous through
    # the missing-O residues but splits at the real gap.
    from os import path
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol = Molecule(path.join(curr_dir, "test_autosegment", "1hgu.pdb"))
    out = autoSegment(mol, fields=("segid",))
    prot = mol.atomselect("protein")
    assert len(set(out.segid[prot])) == 2

    def seg(rid):
        return out.segid[(mol.chain == "A") & (mol.resid == rid)][0]

    # continuity across the missing-O residue 148
    assert seg(147) == seg(148) == seg(149)
    # real gap 36->40 splits
    assert seg(36) != seg(40)


def test_autoSegment_heavy_water_dod():
    # 2mb5 is a neutron structure with 89 DOD (D2O) residues. With DOD recognized
    # as water they collapse into a single water segment instead of 89 'other' ones.
    from os import path
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol = Molecule(path.join(curr_dir, "test_autosegment", "2mb5.pdb"))
    dod = mol.resname == "DOD"
    assert dod.any()
    out = autoSegment(mol, fields=("segid",))
    assert len(set(out.segid[dod])) == 1


def test_autoSegment_glycan_tree():
    # 3gbn: an N-linked glycan tree (NAG-NAG-BMA-MAN) must stay together as one
    # connected-component 'other' segment, separate from isolated small molecules.
    from os import path
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment, _classify_residues
    from collections import defaultdict

    mol = Molecule(path.join(curr_dir, "test_autosegment", "3gbn.pdb"))
    out = autoSegment(mol, fields=("segid",))
    cats, residue_idx = _classify_residues(mol, mol.atomselect("all"))
    seg_res = defaultdict(list)
    for i, c in enumerate(cats):
        if c == "other":
            seg_res[out.segid[residue_idx[i][0]]].append(mol.resname[residue_idx[i][0]])
    # Exactly one 'other' segment is a multi-residue sugar tree
    multi = [rs for rs in seg_res.values() if len(rs) > 1]
    assert len(multi) == 1
    assert set(multi[0]) <= {"NAG", "BMA", "MAN", "FUC", "GAL"}
    assert len(multi[0]) >= 3


def test_autoSegment_5mat_internal_gaps():
    # 5MAT (renin) has two protein chains (A, C), each with a real missing-loop
    # gap, so autoSegment yields 4 polymer segments. The gaps are detected by
    # backbone distance (C-N 4.4 A in chain A, 10.4 A in chain C), not resids.
    from os import path
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol = Molecule(path.join(curr_dir, "test_autosegment", "5mat.pdb"))
    out = autoSegment(mol, fields=("chain", "segid"))
    prot = mol.atomselect("protein")
    assert sorted(set(mol.chain[prot])) == ["A", "C"]
    assert len(set(out.segid[prot])) == 4

    def seg(ch, rid):
        return out.segid[(mol.chain == ch) & (mol.resid == rid)][0]

    # Each chain splits once at its missing-loop gap
    assert seg("A", 98) != seg("A", 101)
    assert seg("C", 98) != seg("C", 104)
    # ...but residues on the same side of a gap stay together
    assert seg("A", 55) == seg("A", 98)


def test_autoSegment_classifies_element_symbol_metal_ion():
    # A standalone monatomic metal ion is named after its element symbol (FE,
    # NI, ...) and may be absent from the VMD-mirroring ``ion_resnames`` list;
    # it must still segment as an ion. A polyatomic residue whose code collides
    # with an element symbol must not (single-atom guard).
    import numpy as np
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import _classify_residues

    mol = Molecule().empty(4)
    mol.resname[:] = ["FE", "NI", "NI", "CA"]
    mol.name[:] = ["FE", "NI1", "NI2", "CA"]
    mol.element[:] = ["Fe", "Ni", "Ni", "Ca"]
    mol.resid[:] = [1, 2, 2, 3]
    mol.record[:] = "HETATM"
    mol.coords = np.zeros((4, 3, 1), dtype=np.float32)

    cats, residue_idx = _classify_residues(mol, mol.atomselect("all"))
    cat_by_resname = {
        str(mol.resname[idx[0]]): cat for cat, idx in zip(cats, residue_idx)
    }
    assert cat_by_resname["FE"] == "ion"  # element-symbol metal not in ion_resnames
    assert cat_by_resname["CA"] == "ion"  # in ion_resnames (sanity)
    assert cat_by_resname["NI"] != "ion"  # 2-atom residue excluded by single-atom guard


def _chain_map_key(mol, i):
    """The exact key format autoSegment's return_chain_map dict uses, built
    from one atom's residue fields on the returned molecule."""
    return f"{mol.chain[i]}:{mol.resid[i]}:{mol.insertion[i]}:{mol.segid[i]}"


def test_autoSegment_return_chain_map_is_opt_in():
    """Existing callers must be untouched: no second return value by default."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol = Molecule(path.join(curr_dir, "pdb", "1a25.pdb"))
    out = autoSegment(mol, fields=("segid", "chain"), _logger=False)
    from moleculekit.molecule import Molecule as _M

    assert isinstance(out, _M), "default return must stay a bare Molecule"


def test_chain_map_reports_a_merge_per_residue():
    """autoSegment collapses ions into one chain. 1A25 has a CA in chain A and
    another in chain B, so the merged chain came from two deposited chains and
    each ion has to be attributable to its own. The idiom this replaces samples
    one atom per new chain and mislabels every atom after the first."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol0 = Molecule(path.join(curr_dir, "pdb", "1a25.pdb"))
    ca_before = {
        (str(mol0.chain[i]), int(mol0.resid[i]))
        for i in mol0.atomselect("resname CA and not protein", indexes=True)
    }
    assert {c for c, _ in ca_before} == {"A", "B"}, "fixture must have CA in A and B"

    mol, cmap = autoSegment(
        mol0, fields=("segid", "chain"), return_chain_map=True, _logger=False
    )
    assert isinstance(cmap, dict)

    ca_after = mol.atomselect("resname CA and not protein", indexes=True)
    merged_chains = {str(mol.chain[i]) for i in ca_after}
    assert len(merged_chains) == 1, "ions should collapse into one chain"

    # Each ion resolves to its OWN deposited chain: three from A, three from B
    resolved = [cmap[_chain_map_key(mol, i)] for i in ca_after]
    assert sorted(resolved) == ["A", "A", "A", "B", "B", "B"]


def test_chain_map_reports_a_split():
    """One deposited chain becoming two must be recoverable from the dict: a
    comprehension grouping keys by their value finds more than one new chain
    behind the same deposited one."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol0 = Molecule(path.join(curr_dir, "test_autosegment", "1gfl.pdb"))
    mol, cmap = autoSegment(
        mol0, fields=("segid", "chain"), return_chain_map=True, _logger=False
    )
    new_chains_by_deposited = {}
    for key, deposited in cmap.items():
        new_chains_by_deposited.setdefault(deposited, set()).add(key.split(":")[0])
    split = {old: new for old, new in new_chains_by_deposited.items() if len(new) > 1}
    assert split, "1gfl has a chain break, so some chain must split"


def test_chain_map_covers_every_segmented_chain():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol0 = Molecule(path.join(curr_dir, "pdb", "1a25.pdb"))
    mol, cmap = autoSegment(
        mol0, fields=("segid", "chain"), return_chain_map=True, _logger=False
    )
    named_chains = {key.split(":")[0] for key in cmap}
    assert named_chains == set(mol.chain) - {""}


def test_chain_map_is_json_serializable():
    """The dict IS the serialized form: a JSON round trip must reproduce it
    exactly, with nothing in between."""
    import json
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol0 = Molecule(path.join(curr_dir, "pdb", "1a25.pdb"))
    _, cmap = autoSegment(
        mol0, fields=("segid", "chain"), return_chain_map=True, _logger=False
    )
    assert json.loads(json.dumps(cmap)) == cmap


def test_chain_map_json_round_trip_names_each_merged_ion():
    """A chain-level-only view would name all six of 1A25's calcium the same
    merged label. The per-residue dict, read back through JSON, must still
    name each one its own deposited chain."""
    import json
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol0 = Molecule(path.join(curr_dir, "pdb", "1a25.pdb"))
    mol0.remove("water", _logger=False)
    mol, cmap = autoSegment(
        mol0, fields=("segid", "chain"), return_chain_map=True, _logger=False
    )
    ca = mol.atomselect("resname CA and not protein", indexes=True)
    assert len(ca) == 6, "1a25 must carry six calcium"
    merged = {str(mol.chain[i]) for i in ca}
    assert len(merged) == 1, "the six calcium must have merged into one chain"

    direct = [cmap[_chain_map_key(mol, i)] for i in ca]
    assert direct == ["A", "A", "A", "B", "B", "B"]

    back = json.loads(json.dumps(cmap))
    through_json = [back[_chain_map_key(mol, i)] for i in ca]
    assert through_json == direct


def test_chain_map_has_no_entry_for_an_unknown_residue():
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol0 = Molecule(path.join(curr_dir, "pdb", "1a25.pdb"))
    _, cmap = autoSegment(
        mol0, fields=("segid", "chain"), return_chain_map=True, _logger=False
    )
    assert "A:999999::P0" not in cmap


def test_chain_map_names_a_multicharacter_deposited_chain():
    """6ORV's deposited chain "AP" is what the user sees and what
    systemPrepare rejects, which is the whole reason the mapping exists."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol0 = Molecule(path.join(curr_dir, "test_autosegment", "6orv.cif"))
    multi = sorted({c for c in set(mol0.chain.tolist()) if len(c) > 1})
    assert multi, "fixture must carry a multi-character chain"

    mol, cmap = autoSegment(
        mol0, fields=("segid", "chain"), return_chain_map=True, _logger=False
    )
    assert all(len(c) == 1 for c in set(mol.chain.tolist()) - {""})

    # The multi-character deposited chain survives as a plain-string value
    assert multi[0] in cmap.values()


def test_chain_map_survives_the_ion_renumbering():
    """7BTI's MG ions collide on (resid, insertion) once merged into one chain,
    so autoSegment renumbers them. The map's keys must match the molecule it
    returns, not the resids the input had."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol0 = Molecule(path.join(curr_dir, "test_autosegment", "7bti.pdb"))
    before = {
        str(mol0.chain[i]): int(mol0.resid[i])
        for i in mol0.atomselect("resname MG", indexes=True)
    }
    assert len(before) > 1 and len(set(before.values())) == 1

    mol, cmap = autoSegment(
        mol0, fields=("segid", "chain"), return_chain_map=True, _logger=False
    )
    mg = mol.atomselect("resname MG", indexes=True)
    # Every renumbered ion resolves, and they resolve to distinct origins
    resolved = [cmap[_chain_map_key(mol, i)] for i in mg]
    assert sorted(resolved) == sorted(before)


def test_chain_map_split_on_1lv1():
    """The handoff's split case: one deposited chain becomes two."""
    from moleculekit.molecule import Molecule
    from moleculekit.tools.autosegment import autoSegment

    mol0 = Molecule(path.join(curr_dir, "test_autosegment", "1lv1.pdb"))
    mol, cmap = autoSegment(
        mol0, fields=("segid", "chain"), return_chain_map=True, _logger=False
    )
    new_chains_by_deposited = {}
    for key, deposited in cmap.items():
        new_chains_by_deposited.setdefault(deposited, set()).add(key.split(":")[0])
    split = {old: new for old, new in new_chains_by_deposited.items() if len(new) > 1}
    assert split, "1lv1 is the split fixture; some deposited chain must split"
