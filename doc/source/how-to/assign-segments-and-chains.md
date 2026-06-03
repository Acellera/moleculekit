# How to assign segments and chains

## Goal

Derive `segid` and/or `chain` fields for a structure that lacks them, splitting the system into segments by following each polymer's physical backbone.

## Minimal example

```python
from moleculekit.molecule import Molecule
from moleculekit.tools.autosegment import autoSegment

mol = Molecule("3PTB")
mol = autoSegment(mol)
print(set(mol.segid))
```

## How segments are decided

A new segment starts between two consecutive residues when any of these holds: the backbone link distance exceeds the cutoff (protein `C(i)–N(i+1)`, nucleic `O3'(i)–P(i+1)`), the `chain` or `segid` already in the file changes, or the polymer type changes. Water collapses into one segment, ions into another, and the remaining ("other") molecules are split one segment per bonded molecule. Because continuity is read from coordinates, a gap in residue *numbering* with an intact backbone stays one segment, while a real spatial break is split.

## Parameters that matter

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `mol` | {py:class}`~moleculekit.molecule.Molecule` | required | Input molecule (a copy is returned; original is unchanged) |
| `sel` | `str` | `"all"` | Restrict segmentation to this atom selection; atoms outside keep their existing `chain`/`segid` |
| `basename` | `str` | `"P"` | Prefix for generated segment names, e.g. `"P"` → `"P0"`, `"P1"`, … |
| `fields` | `tuple` | `("segid",)` | Which field(s) to write: any combination of `"segid"` and `"chain"` |
| `protein_cutoff` | `float` | `2.0` | Max `C(i)–N(i+1)` distance (Å) for two protein residues to be continuous |
| `nucleic_cutoff` | `float` | `2.2` | Max `O3'(i)–P(i+1)` distance (Å) for two nucleic residues to be continuous |
| `ca_fallback_cutoff` | `float` | `5.0` | Max `CA–CA` distance (Å) used when a protein residue lacks `C`/`N` |
| `nucleic_fallback_cutoff` | `float` | `3.2` | Max `C3'–P` distance (Å) used when a nucleic residue lacks `O3'` |
| `single_other_segment` | `bool` | `False` | Put all non-polymer, non-water, non-ion molecules into one segment instead of one per molecule |

## Common variations

```python
# Assign segments to protein chains only
mol = autoSegment(mol, sel="protein")

# Write both chain and segid in one call
mol = autoSegment(mol, fields=("chain", "segid"))

# Lump every ligand/cofactor into a single "other" segment
mol = autoSegment(mol, single_other_segment=True)
```

## Gotchas

- {py:func}`~moleculekit.tools.autosegment.autoSegment` returns a new {py:class}`~moleculekit.molecule.Molecule`; it does not mutate the input.
- Only coordinates and atom names are needed — explicit bonds are not required (they are guessed only for the "other" bucket).
- `segid` can be up to 4 characters (MD force-field convention); `chain` is a single character (PDB convention).
- When writing to PDB, only the `chain` field is stored in the standard CHAIN column; `segid` goes into the SEGID column, which many programs ignore.
- `autoSegment2` is deprecated and forwards to {py:func}`~moleculekit.tools.autosegment.autoSegment` with a `DeprecationWarning`; use `autoSegment` directly.

## See also

- [How to append and merge molecules](append-and-merge-molecules.md)
- [How to filter and remove atoms](filter-and-remove.md)
