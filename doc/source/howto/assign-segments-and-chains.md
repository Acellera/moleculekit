# How to assign segments and chains

## Goal

Derive `segid` and/or `chain` fields for a structure that lacks them, using gap detection to split continuous segments automatically.

## Minimal example

```python
from moleculekit.molecule import Molecule
from moleculekit.tools.autosegment import autoSegment

mol = Molecule("3PTB")
mol = autoSegment(mol)
print(set(mol.segid))
```

## Parameters that matter

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `mol` | {py:class}`~moleculekit.molecule.Molecule` | required | Input molecule (a copy is returned; original is unchanged) |
| `sel` | `str` | `"all"` | Restrict gap detection to this atom selection |
| `basename` | `str` | `"P"` | Prefix for generated segment names, e.g. `"P"` → `"P0"`, `"P1"`, … |
| `spatial` | `bool` | `True` | Treat a residue-numbering gap as a real gap only if Cα distance > `spatialgap` Å |
| `spatialgap` | `float` | `4.0` | Distance threshold in Å for spatial gap detection |

## Common variations

```python
# Assign segments to protein chains only
mol = autoSegment(mol, sel="protein")
```

## Gotchas

- {py:func}`~moleculekit.tools.autosegment.autoSegment` returns a new {py:class}`~moleculekit.molecule.Molecule`; it does not mutate the input.
- `segid` can be up to 4 characters (MD force-field convention); `chain` is a single character (PDB convention).
- Auto-assignment is topology-driven and can fail on structures with non-contiguous or missing residue numbers — inspect the result before use.
- When writing to PDB, only the `chain` field is stored in the standard CHAIN column; `segid` goes into the SEGID column, which many programs ignore.

## See also

- [How to append and merge molecules](append-and-merge-molecules.md)
- [How to filter and remove atoms](filter-and-remove.md)
