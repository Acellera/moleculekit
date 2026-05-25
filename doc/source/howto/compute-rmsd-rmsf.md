# How to compute RMSD and RMSF

## Goal

Measure structural deviation across frames (RMSD) or per-atom positional fluctuation over a trajectory (RMSF).

## Minimal example

```python
from moleculekit.molecule import Molecule
from moleculekit.util import molRMSD

mol = Molecule("3PTB")
ref = mol.copy()

# Select Cα indices for both molecule and reference
ca_idx = mol.atomselect("protein and name CA", indexes=True)

# RMSD of all frames against the reference (returns one value per frame)
rmsd = molRMSD(mol, ref, ca_idx, ca_idx)
print(rmsd)
```

## Parameters that matter

| Parameter | Type | What it does |
|---|---|---|
| `mol` | {py:class}`~moleculekit.molecule.Molecule` | Trajectory to analyse |
| `refmol` | {py:class}`~moleculekit.molecule.Molecule` | Reference structure |
| `rmsdsel1` | integer `np.ndarray` | Atom indices in `mol` |
| `rmsdsel2` | integer `np.ndarray` | Matching atom indices in `refmol` |

## Common variations

```python
# RMSF per Cα atom over a trajectory
from moleculekit.projections.metricfluctuation import MetricFluctuation

rmsf = MetricFluctuation("protein and name CA").project(mol)
# rmsf shape: (n_frames, n_CA); for a single per-atom value over the trajectory use rmsf.mean(axis=0)
```

```python
# Align before computing RMSD to separate rotation from internal motion
mol.align("protein and name CA", refmol=ref)
rmsd_aligned = molRMSD(mol, ref, ca_idx, ca_idx)
```

## Gotchas

- {py:func}`~moleculekit.util.molRMSD` computes raw (non-mass-weighted) RMSD; align the structures first if you want to remove rigid-body motion.
- RMSF is per-atom and depends on the choice of reference (mean structure); the result is sensitive to conformational heterogeneity.
- Both `mol` and `refmol` must have the same number of selected atoms.
- {py:class}`~moleculekit.projections.metricfluctuation.MetricFluctuation` returns an `(n_frames, n_atoms)` array — use `rmsf.mean(axis=0)` to get a per-atom fluctuation averaged over all frames.

## See also

- [How to align structures](align-structures.md)
- [How to compute projections](compute-projections.md)
