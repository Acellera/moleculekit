# How to align structures

## Goal

Superimpose one structure or trajectory onto a reference by minimising RMSD over a chosen atom selection.

## Minimal example

```python
from moleculekit.molecule import Molecule
from moleculekit.util import uniformRandomRotation, molRMSD

ref = Molecule("3PTB")
mol = ref.copy()

# Apply a uniform random rotation around the centroid of mol
centroid = mol.coords[:, :, 0].mean(axis=0)
mol.rotateBy(uniformRandomRotation(), center=centroid)

ca = mol.atomselect("protein and name CA", indexes=True)
ca_ref = ref.atomselect("protein and name CA", indexes=True)
print(f"RMSD before align: {molRMSD(mol, ref, ca, ca_ref):.2f} Å")

mol.align("protein and name CA", refmol=ref)
print(f"RMSD after align:  {molRMSD(mol, ref, ca, ca_ref):.2f} Å")
```

The RMSD drops from several Ångström (random rotation) to essentially zero because the two structures share the same coordinates; alignment can recover the original superposition exactly.

## Parameters that matter

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `sel` | `str` or `np.ndarray` | required | Atoms in `mol` used for the alignment |
| `refmol` | {py:class}`~moleculekit.molecule.Molecule` | `None` (self) | Reference molecule; if `None`, align to `mol`'s first frame |
| `refsel` | `str` or `np.ndarray` | same as `sel` | Atoms in `refmol` to align onto; must have the same count as `sel` |
| `frames` | `list` or `range` | all frames | Which frames of `mol` to align |
| `mode` | `str` | `"index"` | Atom-correspondence rule. `"index"` pairs `sel` and `refsel` in increasing atom-index order (the two selections must yield the same atom count). `"structure"` uses TM-align internally to find the best structural correspondence and is robust to mismatched sequences. |

## Common variations

```python
# Structural alignment via TM-align — robust to mismatched sequences,
# does not require sel and refsel to have the same atom count
mol.align("protein", refmol=ref, mode="structure")
```

```python
# Sequence-based alignment — handles mismatched residue numbering
# by first aligning sequences and then calling .align on matched residues
mol.alignBySequence(ref)
```

## Gotchas

- With `mode="index"` (the default), {py:meth}`~moleculekit.molecule.Molecule.align` requires the same number of atoms in `sel` and `refsel`; a mismatch raises an error.
- With `mode="structure"`, the correspondence is found by TM-align internally — `sel` and `refsel` can have different atom counts and the routine is robust to large conformational differences.
- {py:meth}`~moleculekit.molecule.Molecule.alignBySequence` handles residue count mismatches by first finding a sequence alignment, then calling {py:meth}`~moleculekit.molecule.Molecule.align` (`mode="index"`) on the matched residues.
- If `refmol` is `None`, `mol` is aligned to its own first frame, which removes rigid-body drift across frames.

## See also

- [How to compute RMSD and RMSF](compute-rmsd-rmsf.md)
- [How to wrap trajectories](wrap-trajectories.md)
