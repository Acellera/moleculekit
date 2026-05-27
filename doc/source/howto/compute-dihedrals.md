# How to compute dihedrals

## Goal

Calculate one or many dihedral angles from a trajectory or single structure.

## Minimal example

```python
import numpy as np
from moleculekit.molecule import Molecule

mol = Molecule("3PTB")

# Atom indices defining the psi backbone dihedral of residue 17:
# N(17) - CA(17) - C(17) - N(18)
psi17 = [
    mol.atomselect("resid 17 and name N",  indexes=True)[0],
    mol.atomselect("resid 17 and name CA", indexes=True)[0],
    mol.atomselect("resid 17 and name C",  indexes=True)[0],
    mol.atomselect("resid 18 and name N",  indexes=True)[0],
]

# Read the current value (returns radians)
print(f"psi(17) before: {np.degrees(mol.getDihedral(psi17)):.1f}°")

# Set it to an extended-backbone value (180°) and read it back to confirm
mol.setDihedral(psi17, np.pi)
print(f"psi(17) after : {np.degrees(mol.getDihedral(psi17)):.1f}°")
```

## Parameters that matter

| Method / argument | What it does |
|---|---|
| `mol.getDihedral(atom_quad)` | Returns the dihedral angle in **radians** for the four atom indices in `atom_quad`. Operates on the current `mol.frame`. |
| `mol.setDihedral(atom_quad, radians, bonds=None, guessBonds=False)` | Rotates the downstream half of the molecule around the central bond of the dihedral so the angle becomes `radians`. For a chain of modifications pass `bonds=mol.bonds` (or a pre-built bond array) once to keep the bond table from being re-built on every call. |

## Common variations

```python
# The low-level function — useful when you have raw (4, 3) coords (e.g.
# from a numpy slice or a manually constructed array) and don't want to
# go through a Molecule
from moleculekit.dihedral import dihedralAngle

angle = dihedralAngle(mol.coords[psi17, :, 0])
```

```python
# Compute the same dihedral across every frame of a trajectory
angles = np.array([
    dihedralAngle(mol.coords[psi17, :, f])
    for f in range(mol.numFrames)
])
```

## Gotchas

- Both `getDihedral` and `dihedralAngle` return angles in **radians**; convert with `np.degrees(angle)` if you want degrees.
- `mol.getDihedral` operates on the current `mol.frame` — set `mol.frame = i` first if you want a specific frame.
- `mol.setDihedral` rotates the downstream side of the dihedral in place; the upstream side is held fixed. If the topology is ambiguous (the rotation would split the molecule wrong), the call may fail — guard against this by passing an explicit `bonds=mol.bonds` array.
- For computing many dihedrals at once across a trajectory, prefer {py:class}`~moleculekit.projections.metricdihedral.MetricDihedral` from `moleculekit.projections.metricdihedral` — it batches the work efficiently.

## See also

- [How to compute projections](compute-projections.md)
- [How to compute RMSD and RMSF](compute-rmsd-rmsf.md)
