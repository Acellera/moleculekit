# How to read a trajectory

## Goal

Attach trajectory frames to a topology `Molecule` so that all frames are available for analysis.

## Minimal example

```python
from moleculekit.molecule import Molecule

# Load topology first, then attach the trajectory
mol = Molecule("topology.psf")
mol.read("trajectory.xtc")
print(mol.numFrames)
```

## Parameters that matter

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `frames` | `list` | `None` (all) | Read only the listed frame indices from the trajectory |
| `skip` | `int` | `None` | Skip every N frames (e.g. `skip=10` reads 1 in 10) |
| `append` | `bool` | `False` | Append frames to existing coordinates instead of replacing them |

## Common variations

```python
# Read every 10th frame (uniform stride)
mol = Molecule("topology.psf")
mol.read("trajectory.xtc", skip=10)
```

```python
# Read a single specific frame
mol = Molecule("topology.psf")
mol.read("trajectory.xtc", frames=[500])
```

```python
# Concatenate two trajectory files
mol = Molecule("topology.psf")
mol.read("run1.xtc")
mol.read("run2.xtc", append=True)
print(mol.numFrames)
```

## Gotchas

- All frames are loaded into memory at once; for very long trajectories use `skip` or `frames=` to read a subset.
- The atom count of the trajectory must exactly match the topology — a mismatch raises an error.
- `frames=` is **per file**: when reading a single trajectory it must be a length-1 list whose single element is either an int or a list of ints (e.g. `frames=[[0, 10, 20]]`). For uniform subsampling prefer `skip=`.
- `skip` strides the final merged coordinate array; it applies independently of `frames` and is the simplest way to subsample a long trajectory.
- When `append=False` (default) a second `mol.read(traj)` call replaces the existing coordinates rather than extending them.

## See also

- [How to read a structure](read-a-structure.md)
- [How to wrap trajectories](wrap-trajectories.md)
- [How to align structures](align-structures.md)
