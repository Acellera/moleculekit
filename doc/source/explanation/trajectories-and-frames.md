# Trajectories and frames

Molecular dynamics simulations produce long time series of atom positions.
Moleculekit represents this as a three-dimensional coordinate array stacked
onto a fixed topology, together with per-frame box and provenance metadata.
This page explains the internal layout, how multiple trajectories are loaded
and concatenated, what the periodic box fields mean, and how wrapping and
memory interact with long trajectories.

## The coordinate array

All atomic positions for all frames live in `mol.coords`, a `float32` NumPy
array of shape `(numAtoms, 3, numFrames)`:

- Axis 0 — atoms, indexed `0` to `mol.numAtoms - 1`.
- Axis 1 — Cartesian coordinates `[x, y, z]`, in Ångström.
- Axis 2 — frames, indexed `0` to `mol.numFrames - 1`.

Common slicing patterns:

```python
mol.coords.shape              # (numAtoms, 3, numFrames)
mol.coords[:, :, 0]          # all atoms, frame 0 — shape (numAtoms, 3)
mol.coords[42, :, :]         # all frames for atom 42 — shape (3, numFrames)
mol.coords[:, 0, :]          # x-coords of all atoms, all frames — (numAtoms, numFrames)
mol.coords[:, :, -1]         # last frame
```

For a single-frame structure `mol.numFrames == 1`. The slicing `[:, :, 0]`
always retrieves a familiar `(numAtoms, 3)` matrix.

`mol.frame` tracks the **current frame index** used by distance-based
selections (`within`, `exwithin`) and the viewer. Change it with
`mol.frame = i`.

## Loading a trajectory

### Single trajectory

```python
from moleculekit.molecule import Molecule

mol = Molecule("system.prmtop")  # topology, numFrames == 0
mol.read("run1.xtc")             # fills coords, box, boxangles, fileloc
print(mol.numFrames)
```

### Selective frame reading

The `read` method accepts `frames` and `skip` arguments to load a subset
without reading the whole file into memory:

```python
# Load every 10th frame (uniform stride)
mol.read("run1.xtc", skip=10)

# Load only frame 500
mol.read("run1.xtc", frames=[500])

# Load a specific list of frame indices
mol.read("run1.xtc", frames=[0, 10, 20, 30, 40])
```

`skip=N` applies uniform subsampling. `frames=` selects specific indices and
is **per file**: when reading multiple trajectories in one call (e.g.
`Molecule(["topo.psf", "run1.xtc", "run2.xtc"], frames=[...])`) it must have
one entry per trajectory, each entry being either an int (single frame from
that file) or a list of ints. For the common single-trajectory case shown
here, a flat list of indices is accepted as-is.

### Multi-trajectory loads

Pass topology and trajectories as a single list to the `Molecule` constructor,
or call `read(..., append=True)` repeatedly. Frames are concatenated in order:

```python
mol = Molecule(["system.prmtop", "run1.xtc", "run2.xtc", "run3.xtc"])
# Equivalent to:
mol = Molecule("system.prmtop")
mol.read("run1.xtc")
mol.read("run2.xtc", append=True)
mol.read("run3.xtc", append=True)
```

The second positional argument of {py:class}`~moleculekit.molecule.Molecule`
is `name=`, **not** a trajectory file — passing trajectories there assigns
them to `mol.name` and loads nothing. Always use a single list (as above) or
the explicit `read(..., append=True)` form.

`mol.numFrames` is the total number of frames from all trajectories. Frame
ordering mirrors the order the files were passed.

## Provenance: `fileloc`

`mol.fileloc` is a Python list of `[filename, frame_index]` pairs, one entry
per frame. It records which file each frame came from and what index it had
within that file. This is invaluable when debugging multi-trajectory analyses:

```python
mol.fileloc[0]    # ['run1.xtc', 0]
mol.fileloc[999]  # ['run1.xtc', 999] or ['run2.xtc', 0] after the first file ends
```

## Periodic box: `box` and `boxangles`

Periodic MD simulations carry box geometry with every frame:

- `mol.box` — `float32`, shape `(3, numFrames)`. Box **lengths** `[a, b, c]`
  in Ångström.
- `mol.boxangles` — `float32`, shape `(3, numFrames)`. Box **angles**
  `[α, β, γ]` in degrees.

For a rectangular (orthorhombic) box all three angles equal 90°:

```python
# Box lengths for the first frame
mol.box[:, 0]        # e.g. array([80., 80., 100.])

# Angles for the first frame
mol.boxangles[:, 0]  # array([90., 90., 90.]) for orthorhombic

# Full-precision box vectors (3×3) are available via mol.boxvectors
mol.boxvectors[:, :, 0]   # shape (3, 3) for frame 0
```

Non-periodic structures have zero box arrays.

## Trajectory format characteristics

Moleculekit reads the most common MD trajectory formats. They differ in
precision, what they store, and typical file size:

| Format | Extension | Precision | Box | Velocities | Notes |
|---|---|---|---|---|---|
| XTC | `.xtc` | Reduced (lossy XDR fixed-point, ~0.01 Å) | Yes | No | GROMACS native; very compact |
| DCD | `.dcd` | Full `float32` | Varies | No | NAMD/CHARMM native |
| TRR | `.trr` | Full (file stores up to `float64`; moleculekit holds `coords` as `float32`) | Yes | Yes | GROMACS native; larger files |
| NetCDF | `.nc`, `.ncdf` | Full `float32` | Yes | Optional | AMBER native |

XTC is the most common format encountered in practice. Its compressed storage
means very small files but **slight coordinate rounding**; for high-precision
energy analysis prefer TRR or NetCDF.

DCD files do not always include box information; the box is sometimes stored in
a separate file or in the topology. Pair a DCD trajectory with a topology that
carries the box, or set `mol.box` manually.

## Wrapping

During an MD simulation, molecules are free to drift across periodic boundaries.
The raw trajectory coordinates may then show a molecule "split" across two
sides of the box, which breaks visual inspection and distance calculations.

**Wrapping** re-images all atoms so that each bonded group (molecule or
residue) appears inside the central periodic image:

```python
mol.wrap("protein")   # center wrapping around the protein
```

The argument is an atom selection that defines the **center of the wrapping
box**. Passing `"protein"` puts the protein in the middle, so solvent molecules
wrap around it rather than the protein wrapping around them.

Practical guidance:
- Wrap **after** loading a full trajectory, not before.
- Wrapping requires correct bonds (`mol.bonds`). Use a topology file that
  provides bonds, or call {py:func}`~moleculekit.bondguesser.guess_bonds` first.
- For trajectory analysis with {py:class}`~moleculekit.projections.metricdistance.MetricDistance` or `MetricCoordinate`,
  wrapping is usually needed before computing distances that span the periodic
  boundary.

## Memory considerations

All frames are loaded into RAM simultaneously. For a typical system:

- 50 000 atoms × 3 coordinates × `float32` (4 bytes) = 600 KB per frame.
- 10 000 frames × 600 KB ≈ 6 GB.

For trajectories that do not fit in RAM, use `skip` or `frames=` to load a
representative subset, or process the trajectory in chunks with separate `read`
calls:

```python
# Load every 100th frame to reduce memory usage by 100×
mol.read("long_run.xtc", skip=100)
```

## Further reading

- How-to: [Read a trajectory](../howto/read-a-trajectory.md)
- How-to: [Wrap trajectories](../howto/wrap-trajectories.md)
