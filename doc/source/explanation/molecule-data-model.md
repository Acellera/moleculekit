# The Molecule data model

A {py:class}`~moleculekit.molecule.Molecule` in moleculekit is more than a container for atom coordinates. It
holds the full **topology** of a molecular system — every per-atom field, the
bond graph, periodic box information, and provenance metadata — together with
any number of **trajectory frames**. Understanding the data layout lets you
write efficient code, avoid pitfalls with in-place mutation, and trace data
back to its source files.

## Per-atom arrays

Every atom in a `Molecule` is described by a set of parallel NumPy arrays, all
of length `mol.numAtoms`. The most commonly used fields are:

| Field | dtype | Typical content |
|---|---|---|
| `name` | `object` | Atom name, e.g. `"CA"`, `"OXT"` |
| `resname` | `object` | Residue name, e.g. `"ALA"`, `"HOH"` |
| `resid` | `int` | Residue sequence number (PDB column 23–26) |
| `insertion` | `object` | Insertion code (PDB column 27), often `""` |
| `chain` | `object` | Chain identifier (PDB column 22), e.g. `"A"` |
| `segid` | `object` | Segment identifier (MD convention), e.g. `"P0"` |
| `element` | `object` | Element symbol, e.g. `"C"`, `"N"`, `"O"` |
| `occupancy` | `float32` | PDB occupancy, default 1.0 |
| `beta` | `float32` | PDB B-factor / temperature factor |
| `charge` | `float32` | Partial charge (populated by force-field tools) |
| `masses` | `float32` | Atomic mass in Da |
| `record` | `object` | PDB record type: `"ATOM"` or `"HETATM"` |
| `formalcharge` | `int32` | Integer formal charge (populated by {py:func}`~moleculekit.tools.preparation.systemPrepare`, {py:meth}`~moleculekit.molecule.Molecule.templateResidueFromSmiles`, and {py:meth}`~moleculekit.molecule.Molecule.templateResidueFromMolecule`) |
| `atomtype` | `object` | Force-field atom type (populated downstream) |

Because these arrays are NumPy arrays you can inspect, compare, and manipulate
them directly:

```python
from moleculekit.molecule import Molecule

mol = Molecule("3ptb")
print(mol.name.dtype)       # object
print(mol.resid.dtype)      # int64
print(mol.occupancy.dtype)  # float32

# Vectorised field access — no loop needed
ca_mask = mol.name == "CA"
print(mol.resid[ca_mask])   # resids of all alpha-carbons
```

There is no hidden indirection: `mol.name[i]` is always the atom-name string
for atom `i`, and `mol.resid[i]` is always its residue number.

## Coordinates: `coords`

Atomic positions are stored in a single array of shape
`(numAtoms, 3, numFrames)`, dtype `float32`, in units of Ångström:

```python
mol.coords.shape   # (numAtoms, 3, numFrames)
mol.numAtoms       # first axis
mol.numFrames      # third axis
```

Indexing patterns you will use often:

```python
# First frame, all atoms — shape (numAtoms, 3)
frame0 = mol.coords[:, :, 0]

# All frames for atom 42 — shape (3, numFrames)
traj_atom42 = mol.coords[42, :, :]

# x-coordinates of all atoms in the last frame
x_last = mol.coords[:, 0, -1]
```

For a single-frame structure `mol.numFrames == 1` and `mol.coords[:, :, 0]`
gives the familiar `(numAtoms, 3)` coordinate matrix.

## Bonds: `bonds` and `bondtype`

Bonding information is stored separately from atom fields:

- `mol.bonds` — `uint32` array of shape `(numBonds, 2)`, where each row is a
  pair of atom indices `[i, j]`.
- `mol.bondtype` — `object` array of shape `(numBonds,)`, holding a bond-order
  or bond-type string (e.g. `"1"`, `"2"`, `"ar"`, `"un"`, `"mc"` for metal
  coordination) parallel to `mol.bonds`.

The number of bonds is `mol.bonds.shape[0]`. When a file is read without
explicit bond information (e.g. a plain PDB with no `CONECT` records) the
array is empty; call {py:meth}`~moleculekit.molecule.Molecule.guessBonds`
to populate it. The method updates `mol.bonds` and `mol.bondtype` together
so the two parallel arrays stay consistent.

```python
mol = Molecule("plain.pdb")   # a PDB with no CONECT records
print(mol.bonds.shape[0])     # 0 — no connectivity loaded
mol.guessBonds()
print(mol.bonds.shape[0])     # now populated
print(mol.bondtype.shape[0])  # same length, kept in lockstep
print(mol.bonds[:5])          # first 5 bonds as index pairs
```

Avoid the form `mol.bonds = guess_bonds(mol)`: it updates only `bonds`
and leaves `bondtype` at its old length, so the two arrays drift out of
sync.

Bonds are atom-index pairs, not residue identifiers. If atoms are reordered or
filtered out, index pairs must be regenerated.

## Periodic box: `box` and `boxangles`

For systems with periodic boundary conditions:

- `mol.box` — `float32` array of shape `(3, numFrames)`, holding the box
  **lengths** `[a, b, c]` in Ångström for each frame.
- `mol.boxangles` — `float32` array of shape `(3, numFrames)`, holding the
  box **angles** `[α, β, γ]` in degrees.

Orthorhombic (rectangular) boxes have all angles equal to 90°. A non-periodic
structure has all-zero box arrays.

```python
mol.box[:, 0]           # box lengths of the first frame
mol.boxangles[:, 0]     # box angles of the first frame
# Orthorhombic check for frame 0:
all_ortho = (mol.boxangles[:, 0] == 90).all()
```

## Units

Moleculekit uses a single distance unit throughout: **Ångström (Å)**. This applies to:

- `mol.coords` — atomic positions.
- `mol.box` — periodic-box lengths.
- All readers and writers (regardless of the source format's native units — GROMACS' `.gro` / `.xtc` use nanometres on disk; moleculekit converts to Å on load and converts back on write).
- All distance parameters in the library (`coldist`, `autoSegment`'s `protein_cutoff`, `find_clashes` thresholds, `within X of` selections, etc.).

Angles — `mol.boxangles`, dihedrals returned by {py:meth}`~moleculekit.molecule.Molecule.getDihedral`, and rotation angles passed to {py:meth}`~moleculekit.molecule.Molecule.setDihedral` — are in **radians** for the function APIs, except `mol.boxangles` which is in degrees (matching the PDB convention).

## Provenance: `fileloc`

`mol.fileloc` is a Python list of `[filename, frame_index]` pairs, one per
trajectory frame. It records which file each frame came from and its position
within that file:

```python
# After loading a trajectory
mol.fileloc[0]   # e.g. ['/data/run1/traj.xtc', 0]
mol.fileloc[-1]  # last frame's provenance
```

This is especially useful when you concatenate multiple trajectory files: you
can trace any frame back to its exact source.

## Mutation semantics

Many {py:class}`~moleculekit.molecule.Molecule` methods mutate the object **in place**:

- `mol.filter(sel)` — keeps only selected atoms; modifies `mol` directly.
- `mol.remove(sel)` — removes selected atoms; modifies `mol` directly.
- `mol.set(field, value, sel=...)` — assigns values to a field; modifies
  `mol` directly.
- `mol.wrap(...)` — re-images coordinates into the periodic box; modifies
  `mol` directly.
- `mol.align(sel, refmol)` — superposes frames; modifies `mol` directly.

If you need to preserve the original molecule, call `mol.copy()` first:

```python
mol_orig = Molecule("3ptb")

# Wrong — mol_orig is now the filtered result
mol_orig.filter("protein")

# Correct — keep the original, work on a copy
mol_orig = Molecule("3ptb")
mol_protein = mol_orig.copy()
mol_protein.filter("protein")
```

{py:meth}`~moleculekit.molecule.Molecule.copy` with a `sel=` argument is a convenient shorthand that creates a new {py:class}`~moleculekit.molecule.Molecule`
containing only the selected atoms:

```python
mol_lig = mol.copy(sel="resname BEN")
```

## Topology versus trajectory layering

A `Molecule` is built in two conceptual layers:

1. **Topology layer** — the per-atom arrays (`name`, `resname`, `resid`,
   `chain`, `segid`, `bonds`, ...) that describe what atoms exist and how
   they are connected. `numAtoms` is determined here. Typically read from a
   PDB, PSF, PRMTOP, or similar topology file.

2. **Trajectory layer** — the `coords`, `box`, `boxangles`, `fileloc`,
   `step`, and `time` arrays that carry per-frame data. `numFrames` is
   determined here. Loaded from XTC, DCD, TRR, NetCDF, or similar trajectory
   files.

The standard MD workflow reads topology first, then appends one or more
trajectory files:

```python
mol = Molecule("system.prmtop")   # topology only, numFrames == 0
mol.read("run1.xtc")              # appends frames
mol.read("run2.xtc", append=True) # appends more frames
print(mol.numFrames)              # total frames from both trajectories
```

Topology is stable across trajectory reads. Only frame-indexed arrays change
when trajectories are appended. You cannot load a trajectory whose atom count
differs from the topology — moleculekit will raise an error.

## Further reading

- How-to: [Read a structure](../how-to/read-a-structure.md)
- How-to: [Set and get properties](../how-to/set-and-get-properties.md)
- How-to: [Filter and remove atoms](../how-to/filter-and-remove.md)
