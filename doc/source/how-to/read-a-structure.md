# How to read a structure

## Goal

Load a molecular structure into a {py:class}`~moleculekit.molecule.Molecule` object from a local file or directly from the RCSB PDB by accession ID.

## Minimal example

```python
from moleculekit.molecule import Molecule

# Fetch directly from RCSB
mol = Molecule("3PTB")

# Load from a local file
mol = Molecule("structure.pdb")
```

## Parameters that matter

The `Molecule` constructor signature is `Molecule(filename=None, name=None, **kwargs)`. When `filename` is given, the constructor forwards `**kwargs` to {py:meth}`~moleculekit.molecule.Molecule.read`, so any read-time options can be passed alongside the path.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `filename` | `str` | `None` | Path to the file (any supported format) or a 4-character RCSB PDB ID. If `None`, an empty Molecule is created and you can call `mol.read(...)` later. |
| `name` | `str` | `None` | Optional display label; does not affect atom data. |
| `type` | `str` | `None` | File-type override (`"pdb"`, `"mol2"`, `"prmtop"`, ...). Normally inferred from the extension. |
| `validateElements` | `bool` | `True` | Raises if element symbols are missing or unrecognised; set to `False` for custom atom types. |
| `keepaltloc` | `str` | `"A"` | Which alternate-location indicator to keep when reading PDB/mmCIF. |

## Supported formats

The reader picks a backend from the file extension. Trajectories are covered in [How to read a trajectory](read-a-trajectory.md); the formats below carry **topology** (atoms, residues, optional bonds) and in some cases a single frame of coordinates.

| Extension | Format | Notes |
|---|---|---|
| `pdb`, `ent` | Protein Data Bank | Most common; coords + optional `CONECT` bonds. |
| `pdb.gz` | gzipped PDB | Transparently decompressed. |
| `cif` | mmCIF / PDBx | Modern PDB replacement; full bond / `LINK` info. |
| `bcif` | Binary mmCIF | Compact binary mmCIF. |
| `mmtf` | Macromolecular Transmission Format | Compact binary; deprecated upstream but still readable. |
| `mol2` | Tripos MOL2 | Bonds + bond orders + atom types. |
| `sdf` | Structure-Data File | Small-molecule format with bond orders. |
| `mae` | Schrödinger Maestro | Bond orders + force-field atom types. |
| `prmtop`, `prm`, `parm7` | AMBER topology | Topology only; pair with a coordinate file. |
| `psf` | CHARMM / NAMD topology | Topology only. |
| `top` | GROMACS topology | Falls back to PRMTOP reader if not GROMACS-style. |
| `gro` | GROMACS structure | Via mdtraj. |
| `xyz` | Plain XYZ | Coords + elements only; no bonds. |
| `pdbqt` | AutoDock PDB | PDB with partial charges. |
| `gjf` | Gaussian input | Coords + elements. |
| `rtf` | CHARMM residue topology | Residue templates. |
| `prepi` | AMBER prepi | Residue templates. |
| `crd`, `coor`, `inpcrd` | Coordinate files | Coordinates only — load a topology first. |
| `xtc`, `dcd`, `trr`, `netcdf`/`nc`/`ncdf`, `binpos`, `xsc` | Trajectory | See [How to read a trajectory](read-a-trajectory.md). |
| `h5`, `lh5`, `hdf5`, `arc`, `hoomdxml` | mdtraj-supported | Requires the `mdtraj` package. |
| `json` | moleculekit JSON | Lossless round-trip of a Molecule. |
| `alphafold` | AlphaFold output | AlphaFold-style topology. |

Special case: a bare 4-character string with no extension (e.g. `Molecule("3PTB")`) is interpreted as an **RCSB PDB ID** and fetched over the network. Set the `LOCAL_PDB_REPO` environment variable to point at a local mirror to avoid repeated downloads.

## Common variations

```python
# Load an AMBER PRMTOP topology
mol = Molecule("topology.prmtop")

# Load a PSF topology, then read trajectory frames separately
mol = Molecule("topology.psf")
mol.read("trajectory.xtc")
```

```python
# Two-step read: construct empty, then load
mol = Molecule()
mol.read("structure.mol2")
```

## Gotchas

- PDB format has fixed column widths: residue names longer than 4 characters and atom names longer than 4 characters are silently truncated.
- For PDB IDs that are not in the RCSB, {py:class}`~moleculekit.molecule.Molecule` raises a download error — check the spelling and network connectivity.
- When `validateElements=True` (default) the reader raises on unknown element symbols; set `validateElements=False` if working with custom atom types.

## See also

- [How to write a structure](write-a-structure.md)
- [How to read a trajectory](read-a-trajectory.md)
- [How to fetch from RCSB and OPM](fetch-from-rcsb-and-opm.md)
