# How to write a structure

## Goal

Save a `Molecule` to disk in a chosen file format.

## Which format to use

For a prepared / templated Molecule that carries bonds, bond orders, and formal charges, **prefer mmCIF (`.cif` or `.bcif`)** — it round-trips nearly all the data a `Molecule` holds. PDB is still fine for quick interchange with legacy tools, but you lose bond orders and the file is bound by fixed column widths. JSON is for application development where you need a lossless round-trip of the in-memory representation.

## Minimal example

```python
from moleculekit.molecule import Molecule

mol = Molecule("3PTB")
mol.write("output.cif")
```

## Parameters that matter

The signature is {py:meth}`~moleculekit.molecule.Molecule.write` with parameters `filename`, `sel=None`, `type=None`, and `**kwargs`. Format-specific options (e.g. `writebonds` for PDB) are passed via `**kwargs`.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `filename` | `str` | required | Output file path; the extension determines the format. |
| `sel` | `str`, bool mask, or index array | `None` (all) | Atom selection — only the selected atoms are written. |
| `type` | `str` | `None` | Explicitly override the format (e.g. `"pdb"`) when the extension is ambiguous. |

## Supported formats

| Extension | Format | Carries |
|---|---|---|
| `cif` | mmCIF / PDBx | **Recommended default.** Full topology incl. bonds, bond orders, formal charges, segid, occupancy, B-factors. |
| `bcif`, `bcif.gz` | Binary mmCIF | Compact binary variant of `cif`; same content, smaller and faster. |
| `pdb`, `pdb.gz` | Protein Data Bank | Coords + chain/resid/segid + formal charges (columns 79–80). Bond orders are **not** stored. |
| `pdbqt` | AutoDock PDB | PDB + partial charges. |
| `mmtf` | Macromolecular Transmission Format | Compact binary. |
| `mol2` | Tripos MOL2 | Bonds + bond orders + atom types. |
| `sdf` | Structure-Data File | Bond orders; small molecules. |
| `psf` | CHARMM / NAMD topology | Topology only (no coords). |
| `gro` | GROMACS structure | Coords + topology (single frame). |
| `xyz`, `xyz.gz` | Plain XYZ | Elements + coords only. |
| `xtc` | GROMACS compressed traj | Coordinates only, lossy float16. |
| `dcd` | CHARMM/NAMD binary traj | Coordinates only, full precision. |
| `trr` | GROMACS full-precision traj | Coords + optional velocities/forces. |
| `netcdf`, `nc`, `ncdf` | AMBER NetCDF | Coordinates only, full precision. |
| `binpos` | AMBER binpos | Coordinates only. |
| `xsc` | NAMD extended system | Box / restart info. |
| `coor`, `crd`, `inpcrd` | Coordinate files | Coordinates only — pair with a topology. |
| `h5`, `gro`, `mdcrd`, `lammpstrj`, `ncrst`, `rst7` | mdtraj-supported | Requires `mdtraj`. |
| `json` | moleculekit JSON | Lossless round-trip of the in-memory Molecule. Useful in app development; rarely the right choice for tutorials. |

For trajectory formats (`xtc`, `dcd`, `trr`, `netcdf`, `binpos`), only the coordinates are written — keep a topology file (PSF, PRMTOP, PDB) alongside.

## Common variations

```python
# Write a protein-only sub-selection as mmCIF (preserves bonds + charges)
mol.write("protein.cif", sel="protein")
```

```python
# Write a PDB for interchange with a tool that doesn't read mmCIF
mol.write("output.pdb")
```

```python
# Round-trip a Molecule losslessly through JSON (e.g. for app state)
mol.write("mol.json")
roundtrip = Molecule("mol.json")
```

## Gotchas

- PDB **cannot store explicit bond orders** (single/double/aromatic) — use mmCIF, MOL2, or SDF if those are needed downstream. PDB *can* store formal charges in columns 79–80, but many third-party PDB parsers ignore them.
- The `chain` field is written as a single character in PDB; `segid` (up to 4 characters) survives a PDB round-trip only in the SEGID column, which many programs ignore. mmCIF preserves both faithfully.

## See also

- [How to read a structure](read-a-structure.md)
- [How to filter and remove atoms](filter-and-remove.md)
