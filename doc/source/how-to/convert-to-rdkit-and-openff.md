# How to convert a Molecule to RDKit or OpenFF

## Goal

Hand a {py:class}`~moleculekit.molecule.Molecule` off to RDKit (for cheminformatics) or to the OpenFF Toolkit (for force-field assignment, charge calculation, parameterization), and round-trip the result back if needed.

## Which conversion to use

| You want… | Call |
|---|---|
| An `rdkit.Chem.Mol` for substructure search, descriptors, fingerprints, SMILES generation | {py:meth}`~moleculekit.molecule.Molecule.toRDKitMol` |
| An `openff.toolkit.topology.Molecule` for OpenFF parameterization, partial-charge calculation, force-field assignment | {py:meth}`~moleculekit.molecule.Molecule.toOpenFFMolecule` |
| Round-trip back from RDKit to a moleculekit Molecule | {py:meth}`~moleculekit.molecule.Molecule.fromRDKitMol` |

The OpenFF conversion goes through RDKit internally, so a healthy RDKit conversion is the prerequisite for a healthy OpenFF conversion.

## Minimal example

```python
from moleculekit.molecule import Molecule
from moleculekit.tools.preparation import systemPrepare

mol = Molecule("3PTB")

# Template the ligand from RCSB SMILES so it has correct bond orders and
# formal charges before conversion.
mol.templateResidueFromSmiles(
    mol.resname == "BEN",
    smiles="NC(=N)c1ccccc1",
    addHs=True,
)

# Copy the ligand into a standalone Molecule for cheminformatics work.
lig = mol.copy(sel="resname BEN")
rdmol = lig.toRDKitMol(sanitize=True)
```

`rdmol` is now a fully-fledged `rdkit.Chem.Mol` you can pass into any RDKit pipeline:

```python
from rdkit import Chem
print(Chem.MolToSmiles(rdmol))   # canonical SMILES
print(Chem.Descriptors.MolWt(rdmol))
```

## Parameters that matter

### `Molecule.toRDKitMol`

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `sanitize` | `bool` | `False` | Run RDKit's sanitization (valence, aromaticity, kekulization). Required for most downstream RDKit operations. |
| `kekulize` | `bool` | `False` | Force Kekulé bond perception (alternating single/double) instead of aromatic flags. |
| `assignStereo` | `bool` | `True` | Assign stereochemistry from 3D coordinates. |
| `guessBonds` | `bool` | `False` | If `mol.bonds` is empty, run a distance-based guesser before converting. |

### `Molecule.toOpenFFMolecule`

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `sanitize` | `bool` | `False` | Forwarded to the internal `toRDKitMol` call. |
| `kekulize` | `bool` | `False` | Forwarded to the internal `toRDKitMol` call. |
| `assignStereo` | `bool` | `True` | Forwarded to the internal `toRDKitMol` call. |

Per-atom `mol.charge` values are copied onto `offmol.partial_charges` automatically. Residue / chain / insertion identity is carried through `offmol.atoms[i].metadata` so the OpenFF `Topology` hierarchy schemes can reconstruct the residue structure.

## Common variations

```python
# OpenFF Molecule for parameter assignment
offmol = lig.toOpenFFMolecule(sanitize=True)

# Assign GAFF parameters via OpenFF
from openff.toolkit.typing.engines.smirnoff import ForceField
ff = ForceField("openff-2.1.0.offxml")
system = ff.create_openmm_system(offmol.to_topology())
```

```python
# Build a Molecule from an RDKit Mol (e.g. a SMILES + embedded conformer)
from rdkit import Chem
from rdkit.Chem import AllChem

rdmol = Chem.MolFromSmiles("CCO")
rdmol = Chem.AddHs(rdmol)
AllChem.EmbedMolecule(rdmol)
AllChem.MMFFOptimizeMolecule(rdmol)
mol = Molecule.fromRDKitMol(rdmol)
```

```python
# Convert an entire protein–ligand complex (not just the ligand)
rdmol = mol.toRDKitMol(sanitize=False)   # sanitize=False for non-canonical residues
```

## Gotchas

- Conversion needs explicit bonds. If `mol.bonds` is empty (plain PDB load), run {py:meth}`~moleculekit.molecule.Molecule.templateResidueFromSmiles` for non-canonical residues so the conversion gets correct bond orders, or pass `guessBonds=True` for a distance-based fallback.
- `sanitize=True` will raise on residues whose bonding or formal charges are inconsistent with chemical rules. For a freshly read PDB this often happens around metal centers and covalent ligands; template those residues first (see [Custom residues from SMILES](../tutorials/system-prep/03-custom-residues-from-smiles.md)).
- Hydrogens must be present for stereochemistry / valence checks. Run {py:func}`~moleculekit.tools.preparation.systemPrepare` (or set explicit Hs via `templateResidueFromSmiles(..., addHs=True)`) before conversion.
- `toOpenFFMolecule` requires `openff-toolkit` and `openff-units` installed.

## See also

- [How to compute protein–ligand interactions](compute-protein-ligand-interactions.md)
- [Custom residues from SMILES](../tutorials/system-prep/03-custom-residues-from-smiles.md)
- [Molecule data model](../explanation/molecule-data-model.md)
