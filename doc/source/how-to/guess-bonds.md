# How to guess bonds

## Goal

Populate `mol.bonds` for a structure that was loaded without explicit connectivity information.

## Minimal example

```python
from moleculekit.molecule import Molecule

mol = Molecule("structure.pdb")
mol.guessBonds()
print(mol.bonds.shape)       # (N_bonds, 2)
print(mol.bondtype[:5])      # parallel bond-type array — also populated
```

## Parameters that matter

`mol.guessBonds()` takes no required arguments. Connectivity is inferred from atomic coordinates and covalent radii, and the call updates `mol.bonds` *and* `mol.bondtype` together so the two parallel arrays stay consistent. Bond orders are not inferred — every entry in `mol.bondtype` is `"un"` (unknown).

## Common variations

```python
# Check that the bond count looks reasonable before proceeding
mol.guessBonds()
assert mol.bonds.shape[0] > 0, "No bonds guessed — check coordinates and elements"
```

## Gotchas

- `mol.guessBonds()` does not infer bond orders; every entry in `mol.bondtype` is `"un"` (unknown). When bond orders matter (e.g. for {py:meth}`~moleculekit.molecule.Molecule.toRDKitMol` conversion or SMILES output), template the relevant residues from SMILES with {py:meth}`~moleculekit.molecule.Molecule.templateResidueFromSmiles` instead, or from a reference {py:class}`~moleculekit.molecule.Molecule` (e.g. a CIF carrying correct bond orders and formal charges) with {py:meth}`~moleculekit.molecule.Molecule.templateResidueFromMolecule`.
- Very close non-bonded atoms (e.g. crystal contacts, stacking interactions) can be incorrectly identified as bonded.
- **Do not assign `mol.bonds = guess_bonds(mol)` directly.** Always use `mol.guessBonds()`, which updates `mol.bonds` and `mol.bondtype` together. See [The Molecule data model: Bonds](../explanation/molecule-data-model.md#bonds-bonds-and-bondtype) for why the two arrays must stay in lockstep.

## See also

- [How to assign segments and chains](assign-segments-and-chains.md)
- [How to wrap trajectories](wrap-trajectories.md)
