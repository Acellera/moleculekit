# Small Molecule (Sub)Package

Package for efficient handling of small molecules.


## Voxelization 

example:

```python
from moleculekit.smallmol.smallmol import SmallMol, SmallMolStack
ms = SmallMolStack("/path/to/my/file/x.sdf")  # Generate Molecule stack 
my_gen = ms.voxel_generator(batch_size=32, n_jobs=10)  # Make a Voxel generator
for voxellist in my_gen:  # Iterate over the elements
    pass  # Do something with it.
```

## Tautomer generation
 
 
Generate and filter tautomers:
```python
from moleculekit.smallmol.smallmol import SmallMol

# input smile string / .mol2 file / rdkit molecule
my_mol = SmallMol("/path/to/some/file/mf.mol2")
tautomers, scores = my_mol.getTautomers(returnScores=True)

# Return filtered list of SmallMol objects
tautomers, scores = my_mol.getTautomers(returnScores=True, filterTauts=2)
```