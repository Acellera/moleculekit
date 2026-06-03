# The atom-selection language

Moleculekit ships a VMD-inspired atom-selection language that lets you describe
subsets of atoms in a {py:class}`~moleculekit.molecule.Molecule` using a concise, readable syntax. The same
selection string is accepted wherever an atom selection is expected — by
{py:meth}`~moleculekit.molecule.Molecule.atomselect`, {py:meth}`~moleculekit.molecule.Molecule.filter`, {py:meth}`~moleculekit.molecule.Molecule.remove`, {py:meth}`~moleculekit.molecule.Molecule.copy`, {py:meth}`~moleculekit.molecule.Molecule.set`, {py:meth}`~moleculekit.molecule.Molecule.wrap`, {py:meth}`~moleculekit.molecule.Molecule.align`, and every
other method that takes a `sel` argument.

## What a selection produces

Every selection evaluates to a **boolean mask** — a NumPy array of `bool` with
length `mol.numAtoms`, where `True` marks selected atoms. You can also ask for
an array of integer indices instead:

```python
from moleculekit.molecule import Molecule

mol = Molecule("3ptb")

# Boolean mask (default)
mask = mol.atomselect("protein and backbone")
print(mask.dtype, mask.shape)   # bool (numAtoms,)

# Integer indices
idx = mol.atomselect("resname BEN", indexes=True)
print(idx)   # array of uint32 atom indices
```

The mask can be used everywhere a string is accepted — pass it directly to
`filter`, `copy`, etc. to skip re-parsing (faster when reusing the same
selection many times):

```python
prot_mask = mol.atomselect("protein")
mol_prot = mol.copy(sel=prot_mask)  # reuses precomputed mask
```

## Keyword selections

The following keywords select entire chemical classes based on residue-name
lookups and element checks:

| Keyword | What it selects |
|---|---|
| `protein` | All protein residues (canonical amino acids) |
| `nucleic` | All nucleic acid residues (DNA and RNA) |
| `water` | Water molecules (`HOH`, `WAT`, `TIP3`, ...) |
| `lipid` | Common lipid residues |
| `ion` | Common monatomic ions |
| `backbone` | Protein backbone atoms (`N`, `CA`, `C`, `O`) and nucleic backbone |
| `sidechain` | Protein sidechain atoms (non-backbone, non-hydrogen heavy atoms) |
| `hydrogen` | All atoms with `element == "H"` |
| `noh` | All non-hydrogen atoms |
| `all` | Every atom in the molecule |

## Per-atom field comparisons

You can test any per-atom field against a value or list of values:

```python
# Single value
mol.atomselect("resname ALA")
mol.atomselect("chain A")
mol.atomselect("element C")

# List of values (space-separated, no commas)
mol.atomselect("name CA N C O")
mol.atomselect("resname ALA GLY VAL")
mol.atomselect("chain A B")
```

Fields available for selection strings:

| Field | Description |
|---|---|
| `name` | Atom name |
| `resname` | Residue name |
| `resid` | Residue sequence number |
| `residue` | Zero-based internal residue index (contiguous, ignores `resid`/insertion gaps) |
| `index` | Zero-based atom index |
| `serial` | One-based atom serial number (as stored in the file) |
| `chain` | Chain identifier |
| `segid` (or `segname`) | Segment identifier |
| `element` | Element symbol |
| `altloc` | Alternate location identifier |
| `occupancy` | Occupancy value |
| `beta` | B-factor |
| `charge` | Partial charge |
| `mass` | Atomic mass |
| `insertion` | Insertion code |
| `x`, `y`, `z` | Cartesian coordinates (Å) at the current frame |

Numeric fields can also be wrapped in the functions `abs`, `sqr`, and
`sqrt` (e.g. `abs(charge) > 0.5`, `sqrt(sqr(x) + sqr(y)) < 10`).

A dedicated `backbonetype` selector classifies atoms by backbone type:
`backbonetype proteinback`, `backbonetype nucleicback`, and
`backbonetype normal` (everything that is neither protein nor nucleic
backbone).

```{note}
Mass- and charge-based selections only work if those fields are
populated. A molecule freshly read from a PDB file has all masses (and
usually charges) set to zero, so `mass > 0` would match nothing until
masses are assigned.
```

## Comparison operators and ranges

Numeric fields support comparison operators and range syntax:

```python
# Comparisons
mol.atomselect("resid > 50")
mol.atomselect("occupancy >= 0.5")
mol.atomselect("beta < 20")

# Range (inclusive on both ends)
mol.atomselect("resid 40 to 60")
mol.atomselect("index 0 to 99")

# Negation: use `not`
mol.atomselect("not chain B")
```

The `!=` operator only applies inside the modulo form (e.g.
`resid % 2 != 0`); for plain field comparisons use `not`.

## Boolean composition

Combine selections with `and`, `or`, `not`, and parentheses:

```python
mol.atomselect("protein and chain A")
mol.atomselect("resname ALA or resname GLY")
mol.atomselect("not water")
mol.atomselect("(protein and backbone) or (resname BEN and not hydrogen)")
```

`not` binds tighter than `and`/`or`. Crucially, `and` and `or` have
**equal precedence** (they share one non-associative level), so a chain
of mixed `and`/`or` is grouped left-to-right rather than `and` binding
before `or`. For example:

```python
# Parses as: protein and (name CA or name CB)
mol.atomselect("protein and name CA or name CB")
```

This is **not** the C-like behaviour where `and` binds before `or`.
Because the grouping of mixed `and`/`or` is easy to misread, always use
explicit parentheses when combining the two operators:

```python
mol.atomselect("protein and (name CA or name CB)")  # clear intent
mol.atomselect("(protein and name CA) or name CB")  # the other grouping
```

## Distance operators

Distance-based selections are evaluated at the **current frame** (`mol.frame`):

```python
# All atoms within 5 Å of the ligand (including the ligand itself)
mol.atomselect("within 5 of resname BEN")

# All atoms within 5 Å of the ligand, excluding the ligand
mol.atomselect("exwithin 5 of resname BEN")
```

## `same … as` operators

Expand a selection to cover complete residues, chains, or bond-graph fragments:

```python
# All atoms in any residue that has at least one backbone atom within 5 Å
mol.atomselect("same residue as (backbone and within 5 of resname BEN)")

# All atoms in any chain that contains a titratable histidine
mol.atomselect("same chain as resname HID HIE HIP")

# All atoms in the same covalently bonded fragment as the ligand
mol.atomselect("same fragment as resname BEN")
```

`fragment` groups atoms by connected components of the bond graph. For this to
work correctly, `mol.bonds` must be populated (see
[Guess bonds](../how-to/guess-bonds.md)).

## Cheat-sheet

| Expression | Example | Meaning |
|---|---|---|
| keyword | `protein` | Predefined chemical class |
| `field value` | `resname ALA` | Field equals value |
| `field v1 v2 ...` | `name CA N C` | Field equals any of the values |
| `field A to B` | `resid 40 to 60` | Numeric range (inclusive) |
| `field op val` | `beta > 20` | Numeric comparison |
| `and`, `or`, `not` | `protein and chain A` | Boolean logic |
| `within N of sel` | `within 5 of resname LIG` | Distance from selection |
| `exwithin N of sel` | `exwithin 5 of resname LIG` | Distance, excluding selection |
| `same prop as sel` | `same residue as backbone` | Whole-residue/chain/fragment expansion |

## Mask and index substitution

Any method that accepts a selection string also accepts:

- A **boolean NumPy array** of length `mol.numAtoms` — passed through without
  parsing, ideal for reusing expensive selections.
- An **integer NumPy array** of atom indices — converted automatically.

```python
import numpy as np

# Precompute once, reuse many times
prot_mask = mol.atomselect("protein")

mol.copy(sel=prot_mask)
mol.filter(prot_mask)
mol.set("beta", 0, sel=prot_mask)
```

Note that precomputed masks and index arrays go stale if the number or order
of atoms changes (e.g. after `filter`, `remove`, or adding hydrogens). Always
recompute after such operations.

## What is not supported

- VMD's `index from < N` range variant for loading trajectory subsets is not
  exposed.
- Complex regex on atom names (VMD's `=~` regex operator) is not implemented.
- The `pbwithin` periodic-boundary-aware distance selection is not available;
  use `wrap` first if working with periodic systems.

## Further reading

- Tutorial: [Atom selection](../tutorials/02-atom-selection.md)
- How-to: [Select atoms](../how-to/select-atoms.md)
