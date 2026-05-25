---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Atom selection

**You will learn:** how to select atoms using VMD-style strings, how to compose selections with boolean operators, and how to bypass the parser entirely with NumPy array masks for both clarity and speed.

**Prerequisites:**
- The [First molecule](01-first-molecule.md) tutorial, or basic familiarity with {py:class}`~moleculekit.molecule.Molecule`.

## Setup

```{code-cell} python
from moleculekit.molecule import Molecule

mol = Molecule("3PTB")
```

`3PTB` is bovine trypsin: 1701 atoms covering the protein, crystallographic waters, a calcium ion, and the benzamidine ligand (`BEN`) in the active site.

## Step 1 — VMD-style selections

{py:meth}`~moleculekit.molecule.Molecule.atomselect` accepts a VMD-style selection string and returns a boolean NumPy array with one entry per atom.

Select all protein atoms:

```{code-cell} python
mask = mol.atomselect("protein")
mask.sum()
```

A boolean mask of length `mol.numAtoms` (1701); 1629 atoms belong to the protein chains in this structure.

Select only alpha carbons:

```{code-cell} python
mol.atomselect("name CA").sum()
```

224 alpha carbons — one per residue in the protein.

Select a residue-id range:

```{code-cell} python
mol.atomselect("resid 40 to 60").sum()
```

161 atoms spanning residues 40 through 60 (inclusive).

Select atoms within 5 Å of the benzamidine ligand:

```{code-cell} python
mol.atomselect("within 5 of resname BEN").sum()
```

65 atoms are within 5 Å of any benzamidine atom.

Expand the proximity selection to whole residues:

```{code-cell} python
mol.atomselect("same residue as within 5 of resname BEN").sum()
```

131 atoms — once partial residues touching the 5 Å shell are completed, more atoms are included.

## Step 2 — Boolean composition

Combine keywords with `and`, `or`, and `not` directly inside the selection string:

```{code-cell} python
mol.atomselect("protein and name CA").sum()
```

223 alpha carbons — one less than the raw `name CA` count because a single `CA` atom in this structure belongs to the calcium ion record, not a protein residue.

```{code-cell} python
mol.atomselect("(resid 40 to 60) and not water").sum()
```

161 atoms — in 3PTB none of the resid 40–60 atoms are water, so the count is unchanged here; parentheses are allowed for grouping.

## Step 3 — Indices instead of masks

Pass `indexes=True` to get an integer array of atom indices instead of a boolean mask:

```{code-cell} python
mol.atomselect("resname BEN", indexes=True)
```

The 9 benzamidine atom indices. This is the right shape when you want to index directly into `mol.coords` or similar per-atom arrays without a boolean intermediate.

## Recap

- VMD-style strings are convenient for ad-hoc selections: `mol.atomselect("protein and name CA")`.
- Boolean composition (`and`, `or`, `not`, parentheses) lets you build complex selections without leaving the string form.
- Pass `indexes=True` to get an integer index array instead of a boolean mask.

---

## For developers: bypass the parser with masks

Everything below is an optimisation for tight loops and library code; casual scripting almost never needs it. If you are not sure whether you need this, **stay on the string form** — it is shorter, harder to break, and always re-parsed against whichever Molecule a call operates on.

:::{warning}
**Masks and index arrays are tied to a specific Molecule snapshot.**

Both boolean masks and integer index arrays refer to specific atom indices in the Molecule they were computed against. They go stale the moment the underlying atom array changes — and there is no runtime check that flags this.

Two concrete failure modes:

1. **The structure changes between computing the mask and using it.** Operations like {py:meth}`~moleculekit.molecule.Molecule.filter`, {py:meth}`~moleculekit.molecule.Molecule.remove`, {py:meth}`~moleculekit.molecule.Molecule.append`, {py:meth}`~moleculekit.molecule.Molecule.insert`, and {py:meth}`~moleculekit.molecule.Molecule.mutateResidue` reshape the atom array. Any mask or index array you computed beforehand silently refers to the wrong atoms (or runs off the end). Recompute the selection after any such call.

2. **The mask was computed on a different Molecule.** Functions that take two molecules — most importantly {py:meth}`~moleculekit.molecule.Molecule.align`, which accepts `sel` for `mol` and `refsel` for `refmol` — require each selection to come from its own Molecule. Passing `mol.atomselect("name CA")` as `refsel` is wrong; the mask is sized for `mol`, not `refmol`. Use a string (`"name CA"`) for cross-Molecule calls, or compute each mask on the right Molecule (`refmol.atomselect("name CA")`).

The string form is always safe: it is re-parsed against whichever Molecule the call is operating on.
:::

### Build a mask without the parser

Per-atom attributes on `Molecule` are plain NumPy arrays, so you can construct selection masks with standard array operations:

```{code-cell} python
ben_mask = mol.resname == "BEN"
ben_mask.sum()
```

9 atoms — identical to `mol.atomselect("resname BEN")`, but the string parser is never called. Compose masks with NumPy bitwise operators:

```{code-cell} python
composite = (mol.chain == "A") & (mol.resid > 100)
composite.sum()
```

1076 atoms in chain A with resid above 100 — equivalent to `mol.atomselect("chain A and resid > 100")`.

### Why bother — the performance gap

```{code-cell} python
import timeit

t_str  = timeit.timeit(lambda: mol.atomselect("resname BEN"), number=1000)
t_mask = timeit.timeit(lambda: mol.resname == "BEN",          number=1000)

print(f"string path : {t_str:.3f} s for 1000 calls")
print(f"mask path   : {t_mask:.3f} s for 1000 calls")
```

The mask path is one vectorised NumPy comparison; the string path runs a full parser invocation per call. The gap matters in inner loops; it does not matter in interactive scripting.

### Pass masks (and index arrays) anywhere a string is accepted

Any selector argument that ultimately dispatches through {py:meth}`~moleculekit.molecule.Molecule.atomselect` accepts a string, a boolean mask of length `mol.numAtoms`, or a `numpy.ndarray` of integer indices interchangeably. The non-string branch simply uses the array as the mask directly after normalisation.

```{code-cell} python
ben_mol = mol.copy(sel=mol.resname == "BEN")
ben_mol.numAtoms
```

```{code-cell} python
protein_mask = mol.atomselect("protein")
non_protein  = mol.copy(sel=~protein_mask)
non_protein.numAtoms
```

```{code-cell} python
mol.name[mol.resname == "BEN"]
```

The trio (string / bool mask / index array) is accepted by:
- {py:meth}`~moleculekit.molecule.Molecule.remove`, {py:meth}`~moleculekit.molecule.Molecule.filter`, {py:meth}`~moleculekit.molecule.Molecule.get`, {py:meth}`~moleculekit.molecule.Molecule.set`, {py:meth}`~moleculekit.molecule.Molecule.copy`
- {py:meth}`~moleculekit.molecule.Molecule.templateResidueFromSmiles` (documented as `str or numpy.ndarray`)
- the `sel` parameter on `Metric*` projection classes
- each per-residue entry in {py:func}`~moleculekit.tools.preparation.systemPrepare`'s `no_opt`, `no_prot`, `no_titr`, and `force_protonation` lists — each entry is forwarded to {py:meth}`~moleculekit.molecule.Molecule.atomselect`, so a bool mask or index array works in place of a string (each entry must resolve to exactly one residue)

## Next

- [Select atoms (how-to)](../howto/select-atoms.md)
- [Atom selection language explained](../explanation/atom-selection-language.md)
