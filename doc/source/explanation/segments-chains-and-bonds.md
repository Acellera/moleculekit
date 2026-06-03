# Segments, chains, and bonds

Moleculekit inherits two parallel notions of molecular grouping — **chains**
from the PDB world and **segments** from the MD world — and stores covalent
connectivity in a separate **bond array**. Understanding why both exist, what
populates them, and when each matters helps you avoid subtle bugs in
downstream preparation and parameterization workflows.

## Chains: PDB heritage

The `chain` field (dtype `object`, shape `(numAtoms,)`) is read directly from
column 22 of a PDB file, or from the equivalent field in mmCIF/PDBx. It is
typically a single uppercase letter: `"A"`, `"B"`, `"C"`, etc.

PDB convention: each **polymer chain** (protein, DNA, RNA) gets its own chain
identifier. Small molecules, ions, and water are often all lumped into a single
chain (commonly `"A"` or left blank). Chain identifiers reset on every new
deposited structure; they are not globally unique.

After loading a PDB:

```python
from moleculekit.molecule import Molecule

mol = Molecule("3ptb")
import numpy as np
print(np.unique(mol.chain))   # e.g. ['A']
```

Chain identifiers matter for:
- Identifying polymer chains visually and in selection strings (`chain A`).
- PDB output: molecules with no chain set (`""`) still write a valid file, but
  tools that expect a populated chain column may fail silently.

## Segments: MD heritage

The `segid` field (dtype `object`, shape `(numAtoms,)`) carries up to four
characters and originates from the CHARMM/NAMD world. AMBER's `tleap`,
CHARMM's PSF builder, and related parameterization tools use the segment
identifier to group atoms into logical units that can receive independent
force-field parameters.

A freshly loaded PDB has **empty** segment identifiers (`""`). A PSF or
PRMTOP file — which is written by tleap or similar — almost always has segment
identifiers populated.

```python
print(np.unique(mol.segid))   # [''] for a plain PDB
```

## Why both fields exist

PDB files were designed for structure deposition and exchange. Column 22 (one
character, chain) is the only built-in grouping. MD topology formats needed
more granularity — multiple segments per chain for long proteins split across
multiple tleap units, separate segments for lipid tails vs. head-groups, etc.
— so they introduced the segment concept independently.

In practice:
- **Use `chain`** when reading PDB output, working with visualization tools, or
  writing chain-aware selection strings.
- **Use `segid`** when preparing a system for MD parameterization. Tools like
  `tleap` read the segment column and need it non-empty and consistent.

If you load a PDB and then hand it to a parameterizer without populating
`segid`, you will often get an error. Running {py:func}`~moleculekit.tools.autosegment.autoSegment` (see below) before
{py:func}`~moleculekit.tools.preparation.systemPrepare` is the standard fix.

## autoSegment: populating segments automatically

{py:func}`~moleculekit.tools.autosegment.autoSegment` assigns segments by
following the **physical backbone** of each polymer. It walks the residues of a
selection in file order and starts a new segment only where the backbone is
actually broken — judged from atomic coordinates, not from residue numbering —
so it is robust to structures where residues were deleted from the sequence
while the chain stayed continuous.

```python
from moleculekit.tools.autosegment import autoSegment

mol_seg = autoSegment(mol)   # returns a modified copy
import numpy as np
print(np.unique(mol_seg.segid))   # e.g. ['P0', 'P1', 'P2']
```

A new segment begins between two consecutive residues when **any** of these holds:

- the backbone link distance exceeds the cutoff — for protein the `C(i)–N(i+1)`
  peptide bond (default `protein_cutoff=2.0` Å, falling back to `CA–CA` when the
  carbonyl/amide atoms are missing), for nucleic acids the `O3'(i)–P(i+1)`
  phosphodiester bond (default `nucleic_cutoff=2.2` Å);
- the `chain` or `segid` already recorded in the file changes;
- the polymer type changes (protein vs nucleic).

Because continuity is judged from coordinates, a gap in residue numbering with an
intact backbone (for example residues mutated out of a sequence) stays a
**single** segment, while a genuine spatial break — even one with continuous
numbering — is split into two.

Non-polymer atoms are grouped separately: all **water** collapses into one
segment, all **ions** into another, and the remaining molecules ("other") are
split into one segment per bonded molecule. Pass `single_other_segment=True` to
place all of the "other" molecules into a single segment instead.

The `basename` argument controls naming (`basename="P"` produces `P0`, `P1`,
`P2`, ...). The `fields` argument controls which field(s) are written:
`("segid",)` (default), `("chain",)`, or `("segid", "chain")`.

Run {py:func}`~moleculekit.tools.autosegment.autoSegment` before
{py:func}`~moleculekit.tools.preparation.systemPrepare` so the parameterizer
receives a populated, consistent `segid`.

```{note}
The older `autoSegment2` — which segmented by the covalent bond graph — is
deprecated. It now forwards to
{py:func}`~moleculekit.tools.autosegment.autoSegment` and emits a
`DeprecationWarning`; call `autoSegment` directly instead.
```

## Bonds: the connectivity layer

Bonds live in two parallel arrays:

- `mol.bonds` — `uint32`, shape `(numBonds, 2)`. Each row `[i, j]` is an
  atom-index pair.
- `mol.bondtype` — `object`, shape `(numBonds,)`. Bond-order or type string,
  parallel to `mol.bonds`. Common values: `"1"` (single), `"2"` (double),
  `"ar"` (aromatic), `"un"` (unknown), `"mc"` (metal coordination).

The bond count is `mol.bonds.shape[0]`.

### When are bonds present?

Bonds are populated whenever the source file contains explicit connectivity:

| Format | Bonds present? | Source |
|---|---|---|
| PDB with `CONECT` records | Yes | `CONECT` + `LINK` blocks |
| mmCIF / PDBx | Yes | `_struct_conn` entries |
| PSF (CHARMM) | Yes | `!NBOND` section |
| PRMTOP (AMBER) | Yes | `BONDS_*` section |
| MOL2 | Yes | `@<TRIPOS>BOND` section |
| SDF / MOL | Yes | bond table |
| Plain PDB (no `CONECT`) | No | — |

For plain PDB files, call {py:meth}`~moleculekit.molecule.Molecule.guessBonds`
to infer bonds from inter-atomic distances and radii — it updates `mol.bonds`
and `mol.bondtype` together:

```python
mol = Molecule("structure.pdb")
mol.guessBonds()
print(mol.bonds.shape[0])      # now populated
print(mol.bondtype.shape[0])   # same length, kept in lockstep
```

### Metal coordination bonds

The PDB and mmCIF readers populate **metal-coordination bonds** in addition
to covalent ones. These appear as regular rows in
`mol.bonds` with `bondtype == "mc"`. They are preserved through
{py:func}`~moleculekit.tools.preparation.systemPrepare` when `hold_nonpeptidic_bonds=True` (the default), so metal
chelation geometries survive the preparation pipeline.

```python
# Select all metal-coordination bonds
mc_mask = mol.bondtype == "mc"
mc_bonds = mol.bonds[mc_mask]
print(mc_bonds)   # index pairs involving metal atoms
```

## Topology versus connectivity

It helps to think of a `Molecule` as two orthogonal layers:

- **Topology** — atoms and their per-atom fields (`name`, `resname`, `resid`,
  `chain`, `segid`, ...). This layer answers: "what atoms exist and what are
  their identities?"
- **Connectivity** — bonds. This layer answers: "which atoms are covalently (or
  metal-coordination) bonded to which?"

PDB files deliver topology but **typically not connectivity** (unless `CONECT`
records are present). Connectivity-rich formats like PSF and PRMTOP deliver
both. This distinction matters:

- Selection strings that rely on `same fragment as` or bond-graph traversal
  require non-empty `mol.bonds`.
- {py:func}`~moleculekit.tools.preparation.systemPrepare` can work without explicit bonds but calls {py:func}`~moleculekit.bondguesser.guess_bonds`
  internally if needed.
- Any visualization in VMD or the built-in Molstar viewer will draw bonds only
  if they are present.

## Further reading

- How-to: [Guess bonds](../how-to/guess-bonds.md)
- How-to: [Assign segments and chains](../how-to/assign-segments-and-chains.md)
