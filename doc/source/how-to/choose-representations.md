# Choose representations and colours

**Goal:** control how a molecule is drawn, by `view()` and by `render()`, using
`mol.reps`.

## Example

```python
from moleculekit.molecule import Molecule

mol = Molecule("3PTB")

mol.reps.addDefaults()            # the scene you get with no representations
mol.reps.remove(1)                # drop the automatic waters/ion/ligand

mol.reps.add("protein and within 5 of resname BEN", "Surf", "Hydrophobicity",
             opacity=0.3)
mol.reps.add("resname BEN", "ball-and-stick", "element-symbol", size=0.4)
mol.reps.add("resname BEN", "Labels", "black", size=1.6)

mol.render("pocket.png", center="resname BEN", zoom=0.3)
```

`mol.reps.addDefaults()` writes the automatic scene out as ordinary entries you
can edit, reorder or remove. Without it, adding any representation replaces the
automatic scene entirely, so colouring one ligand costs you the cartoon and
everything else.

## Styles

Each style can be written in VMD's vocabulary or Mol\*'s. Spacing, case and
hyphens are ignored, so `Secondary Structure`, `secondary-structure` and
`SecondaryStructure` are one name.

| VMD name | Mol\* name | Draws |
| --- | --- | --- |
| `NewCartoon`, `Cartoon` | `cartoon` | Ribbon following the backbone |
| `Licorice`, `CPK` | `ball-and-stick` | Sticks with atoms as spheres |
| `VDW` | `spacefill` | Space-filling spheres |
| `Lines` | `line` | Bond lines, cheapest to draw |
| `Surf` | `molecular-surface` | Solvent-excluded surface |
| `QuickSurf` | `gaussian-surface` | Coarser gaussian surface, cheaper |
| `Points` | `point` | One dot per atom |
| `Putty` | `putty` | Tube thickened by B factor |
| `Labels` | `atom-label`, `label` | Each atom's name beside it |
| `FormalCharges` | | `+1`/`-1` on atoms carrying a formal charge |

`Labels` and `FormalCharges` draw text on top of another representation, so
they need one that draws the atoms themselves. A style from neither column is
passed to the VMD backend as written, so VMD's own styles still work there.

## Colours

| VMD name | Mol\* name | Colours by |
| --- | --- | --- |
| `Name`, `Element` | `element-symbol` | Chemical element |
| `Chain` | `chain-id` | Chain |
| `ResName` | `residue-name` | Residue name |
| `Index` | `sequence-id` | Position in the sequence |
| `Secondary Structure` | `secondary-structure` | Helix, sheet, coil |
| `Beta` | `uncertainty` | B factor, or pLDDT for predicted structures |
| `Occupancy` | `occupancy` | Occupancy |
| `Hydrophobicity` | `hydrophobicity` | Residue hydrophobicity |
| `Molecule Type` | `molecule-type` | Protein, nucleic, water, ligand |
| `Atom ID` | `atom-id` | Atom index |

Any SVG colour name (`"red"`, `"steelblue"`) or `#rrggbb` string gives a
uniform colour, as does a VMD ColorID integer.

## Size and transparency

```python
mol.reps.add("resname BEN", "Licorice", "Name", size=0.4)                 # thin sticks
mol.reps.add("resname BEN", "VDW", "Name", size=0.85, opacity=0.25)       # ghost spheres
```

| Parameter | Meaning |
| --- | --- |
| `size` | Scales the drawn size: stick and sphere radius, surface probe, point size, label text. Each style keeps its own sensible size at `1`. |
| `opacity` | `0` is fully transparent, `1` fully opaque. |

Representations layer, so a transparent `VDW` over a `Licorice` of the same
selection gives sticks inside a ghost surface.

## Gotchas

- **Representations replace the automatic scene, they do not add to it.** This
  matches the VMD backend. Use `mol.reps.addDefaults()` to start from the
  automatic scene, or leave `mol.reps` empty to get it.
- **Formal charge labels follow the same rule.** With no representations they
  are drawn automatically on charged atoms; once you set any, add a
  `FormalCharges` representation to keep them.
- **An unknown style or colour raises.** It used to draw ball-and-stick or a
  garbage uniform colour, which looked like a deliberate picture rather than a
  typo.
- **A representation whose selection matches no atoms is dropped with a
  warning**, and if every one matches nothing, the scene raises rather than
  drawing an empty image.
- **`Putty`, `Labels` and `FormalCharges` only reach Mol\***. The VMD and NGL
  backends skip them rather than being sent a style they have no
  representation for.
- **`frames` only applies to the interactive viewer.** A render is a single
  still, so `mol.reps.add(..., frames=...)` has nothing to act on there.

## See also

- [View a molecule](view-a-molecule.md)
- [Render a molecule to an image file](render-an-image.md)
- [Select atoms](select-atoms.md)
