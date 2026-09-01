# Choose representations and colours

**Goal:** control how a molecule is drawn, by {py:meth}`~moleculekit.molecule.Molecule.view` and by {py:meth}`~moleculekit.molecule.Molecule.render`, using
`mol.reps`, a {py:class}`~moleculekit.representations.Representations` object.

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

{py:meth}`~moleculekit.representations.Representations.addDefaults` writes the automatic scene out as ordinary entries you can edit,
reorder or {py:meth}`~moleculekit.representations.Representations.remove`. Without it, {py:meth}`~moleculekit.representations.Representations.add` replaces the automatic scene entirely, so
colouring one ligand costs you the cartoon and everything else.

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
| `Backbone` | `backbone` | Backbone trace, proteins and nucleic acids |
| `Ellipsoid` | `ellipsoid` | Residues as ellipsoids |
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
| | `element-index` | Position along the chain, a different gradient |
| | `entity-id` | Entity: polymer, ligand, water |
| | `polymer-id` | Polymer |
| | `model-index` | Model |
| | `structure-index` | Structure |
| | `illustrative` | Chain, with non-carbon atoms picked out |

Any SVG colour name (`"red"`, `"steelblue"`) or `#rrggbb` string gives a
uniform colour, as does a VMD ColorID integer.

## Size and transparency

```python
mol.reps.add("resname BEN", "Licorice", "Name", size=0.4)                 # thin sticks
mol.reps.add("resname BEN", "VDW", "Name", size=0.85, opacity=0.25)       # ghost spheres
mol.reps.add("resname BEN", "Licorice", "Name", c_atom_color="#66ccff")   # cyan carbons
```

| Parameter | Meaning |
| --- | --- |
| `size` | Scales the drawn size: stick and sphere radius, surface probe, point size, label text. Each style keeps its own sensible size at `1`. |
| `opacity` | `0` is fully transparent, `1` fully opaque. |
| `c_atom_color` | Colours carbon only, leaving N, O and S their element colours. A colour, or one of `chain-id`, `entity-id`, `model-index`, `structure-index`. Needs an element-coloured representation, which is what it modifies. |
| `size_theme` | How sizes are decided before `size` scales them: `physical` for atomic radii, `uniform` for one size everywhere, `uncertainty` for the B factor. Each style picks a sensible one already. |
| `label_fields` | What a `Labels` representation writes beside each atom: any per-atom fields of the molecule, joined by spaces. Without it the label is the atom name. |
| `label_style` | Cosmetics for a `Labels` representation: `border_width`, `border_color`, `bg_color`, `bg_opacity`, `bg_margin`, `offset_x`, `offset_y`, `offset_z`. |
| `visibility` | `False` keeps the representation in the list but stops drawing it. |

Representations layer, so a transparent `VDW` over a `Licorice` of the same
selection gives sticks inside a ghost surface.

## Labelling atoms

```python
mol.reps.add("resname BEN", "Licorice", "Name")
mol.reps.add("resname BEN", "Labels", "black",
             label_fields=["resname", "resid", "name"], size=1.4)
```

```python
mol.reps.add("resname BEN", "Labels", label_fields="name", size=1.6,
             label_style={"bg_color": "#003366", "bg_opacity": 0.85, "offset_y": 1.5})
```

`label_fields` takes any per-atom fields the molecule has — `name`, `element`,
`resname`, `resid`, `chain`, `segid`, `beta`, `occupancy`, `index` and so on —
and writes them space-separated. Each label is drawn separately, so label a
selection worth naming rather than a whole structure; past a couple of hundred
atoms they are skipped with a warning.

## Changing a representation you already added

{py:meth}`~moleculekit.representations.Representations.update` changes one entry in place and leaves the rest of it
alone, so recolouring does not cost you the size or the selection. It keeps the
entry where it is, which is what makes indices stable: removing and re-adding
would move it to the end and renumber everything after it.

```python
mol.reps.update(0, color="Chain")
mol.reps.update(0, visibility=False)     # keep it, stop drawing it
mol.reps.update(0, visibility=True)
```

Only what you pass is changed, so passing nothing changes nothing.

## Gotchas

- **Representations replace the automatic scene, they do not add to it.** This
  matches the VMD backend. Use {py:meth}`~moleculekit.representations.Representations.addDefaults` to start from the automatic scene, or
  leave `mol.reps` empty to get it.
- **Formal charge labels follow the same rule.** With no representations they
  are drawn automatically on charged atoms; once you set any, add a
  `FormalCharges` representation to keep them.
- **`update_sel_every_frame` is for the interactive viewer.** It re-evaluates a
  coordinate-dependent selection such as `within 5 of resname BEN` on every
  trajectory frame. A rendered image is a single frame, so it is carried on the
  representation and changes nothing about the picture.
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
  still, so {py:meth}`~moleculekit.representations.Representations.add`'s `frames` argument has nothing to act on there.

## See also

- [View a molecule](view-a-molecule.md)
- [Render a molecule to an image file](render-an-image.md)
- [Select atoms](select-atoms.md)
