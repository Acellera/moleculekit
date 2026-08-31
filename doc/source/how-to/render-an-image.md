# Render a molecule to an image file

**Goal:** save a picture of a molecule to a file, without opening a viewer
window, using {py:meth}`~moleculekit.molecule.Molecule.render`. Useful in a
script that produces figures, or anywhere no screen is available.

## Example

```python
from moleculekit.molecule import Molecule

mol = Molecule("3PTB")
mol.filter("not water")
mol.reps.add(sel="protein", style="NewCartoon", color="SecondaryStructure")
mol.reps.add(sel="resname BEN", style="Licorice", color=4)

mol.render("trypsin.png", size=(1200, 900), rotate="top", center="resname BEN")
```

Leave out the file name and you get the image data back instead, which is handy
for writing it somewhere else:

```python
png = mol.render(size=(800, 600))
```

## Options

| Option | What it does |
| --- | --- |
| `output` | File to write. Leave it out to get the image data back. |
| `size` | Width and height of the image in pixels. |
| `quality` | `"fast"` or `"high"`. `"high"` adds soft shading in the crevices and smoother edges, and takes noticeably longer without a graphics card. |
| `center` | Atom selection to point the camera at, such as `"resname BEN"`. |
| `rotate` | `"front"`, `"back"`, `"left"`, `"right"`, `"top"`, `"bottom"`, or your own `(x, y, z)` rotation in degrees. |
| `zoom` | How close the camera sits. Larger is closer. |
| `clip` | Show only a slab this many Angstrom thick around what the camera is pointed at, cutting away what is in front and behind. Leave it out to show the whole structure. |
| `fog` | How much distant parts fade into the background, from `0` for none to `100` for the most. |
| `background` | Background colour, by name or as `#rrggbb`. |
| `transparent` | Give the image a transparent background instead of a colour. |
| `timeout` | How many seconds to allow for one image before giving up. |

## Choosing what is drawn

Representations come from `mol.reps`, exactly as for
{py:meth}`~moleculekit.molecule.Molecule.view`, so a molecule renders the way
you would see it on screen. Leave `mol.reps` empty and you get a sensible
default picture: a cartoon for the protein or nucleic acid, with ligands, ions
and waters drawn as ball-and-stick.

Adding a representation replaces that default picture rather than adding to it,
so if you only want to change one thing, start from the default with
{py:meth}`~moleculekit.representations.Representations.addDefaults` and edit
it:

```python
mol.reps.addDefaults()
mol.reps.remove(1)          # drop the waters, ligand and ions
mol.reps.add("resname BEN", "VDW", "Name")
```

See [Choose representations and colours](choose-representations.md) for the
available styles, colours, sizes and transparency.

## Requirements

Rendering uses the Chromium browser behind the scenes, so it needs Chromium
(or Chrome) installed, plus one optional package:

```
pip install moleculekit[notebook]
```

moleculekit finds `chromium`, `chromium-browser`, `google-chrome` or `chrome`
on your PATH by itself. To point it at a particular one, set the
`MOLECULEKIT_CHROMIUM` environment variable to that program's location.

Nothing is uploaded anywhere: no internet connection is needed and your
structure never leaves your machine.

## Good to know

- **The picture shows the current frame**, `mol.frame`. For a trajectory, set
  `mol.frame = n` first and render again for another frame. Each call makes one
  still image, not a movie.
- **`center` is also what `rotate` turns around and what `zoom` scales.**
  Pointing at a ligand and stepping `rotate` through several angles gives you a
  turntable around it:

  ```python
  for angle in (0, 90, 180, 270):
      mol.render(f"ligand_{angle}.png", center="resname BEN", rotate=(0, angle, 0))
  ```

- **`center` has to match at least one atom**, otherwise you get an error
  rather than a picture of something you did not ask for.
- **The framing follows the size you ask for.** A wider image shows more around
  the molecule rather than the same picture stretched.
- **A representation matching no atoms is skipped with a warning**, and if none
  of them match anything you get an error instead of a blank image.
- **`frames` has no effect here.** It chooses frames for trajectory playback in
  the interactive viewer, and a rendered image is a single still.
- **Render one image at a time.** If your program uses threads, do not call
  render from several of them at once.
