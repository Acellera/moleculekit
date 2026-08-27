# Render a molecule to an image file

**Goal:** produce a PNG of a molecule with no viewer window, for a script, a
report, or an agent that cannot look at a screen.

## Example

```python
from moleculekit.molecule import Molecule

mol = Molecule("3PTB")
mol.filter("not water")
mol.reps.add(sel="protein", style="NewCartoon", color="SecondaryStructure")
mol.reps.add(sel="resname BEN", style="Licorice", color=4)

mol.render("trypsin.png", size=(1200, 900), rotate="top", center="resname BEN")
```

Passing no path returns the PNG bytes instead, which suits writing to a stream:

```python
png = mol.render(size=(800, 600))
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `output` | Path to write. Omit to get the PNG bytes back. |
| `size` | `(width, height)` in pixels. |
| `quality` | `"fast"` or `"high"`. `"high"` adds ambient occlusion, which is close to free on a GPU and costs roughly three times the time on the software fallback. |
| `center` | Atom selection to frame the camera on. |
| `rotate` | `"front"`, `"back"`, `"left"`, `"right"`, `"top"`, `"bottom"`, or `(rx, ry, rz)` in degrees. |
| `zoom` | Larger values move the camera closer. |
| `clip` | Half-thickness in Angstrom of the slab drawn around what the camera frames, cutting away what is in front of and behind it. Omit to draw the whole structure. |
| `fog` | Depth cueing strength, `0` to `100`. Fog fades distant geometry into the background colour. Omit for Mol\*'s own strength. |
| `background` | Colour name or hex string. Ignored when `transparent=True`. |
| `transparent` | Render onto a transparent background, ignoring `background`. |
| `timeout` | Seconds allowed for one render. |

## Requirements

Rendering needs a chromium binary and the optional `molviewspec` dependency:

```
pip install moleculekit[notebook]
```

moleculekit looks for `chromium`, `chromium-browser`, `google-chrome` or
`chrome` on your PATH. To use a specific binary, set `MOLECULEKIT_CHROMIUM` to
its path. No network connection is needed: the Mol* bundle ships with
moleculekit and the structure never leaves the machine.

## Choosing the graphics backend

Rendering uses your GPU when one is reachable and falls back to a software
rasteriser when it is not, which is what makes it work unchanged inside
containers that expose no graphics device. The difference is large: a 1200x900
render measured 0.5 seconds on a GPU against 6.9 seconds in software.

The choice is automatic. To pin it, set `MOLECULEKIT_RENDER_GL`:

```
MOLECULEKIT_RENDER_GL=hardware   # require the GPU, fail if it is unavailable
MOLECULEKIT_RENDER_GL=software   # always use the software rasteriser
MOLECULEKIT_RENDER_GL=auto       # the default
```

Forcing `hardware` is useful when a render unexpectedly falls back and you want
the failure to be loud rather than slow. Forcing `software` is useful for
reproducing what a GPU-less machine will produce.

## Gotchas

- **Representations come from `mol.reps`**, the same ones `view()` uses, and
  they replace the automatic scene rather than adding to it. Setting any
  representation means you are describing the whole picture, which matches how
  the VMD backend behaves. Leave `mol.reps` empty to get the automatic scene of
  a cartoon polymer with ball-and-stick ligands, or call `mol.reps.addDefaults()`
  to start from it and edit. See
  [Choose representations and colours](choose-representations.md) for the
  styles, colour modes, size and transparency.
- **A representation whose selection matches no atoms is dropped with a
  warning.** If every representation matches nothing, `render()` raises rather
  than writing an empty image.
- **A representation's `frames` argument is ignored.** `mol.reps.add(..., frames=...)`
  selects frames for the interactive viewer's trajectory playback; a render is a
  single still, so `frames` has nothing to act on.
- **The frame rendered is `mol.frame`.** For a trajectory, set `mol.frame = n`
  first. `render()` produces a single still, not a movie.
- **`center` must match at least one atom.** A selection that matches nothing
  raises `ValueError`, rather than silently falling back to the default view.
- **`center` is also what `rotate` orbits and what `zoom` scales.** Framing a
  ligand and stepping `rotate` through angles gives a turntable around it.
- **The framing follows the requested `size`.** A wider image shows more around
  the structure rather than the same picture stretched.
- **The first call is slower** because it starts the browser. Later calls reuse
  it, so rendering many images in one process costs the startup once.
- **Speed depends on whether a GPU is available.** With one, a 1200x900 render
  takes well under a second. Without one, rendering falls back to a software
  rasteriser that fills every pixel on the CPU, which costs several seconds and
  scales with pixel count, so a 4K image costs roughly four times a 1080p one.
  See "Choosing the graphics backend" below.
- **Call `render()` from one thread at a time.** The headless browser is a
  singleton with a single devtools connection, so concurrent calls from
  different threads (a multi-threaded service handling several requests at
  once, for example) can read each other's responses. Serialise calls to
  `render()`, for example behind a lock, if more than one thread might call it.
