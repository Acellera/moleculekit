# Wrapping tutorial: "why is my simulation exploding?"

Short answer: your simulation is not exploding. What you are seeing is an
artifact of how MD trajectories store coordinates. This page walks through
periodic boundary conditions, why molecules appear to drift "out of the box",
and how to wrap them back in for visualization.

## Periodic boundary conditions (PBC)

MD simulations are computationally expensive. Simulation time scales linearly
with the number of atoms, so we keep systems small — typically a few hundred
thousand atoms — for tractable performance. But a small isolated system does
not reflect the reality inside cells, where molecules are surrounded by a vast
amount of solvent. Periodic boundary conditions are how we approximate
infinite solvent without simulating it.

### What does this mean?

When computing interactions, we pretend the simulation box has identical
copies of itself on every side. Those copies in turn have copies on theirs,
and so on, tiling all of space. We call these copies **periodic images** of
the central box. We only ever store and integrate the central box — the
images are notional, used only when computing forces across the box
boundaries.

### Example

Consider a simulation box with four water molecules.

![box](wrapping_img/box.svg)

When we calculate interactions we pretend that we have copies of the system
on each side. Of course we are not integrating those copies — that would be a
huge waste. All copies are identical to the central box.

![box copies](wrapping_img/box2-2.svg)

In MD we use cutoffs for non-bonded interactions, which limits the distance
an atom "sees". Simplifying the example by treating each water as a single
point, if the orange circle is the cutoff, each water interacts with every
molecule inside its circle — including atoms in neighbouring periodic
images. That is how we pretend to simulate infinite solvent.

![cutoffs across the boundary](wrapping_img/box3-2.svg)

## Storing coordinates during a simulation

Molecules can reach the borders of the simulation box. When they do we have
two options:

1. Teleport them to the opposite side of the box.
2. Let them keep moving past the boundary; record the absolute coordinates.

The first approach is visually pleasant but has two downsides. You can no
longer measure the total drift of the atoms, and if done carelessly it can
cause precision issues in force calculations.

The second approach is generally preferred. The atoms' stored coordinates are
not reset; interactions are still computed as if every atom were located in
the central simulation box.

So if you see atoms "leaving" the simulation box in your trajectory, that
does **not** mean they have stopped interacting with the rest of the system —
they have simply moved into the next periodic image, and the engine still
treats them correctly.

### Example

Consider a simulation of villin in a water box. At time zero the protein
looks nicely centred in the waters because of how we built the box.

![villin built](wrapping_img/villin_build.png)

After running for a while the visualization looks like this. This is normal —
we chose the second storage strategy, so molecules can move outside the
original box into periodic images.

![villin no wrap](wrapping_img/villin_sim_nowrap.png)

In reality all atoms are still close to each other and interacting.

In some viewers (e.g. VMD) you can render the periodic images directly. If we
do that for the same trajectory frame, the picture shows that the system
atoms are still interacting through the periodic copies:

![villin periodic](wrapping_img/villin_sim_nowrap_periodics.png)

That's confusing to interpret with so many copies around. We'd rather have a
single wrapped box.

## Wrapping

When we visualize a simulation we only see the real atoms, not the notional
periodic copies. If a molecule has drifted into a neighbouring image, its
interactions with the rest of the system become invisible to the eye.

**Wrapping** is the operation that places all atoms back into a single
simulation box. Since we know the box size and the absolute position of every
atom, it is mechanical to fold each atom back into the central image.

### How does wrapping work?

One question remains: where is the centre of the box? Our system drifts
freely during the simulation, so we cannot use a fixed XYZ coordinate as the
centre. Instead, wrapping defines the centre by **the coordinates of a
chosen atom selection** — either a single atom or the average over a set of
atoms.

### How to select a good wrapping centre

This depends on the system and on personal taste.

If the simulation contains a single protein, you usually want to wrap so the
protein sits in the middle of the box. Use the average coordinate of the
protein as the wrapping centre:

![villin centred on protein](wrapping_img/villin_wrapped_correct.png)

The waters now wrap nicely around the protein.

This is an intuitive case. You can still get it wrong, e.g. by wrapping
around a single water residue instead of the protein:

![villin centred on bad residue](wrapping_img/villin_wrapped_bad_4.png)

Now the protein sticks out of one face of the box and "vacuum" appears on
the others. There is no real vacuum — the piece of the protein that sticks
out occupies that space in a periodic image — but moleculekit prefers to
keep bonded molecules intact rather than break bonds across the boundary,
so the visualization stays one-sided.

If the simulation has two proteins, you usually centre on one of them. The
average of two proteins varies strongly during the simulation, so the choice
of centre changes from frame to frame. When the proteins are in contact the
average works (as below); when there are unbinding events, choose one or the
other.

![two proteins](wrapping_img/image.png)

If the simulation has a protein and a membrane, the right centre depends on
what you want to see. In the first image we centred on the protein — the
membrane wraps to the top. In the second we centred on the membrane — the
protein wraps to the bottom. In the third we picked a residue half-way
between the two for a clean visualization with both whole.

![membrane wrapping](wrapping_img/membrane_wrapping.png)

## How to wrap with moleculekit

```python
from moleculekit.molecule import Molecule

mol = Molecule("./structure.prmtop")   # PSF is also fine
mol.read("./output.xtc")               # DCD is also fine
mol.wrap("protein")                    # centre on the average protein coordinate
mol.write("./output_wrapped.xtc")
```

If your system is more complex than a single protein in solvent you may need
to find a specific residue or atom to wrap around. Visualize the simulation
in your viewer of choice, look around, and pick a residue that gives the
cleanest picture:

```python
mol.view()
mol.wrap("resid 15 and chain A")
mol.write("./output_wrapped.xtc")
```

Wrapping only affects the visual layout of the trajectory — it is not a sign
of a broken or exploding simulation. The right choice of wrapping centre
depends on the system; once you find it you can apply the same selection to
every trajectory of the same setup.

## See also

- [How to wrap trajectories](wrap-trajectories.md) — the focused recipe (no MD background).
- [Trajectories and frames](../explanation/trajectories-and-frames.md) — how moleculekit stores coordinates, frames, and box arrays.
