# How to compute projections

## Goal

Extract a per-frame feature vector from a trajectory using a `Metric*` projection class.

## Minimal example

```python
from moleculekit.molecule import Molecule
from moleculekit.projections.metricdistance import MetricDistance

mol = Molecule("3PTB")

# Configure the metric once
metric = MetricDistance(
    "protein and name CA",
    "resname BEN",
    periodic="selections",
)

# Project returns (n_frames, n_features)
data = metric.project(mol)
print(data.shape)
```

## Parameters that matter

| Parameter | Type | What it does |
|---|---|---|
| `sel1` | `str` | First atom selection |
| `sel2` | `str` | Second atom selection |
| `metric` | `str` | `"distances"` (default) or `"contacts"` |
| `periodic` | `str` or `None` | `"selections"` computes periodic distances between the two selections; `"chains"` computes periodic distances between different chains; `None` disables periodic distances |

## Common variations

```python
# Cα RMSD versus a reference structure
from moleculekit.projections.metricrmsd import MetricRmsd

rmsd_metric = MetricRmsd(mol, "protein and name CA")
rmsd_data = rmsd_metric.project(mol)
```

```python
# Secondary structure assignment per frame
from moleculekit.projections.metricsecondarystructure import MetricSecondaryStructure

ss_metric = MetricSecondaryStructure()
ss_data = ss_metric.project(mol)
```

## Gotchas

- `Metric*` classes are configured once and can be applied to multiple trajectories in a loop — reuse the same object for efficiency.
- Output shape is always `(n_frames, n_features)`.
- {py:class}`~moleculekit.projections.metricdistance.MetricDistance` with `metric="contacts"` returns binary values; with `metric="distances"` it returns float distances in Å.
- For periodic-box trajectories make sure `mol.box` is populated; otherwise set `periodic=None`.

## See also

- [How to compute RMSD and RMSF](compute-rmsd-rmsf.md)
- [How to compute distances and contacts](compute-distances-and-contacts.md)
- [How to compute dihedrals](compute-dihedrals.md)
