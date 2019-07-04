Projections
===========

The projections in MoleculeKit. These are a set of classes which take Molecule objects and calculate
some projections (or features if you wish) for each conformation inside the Molecule.
They range from typical RMSD calculations to more complex calculations such as SASA, secondary
structure, Plumed2 projections and more.

These can be used to analyse your simulations as well as build Markov models in conjunction with the
HTMD package.

Contents
--------

.. toctree::
   :maxdepth: 1
   
   MetricCoordinate: xyz coordinate projection  <moleculekit.projections.metriccoordinate>
   MetricDihedral: dihedral angles projection <moleculekit.projections.metricdihedral>
   MetricDistance: atom distances/contacts projection <moleculekit.projections.metricdistance>
   MetricFluctuation: atom position fluctuations <moleculekit.projections.metricfluctuation>
   MetricPlumed2: support for Plumed2 projections <moleculekit.projections.metricplumed2>
   MetricRmsd: root mean square deviation of structures to a reference <moleculekit.projections.metricrmsd>
   MetricSasa: solvent accessible surface area projection <moleculekit.projections.metricsasa>
   MetricSecondaryStructure: protein secondary structure projection <moleculekit.projections.metricsecondarystructure>
   MetricShell: density of atoms in shells around other atoms <moleculekit.projections.metricshell>
   MetricSphericalCoordinate: spherical coordinates of two COMs <moleculekit.projections.metricsphericalcoordinate>
   MetricTMscore: TMscore projection <moleculekit.projections.metrictmscore>