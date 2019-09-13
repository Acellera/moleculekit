Moleculekit
===========

Welcome to Moleculekit's documentation!

Moleculekit is a python library that provides object-oriented classes and methods for manipulation of biomolecular structures.
With a few simple python commands you can read, write and convert between many different file formats, align structures and
calculate various projections from your molecules such as RMSD, RMSF, secondary structure, SASA, contact maps and much more.

The two main classes are `Molecule` and SmallMol. Molecule is used for manipulating larger molecules and typical molecular dynamics (MD) 
related formats such as PDB/PSF/PRMTOP etc. SmallMol (in conjunction with SmallMolLib) on the other hand is meant to handle large 
libraries of small molecules such as SDF files in a fast but more limited manner. 

Both classes support visualization in both the VMD desktop viewer and the NGL jupyter-notebook viewer for easy visualization
of the molecular structures.

The source code of Moleculekit is available under https://github.com/Acellera/moleculekit and you can use the issue tracker
there to report any bugs or issues you encounter using the library.

Contents
--------

.. toctree::
   :maxdepth: 3
   
   Installation <installation>
   The Molecule class <moleculekit.molecule>
   The SmallMol class <moleculekit.smallmol.smallmol>
   Molecule tools <moleculetools>
   Projections <projections>


**Indices and tables**

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
