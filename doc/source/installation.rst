Installation
============

You can install MoleculeKit either using pip or conda.
Various methods of MoleculeKit have additional dependencies which are not installed by default to keep the package
lighter and keep it installable by pip. If you encounter errors with missing libraries you can install these manually to resolve the issues.

We generally advise to use a conda installation regardless if installing MoleculeKit over pip or conda
as well as creating a separate conda environment for the installation to avoid conflicts with other libraries
or python version. If you however prefer to install it to your system python you can ignore the following two steps
related to conda installation and environments at your own risk.

Downloading miniconda
---------------------
Miniconda is a lightweight installer of conda containing just a few basic python libraries.
Download miniconda from the following URL https://docs.conda.io/en/latest/miniconda.html
and install it following the given instructions.

Create a conda environment
--------------------------
After installing miniconda you can create a conda environment for MoleculeKit with the following command::

   conda create -n moleculekit

This will not install MoleculeKit, it simply creates a clean empty python environment named `moleculekit`.
To now operate within this new environment use the following command to activate it. Anything installed with
conda or pip after this command will be installed into this clean python environment.:: 

   conda activate moleculekit

Install moleculekit with conda
------------------------------
::

   conda install moleculekit python=3.9 -c acellera -c conda-forge

This will install moleculekit alongside python 3.9 version which is the currently developed version of moleculekit.