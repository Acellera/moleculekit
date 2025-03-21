Installation
============

You can install MoleculeKit either using pip or conda.

We generally advise to use a conda or `uv` installation regardless if installing MoleculeKit over pip or conda
as well as creating a separate environment for the installation to avoid conflicts with other libraries
or python versions. 
If you however prefer to install MoleculeKit to your system python you can ignore the following steps
related to Miniforge/UV installation and environments at your own risk.

Installing Miniforge
--------------------
Miniforge is a lightweight installer of the conda package manager.
Download Miniforge from the following URL https://github.com/conda-forge/miniforge?tab=readme-ov-file#install
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

   conda install moleculekit -c acellera -c conda-forge


Install moleculekit with pip
------------------------------
::

   conda install python=3.10
   pip install moleculekit


Using MoleculeKit with uv
-------------------------

Alternatively you can install `uv` to manage your python installations and environments.
Follow the installation instructions at https://github.com/astral-sh/uv?tab=readme-ov-file#installation

`uv` environments are stored in the filesystem in project directories. Let's create here a project directory 
called `myproject` and install moleculekit within it with the following commands::

::

   uv init myproject
   cd myproject
   uv add moleculekit ipython
   uv run ipython
