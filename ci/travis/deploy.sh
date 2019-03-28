#!/usr/bin/env bash

# Build for conda from PyPI
source deactivate test
conda build package/htmdmol/

# If the build did not fail, if it is not PR, it's on Acellera/htmdmol and a tagged build, do anaconda upload tasks
bash -x ./ci/travis/conda_upload.sh;
