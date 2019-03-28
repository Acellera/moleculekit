#!/usr/bin/env bash

# Build for conda from PyPI
source deactivate test
conda build package/moleculekit/
bash -x ./ci/travis/conda_upload.sh;
