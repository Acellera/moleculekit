#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}


pythonversions=(
    "cp36-cp36m"
    "cp37-cp37m"
    "cp38-cp38"
)

# Compile wheels
for PYBIN in "${pythonversions[@]}"; do
    "/opt/python/${PYBIN}/bin/pip" install -r /io/requirements.txt
    "/opt/python/${PYBIN}/bin/pip" install Cython>=0.29.21 
    "/opt/python/${PYBIN}/bin/pip" wheel /io/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# # Install packages and test
# for PYBIN in /opt/python/*/bin/; do
#     "${PYBIN}/pip" install python-manylinux-demo --no-index -f /io/wheelhouse
#     (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
# done
