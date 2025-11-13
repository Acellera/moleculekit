export SETUPTOOLS_SCM_PRETEND_VERSION=${PKG_VERSION}
$PYTHON -m build -v -w --no-isolation
$PYTHON -m pip install dist/*.whl
