@echo on
setlocal enabledelayedexpansion

rem Inject version from rattler/conda build environment
set SETUPTOOLS_SCM_PRETEND_VERSION=%PKG_VERSION%

%PYTHON% -m build -v -w --no-isolation
for %%F in (dist\*.whl) do %PYTHON% -m pip install "%%F"
