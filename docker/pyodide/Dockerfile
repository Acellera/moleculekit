# Build moleculekit as a Pyodide (WASM) wheel.
#
# Build:
#   docker build -f docker/pyodide/Dockerfile -t moleculekit-pyodide .
# Extract the wheel(s):
#   docker run --rm -v "$PWD/dist_pyodide:/out" moleculekit-pyodide \
#       sh -c 'cp /src/dist_pyodide/*.whl /out/'

FROM python:3.13-slim-bookworm

ARG PYODIDE_VERSION=0.28.3
# pyodide-build is versioned independently from the pyodide runtime; 0.28.0 is
# the build-tool release that targets the 0.28.x runtime line.
ARG PYODIDE_BUILD_VERSION=0.28.0
ARG EMSCRIPTEN_VERSION=4.0.9
ARG EMSDK_DIR=/opt/emsdk

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        pkg-config \
        python3-dev \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install emsdk pinned to the emscripten version pyodide 0.28.3 was built against.
# (Verify with `pyodide config get emscripten_version` if you bump PYODIDE_VERSION.)
RUN git clone https://github.com/emscripten-core/emsdk.git ${EMSDK_DIR} \
    && cd ${EMSDK_DIR} \
    && ./emsdk install ${EMSCRIPTEN_VERSION} \
    && ./emsdk activate ${EMSCRIPTEN_VERSION}

ENV PATH=${EMSDK_DIR}:${EMSDK_DIR}/upstream/emscripten:${PATH} \
    EMSDK=${EMSDK_DIR} \
    EM_CONFIG=${EMSDK_DIR}/.emscripten

# pyodide-build 0.28.x stores the cross-build env in ./.pyodide-xbuildenv-<ver>
# *relative to the working directory*, and `pyodide build` falls back to
# downloading the latest xbuildenv if it can't find one in its CWD. So the
# prefetch below and the `pyodide build` further down must run from the same
# directory (/src) — otherwise the build pulls an incompatible newer xbuildenv.
WORKDIR /src

# wheel>=0.46 removed wheel.cli, which auditwheel-emscripten (a pyodide-build
# dep) still imports. Pin wheel<0.46 until the upstream chain catches up.
RUN pip install "pyodide-build==${PYODIDE_BUILD_VERSION}" "wheel<0.46"

# pyodide-build 0.28.x's hardcoded default metadata URL
# (raw.githubusercontent.com/pyodide/pyodide/main/pyodide-cross-build-environments.json)
# now 404s — the file was relocated and pyodide-build 0.30.4+ ships a new default
# pointing at gh-pages. Override the 0.28.0 default to the same gh-pages URL.
ENV PYODIDE_CROSS_BUILD_ENV_METADATA_URL=https://pyodide.github.io/pyodide/api/pyodide-cross-build-environments.json

# Pre-fetch the cross-build environment so the build step is offline-friendly
# and pinned to PYODIDE_VERSION (not whatever "latest" happens to be).
RUN pyodide xbuildenv install ${PYODIDE_VERSION}

# Confirm the pinned EMSCRIPTEN_VERSION matches what pyodide expects; fail fast otherwise.
RUN expected=$(pyodide config get emscripten_version) \
    && [ "$expected" = "${EMSCRIPTEN_VERSION}" ] \
    || (echo "Emscripten mismatch: pyodide wants $expected, image has ${EMSCRIPTEN_VERSION}" && exit 1)

# Copy the sources last so editing them doesn't invalidate the (slow) emsdk /
# pyodide-build / xbuildenv layers above.
COPY . /src

# PyO3 cross-compilation env vars are only needed when building Rust deps; set
# placeholders so the build doesn't error out trying to find them. See:
# https://github.com/pyodide/pyodide/issues/4347
ENV PYO3_CROSS_INCLUDE_DIR=/tmp/pyo3-stub \
    PYO3_CROSS_LIB_DIR=/tmp/pyo3-stub
RUN mkdir -p /tmp/pyo3-stub

RUN rm -rf dist_pyodide \
    && . ${EMSDK_DIR}/emsdk_env.sh \
    && pyodide build -o dist_pyodide

CMD ["sh", "-c", "ls -la /src/dist_pyodide"]
