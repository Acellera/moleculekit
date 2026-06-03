#!/usr/bin/env bash
# Build the pyodide wheel for moleculekit and drop it into ./dist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-moleculekit-pyodide}"
DEST_DIR="${REPO_ROOT}/dist"

mkdir -p "${DEST_DIR}"

docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE_TAG}" "${REPO_ROOT}"

docker run --rm -v "${DEST_DIR}:/out" "${IMAGE_TAG}" \
    sh -c 'cp /src/dist_pyodide/*.whl /out/ && ls -la /out/*.whl'

echo
echo "Wheel(s) copied to ${DEST_DIR}"
