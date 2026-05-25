"""End-to-end smoke test for the molstar viewer.

Run with:
    uv run python examples/molstar_viewer_demo.py

Opens a browser tab with the mol*-based viewer, loads a small protein
from the test PDB set, then mutates coordinates and topology to exercise
both the coords-only fast path and the topology rebuild path.
"""

import os
import time

import numpy as np

from moleculekit.molecule import Molecule


def main():
    pdb_path = os.path.join(
        os.path.dirname(__file__), os.pardir, "tests", "pdb", "1awf.pdb"
    )
    protein = Molecule(pdb_path)
    protein.view(viewer="molstar", name="receptor")

    print(f"Opened viewer with {protein.numAtoms} atoms.")
    print("Waiting 3 s for the browser to connect ...")
    time.sleep(3)

    print("Applying 20 coords-only updates (should not rebuild the scene) ...")
    rng = np.random.default_rng(0)
    for _ in range(20):
        protein.coords[:, :, 0] += rng.normal(
            scale=0.05, size=protein.coords[:, :, 0].shape
        ).astype(np.float32)
        time.sleep(0.4)

    print("Filtering to CA atoms ... (expect a topology rebuild)")
    protein.filter("name CA")
    time.sleep(5)

    print("Demo done. Press Ctrl-C to close the server.")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
