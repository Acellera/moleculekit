from moleculekit.molecule import Molecule
import sys
import os

batchidx = os.environ["BATCH_TASK_INDEX"]
indir = os.path.join("/workdir", sys.argv[1])
outdir = os.path.join("/workdir", sys.argv[2])

mol = Molecule(os.path.join(indir, f"{batchidx}.pdb"))
os.makedirs(outdir, exist_ok=True)
mol.write(os.path.join(outdir, f"results_{batchidx}.pdb"))
