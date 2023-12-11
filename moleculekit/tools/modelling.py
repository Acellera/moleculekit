from moleculekit.molecule import Molecule
from moleculekit.util import find_executable
from subprocess import run
import tempfile
import os

CODE = """
from ost import io
from promod3 import modelling, loop

minimize = {minimize}
build_sidechains = {build_sidechains}

# setup
merge_distance = {merge_distance}
fragment_db = loop.LoadFragDB()
structure_db = loop.LoadStructureDB()
torsion_sampler = loop.LoadTorsionSamplerCoil()

# get raw model
tpl = io.LoadPDB("input.pdb")
aln = io.LoadAlignment("input.fasta")
aln.AttachView(1, tpl.CreateFullView())
mhandle = modelling.BuildRawModel(aln)

# we're not modelling termini
modelling.RemoveTerminalGaps(mhandle)

# perform loop modelling to close all gaps
modelling.CloseGaps(
    mhandle, merge_distance, fragment_db, structure_db, torsion_sampler
)

# build sidechains
if build_sidechains:
    modelling.BuildSidechains(
        mhandle, merge_distance, fragment_db, structure_db, torsion_sampler
    )

# minimize energy of final model using molecular mechanics
if minimize:
    modelling.MinimizeModelEnergy(mhandle)

# check final model and report issues
modelling.CheckFinalModel(mhandle)

# extract final model
final_model = mhandle.model
io.SavePDB(final_model, "model.pdb")"""


def model_gaps(
    mol,
    sequence,
    segid,
    promod_img,
    minimize=False,
    build_sidechains=True,
    merge_distance=4,
):
    """Model gaps in a protein sequence using ProMod3.

    This method will also mutate any residues in the Molecule that do not match the input sequence.

    Parameters
    ----------
    mol : Molecule
        The molecule containing the segment to model.
    sequence : str
        The sequence to model.
    segid : str
        The segment ID of the segment to model.
    promod_img : str
        The path to the ProMod3 apptainer/singularity image. Follow the instructions at
        https://openstructure.org/promod3/3.4/container/singularity/ to obtain this image.
    minimize : bool
        Whether to minimize the model after building it.
    build_sidechains : bool
        Whether to build sidechains after building the model.
    merge_distance : float
        The distance to merge fragments at.

    Returns
    -------
    modeled_segment : Molecule
        The modeled segment.

    Examples
    --------
    >>> from moleculekit.molecule import Molecule
    >>> from moleculekit.tools.modelling import model_gaps
    >>> mol = Molecule("5VQ6")
    >>> sequence = "HMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKSDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEK"
    >>> res = model_gaps(mol, sequence, "0", "./promod.img")
    """
    try:
        from Bio import pairwise2
        from Bio.Align import substitution_matrices
    except ImportError:
        raise ImportError(
            "You need to install the biopython package to use this function. Install it with `conda install biopython`."
        )

    promod_img = os.path.abspath(promod_img)

    blosum62 = substitution_matrices.load("BLOSUM62")

    apptainer = find_executable("apptainer")
    if not apptainer:
        apptainer = find_executable("singularity")
    if not apptainer:
        raise RuntimeError(
            "Could not find apptainer or singularity. Please install one of them."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        pdbfile = os.path.join(tmpdir, "input.pdb")
        mol_seg = mol.copy(sel=f"segid {segid}")
        mol_seg.write(pdbfile)

        molseq = mol.sequence()[segid]

        # -11 is gap creation penalty. -1 is gap extension penalty. Taken from https://www.arabidopsis.org/Blast/BLASToptions.jsp BLASTP options
        alignments = pairwise2.align.globalds(sequence, molseq, blosum62, -11.0, -1.0)
        # elif segment_type == "nucleic":
        #     alignments = pairwise2.align.globalxx(sequence, molseq)

        print(alignments[0])

        fastafile = os.path.join(tmpdir, "input.fasta")
        with open(fastafile, "w") as f:
            # Need to add gaps to sequence
            f.write(f">REFERENCE\n{sequence}\n")
            f.write(f">{segid}\n{alignments[0].seqB}")

        runpy = os.path.join(tmpdir, "run.py")
        with open(runpy, "w") as f:
            f.write(
                CODE.format(
                    minimize=minimize,
                    build_sidechains=build_sidechains,
                    merge_distance=merge_distance,
                )
            )

        run([apptainer, "run", "--app", "OST", promod_img, runpy], cwd=tmpdir)
        outfile = os.path.join(tmpdir, "model.pdb")
        if not os.path.exists(outfile):
            raise RuntimeError("Model could not be generated. Please check input.")

        modeled_segment = Molecule(outfile)

    return modeled_segment
