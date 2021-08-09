# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import logging
import unittest
from moleculekit.util import sequenceID
import tempfile
import numpy as np
import os
from moleculekit.molecule import Molecule

try:
    from pdb2pqr.main import main_driver, build_main_parser
except ImportError:
    raise ImportError(
        "pdb2pqr not installed. To use the molecule preparation features please do `conda install pdb2pqr -c conda-forge`"
    )


logger = logging.getLogger(__name__)


def _selToHoldList(mol, sel):
    ret = None
    if sel:
        tx = mol.copy()
        tx.filter(sel)
        ret = [
            list(x)
            for x in set(tuple(x) for x in zip(tx.resid, tx.chain, tx.insertion))
        ]
    return ret


def _fixupWaterNames(mol):
    """Rename WAT / OW HW HW atoms as O H1 H2"""
    mol.set("name", "O", sel="resname WAT and name OW")
    mol.set("name", "H1", sel="resname WAT and name HW and serial % 2 == 0")
    mol.set("name", "H2", sel="resname WAT and name HW and serial % 2 == 1")


def _warnIfContainsDUM(mol):
    """Warn if any DUM atom is there"""
    if any(mol.atomselect("resname DUM")):
        logger.warning(
            "OPM's DUM residues must be filtered out before preparation. Continuing, but crash likely."
        )


def _checkChainAndSegid(mol, _loggerLevel):
    from moleculekit.util import sequenceID
    import string

    emptychains = mol.chain == ""
    emptysegids = mol.segid == ""

    if np.all(emptychains) and np.all(emptysegids):
        raise RuntimeError(
            "No chains or segments defined in Molecule.chain / Molecule.segid. Please assign either to continue with preparation."
        )

    if np.all(emptychains) and np.any(~emptysegids):
        logger.info(
            "No chains defined in Molecule. Using segment IDs as chains for protein preparation."
        )
        mol = mol.copy()
        chain_alphabet = np.array(
            list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
        )
        mol.chain[:] = chain_alphabet[sequenceID(mol.segid)]

    if np.any(~emptysegids) and np.any(~emptychains):
        chainseq = sequenceID(mol.chain)
        segidseq = sequenceID(mol.segid)
        if not np.array_equal(chainseq, segidseq):
            logger.warning(
                "Both chains and segments are defined in Molecule.chain / Molecule.segid, however they are inconsistent. "
                "Protein preparation will use the chain information."
            )

    if _loggerLevel is None or _loggerLevel == "INFO":
        chainids = np.unique(mol.chain)
        if np.any([len(cc) > 1 for cc in chainids]):
            raise RuntimeError(
                "The chain field should only contain a single character."
            )

        print("\n---- Molecule chain report ----")
        for c in chainids:
            chainatoms = np.where(mol.chain == c)[0]
            firstatom = chainatoms[0]
            lastatom = chainatoms[-1]
            print(f"Chain {c}:")
            print(
                f"    First residue: {mol.resname[firstatom]}:{mol.resid[firstatom]}:{mol.insertion[firstatom]}"
            )
            print(
                f"    Final residue: {mol.resname[lastatom]}:{mol.resid[lastatom]}:{mol.insertion[lastatom]}"
            )
        print("---- End of chain report ----\n")

    return mol


def proteinPrepare(
    mol_in,
    titration=True,
    ligmol2=None,
    pH=7.0,
    verbose=True,
    returnDetails=False,
    hydrophobicThickness=None,
    holdSelection=None,
    plotpka=None,
    _loggerLevel="ERROR",
):
    """A system preparation wizard for HTMD.

    Returns a Molecule object, where residues have been renamed to follow
    internal conventions on protonation (below). Coordinates are changed to
    optimize the H-bonding network. This should be roughly equivalent to mdweb and Maestro's
    preparation wizard.

    The following residue names are used in the returned molecule:

    === ===============================
    ASH Neutral ASP
    CYX SS-bonded CYS
    CYM Negative CYS
    GLH Neutral GLU
    HIP Positive HIS
    HID Neutral HIS, proton HD1 present
    HIE Neutral HIS, proton HE2 present
    LYN Neutral LYS
    TYM Negative TYR
    AR0 Neutral ARG
    === ===============================

    ========= ======= =========
    Charge +1 Neutral Charge -1
    ========= ======= =========
    -         ASH     ASP
    -         CYS     CYM
    -         GLH     GLU
    HIP       HID/HIE -
    LYS       LYN     -
    -         TYR     TYM
    ARG       AR0     -
    ========= ======= =========

    A detailed table about the residues modified is returned (as a second return value) when
    returnDetails is True (see PreparationData object).

    If hydrophobicThickness is set to a positive value 2*h, a warning is produced for titratable residues
    having -h<z<h and are buried in the protein by less than 75%. Note that the heuristic for the
    detection of membrane-exposed residues is very crude; the "buried fraction" computation
    (from propKa) is approximate; also, in the presence of cavities,
    residues may be solvent-exposed independently from their z location.


    Notes
    -----
    In case of problems, exclude water and other dummy atoms.

    Features:
     - assign protonation states via propKa
     - flip residues to optimize H-bonding network
     - debump collisions
     - fill-in missing atoms, e.g. hydrogen atoms


    Parameters
    ----------
    mol_in : moleculekit.molecule.Molecule
        the object to be optimized
    titration : bool
        If set to True it will use propka to set the titration state of residues. Otherwise it will just add and optimize the hydrogens.
    pH : float
        pH to decide titration
    verbose : bool
        verbosity
    returnDetails : bool
        whether to return just the prepared Molecule (False, default) or a molecule *and* a ResidueInfo
        object including computed properties
    hydrophobicThickness : float
        the thickness of the membrane in which the protein is embedded, or None if globular protein.
        Used to provide a warning about membrane-exposed residues.
    holdSelection : str
        (Untested) Atom selection to be excluded from optimization.

    Returns
    -------
    mol_out : Molecule
        the molecule titrated and optimized. The molecule object contains an additional attribute,
    resData : PreparationData
        a table of residues with the corresponding protonation states, pKas, and other information


    Examples
    --------
    >>> tryp = Molecule('3PTB')

    >>> tryp_op, prepData = proteinPrepare(tryp, returnDetails=True)
    >>> tryp_op.write('proteinpreparation-test-main-ph-7.pdb')
    >>> prepData.data.to_excel("/tmp/tryp-report.xlsx")
    >>> prepData                                                        # doctest: +NORMALIZE_WHITESPACE
    PreparationData object about 290 residues.
    Unparametrized residue names: CA, BEN
    Please find the full info in the .data property, e.g.:
      resname  resid insertion chain       pKa protonation flipped     buried
    0     ILE     16               A       NaN         ILE     NaN        NaN
    1     VAL     17               A       NaN         VAL     NaN        NaN
    2     GLY     18               A       NaN         GLY     NaN        NaN
    3     GLY     19               A       NaN         GLY     NaN        NaN
    4     TYR     20               A  9.590845         TYR     NaN  14.642857
     . . .
    >>> x_HIE91_ND1 = tryp_op.get("coords","resid 91 and  name ND1")
    >>> x_SER93_H =   tryp_op.get("coords","resid 93 and  name H")
    >>> len(x_SER93_H) == 3
    True
    >>> np.linalg.norm(x_HIE91_ND1-x_SER93_H) < 3
    True

    >>> tryp_op = proteinPrepare(tryp, pH=1.0)
    >>> tryp_op.write('proteinpreparation-test-main-ph-1.pdb')

    >>> tryp_op = proteinPrepare(tryp, pH=14.0)
    >>> tryp_op.write('proteinpreparation-test-main-ph-14.pdb')

    >>> mol = Molecule("1r1j")
    >>> mo, prepData = proteinPrepare(mol, returnDetails=True)
    >>> prepData.missedLigands
    ['NAG', 'ZN', 'OIR']


    >>> mor = Molecule("4dkl")
    >>> mor.filter("protein and noh")
    >>> mor_opt, mor_data = proteinPrepare(mor, returnDetails=True,
    ...                                    hydrophobicThickness=32.0)
    >>> exposedRes = mor_data.data.membraneExposed
    >>> mor_data.data[exposedRes].to_excel("/tmp/mor_exposed_residues.xlsx")

    >>> im=Molecule("4bkj")
    >>> imo,imd=proteinPrepare(im,returnDetails=True)
    >>> imd.data.to_excel("/tmp/imatinib_report.xlsx")

    Notes
    -----
    Unsupported/To Do/To Check:
     - ligands
     - termini
     - nucleic acids
     - coupled titrating residues
     - Disulfide bridge detection (implemented but unused)
    """
    from pdb2pqr.config import VERSION

    old_level = logger.getEffectiveLevel()
    if not verbose:
        logger.setLevel(logging.WARNING)

    if _loggerLevel is not None:
        # logger.setLevel(_loggerLevel)
        logging.getLogger(f"PDB2PQR{VERSION}").setLevel(_loggerLevel)
        logging.getLogger(f"PDB2PQR{VERSION}").propagate = False
        logging.getLogger("pdb2pqr").setLevel(_loggerLevel)
        logging.getLogger("propka").setLevel(_loggerLevel)
    logger.debug("Starting.")

    _warnIfContainsDUM(mol_in)

    mol_in = _checkChainAndSegid(mol_in, _loggerLevel)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpin = os.path.join(tmpdir, "input.pdb")
        mol_in.write(tmpin)

        hlist = _selToHoldList(mol_in, holdSelection)
        if hlist:
            logger.warning(
                "The holdSelection option is untested and deprecated. Please use reprepare()"
            )

        pqrout = os.path.join(tmpdir, "out.pqr")
        pdbout = os.path.join(tmpdir, "out.pdb")
        parser = build_main_parser()
        args = [
            "--log-level",
            "WARNING",
            "--pdb-output",
            pdbout,
            "--with-ph",
            str(pH),
            "--ff",
            "PARSE",
            "--ffout",
            "AMBER",
            "--quiet",
            tmpin,
            pqrout,
        ]
        if titration:
            args = ["--titration-state-method", "propka"] + args
        if ligmol2:
            args = ["--ligand", ligmol2]
        args = parser.parse_args(args)

        missedres, pka_df, biomolecule = main_driver(args)
        mol_out = _biomolecule_to_molecule(biomolecule)
        # mol_out = Molecule(pdbout)

    # Diagnostics
    missedres = set([m.residue.name for m in missedres])
    for resn in missedres:
        logger.warning(f"The following residue has not been optimized: {resn}")

    # assert not np.any(
    #     mol_out.element == "X"
    # ), "pdb2pqr left some lonepair atoms in. report this issue to moleculekit github issue tracker"
    # mol_out.element = mol_out._guessMissingElements()

    mol_out.box = mol_in.box
    _fixupWaterNames(mol_out)

    df = _create_table(mol_in, mol_out, pka_df)

    _list_modifications(df)

    if titration:
        if plotpka is not None:
            _get_pka_plot(df, plotpka)
        _warn_pk_close_to_ph(df, pH)
        if hydrophobicThickness:
            # TODO: I think this only works if the protein is assumed aligned to the membrane and the membrane is centered at Z=0
            _warn_buried_residues(df, mol_out, hydrophobicThickness)

    # resData.warnIfTerminiSuspect()

    logger.setLevel(old_level)

    if returnDetails:
        return mol_out, df
    return mol_out


def _biomolecule_to_molecule(biomolecule):
    propmap = [
        ("type", "record"),
        ("serial", "serial"),
        ("name", "name"),
        ("alt_loc", "altloc"),
        ("res_name", "resname"),
        ("chain_id", "chain"),
        ("res_seq", "resid"),
        ("ins_code", "insertion"),
        ("occupancy", "occupancy"),
        ("temp_factor", "beta"),
        ("seg_id", "segid"),
        ("element", "element"),
        ("charge", "charge"),
    ]
    mol = Molecule().empty(len(biomolecule.atoms))
    mol.coords = np.zeros((mol.numAtoms, 3, 1), dtype=Molecule._dtypes["coords"])
    bonds = []
    for i, atom in enumerate(biomolecule.atoms):
        for pp1, pp2 in propmap:
            val = getattr(atom, pp1)
            if pp1 == "charge":
                val = float(val) if val != "" else 0
            mol.__dict__[pp2][i] = val
        mol.coords[i, :, 0] = [atom.x, atom.y, atom.z]
        bonds += [[i, x.serial - 1] for x in atom.bonds]

    guessedBonds = mol._guessBonds()
    bb = mol.atomselect("backbone", indexes=True)
    bb = guessedBonds[np.all(np.isin(guessedBonds, bb), axis=1)]
    mol.bonds = np.vstack(bb.tolist() + bonds)

    return mol


def _create_table(mol_in, mol_out, pka_df):
    import pandas as pd

    uqresid_in = sequenceID(
        (mol_in.resid, mol_in.insertion, mol_in.chain, mol_in.segid)
    )
    uqresid_out = sequenceID(
        (mol_out.resid, mol_out.insertion, mol_out.chain, mol_out.segid)
    )
    if uqresid_in.max() != uqresid_out.max():
        logger.warning("Residue numbering was changed in pdb2pqr!")
        return None

    data = []
    for idx in np.unique(uqresid_out):
        old_resn = mol_in.resname[uqresid_in == idx][0]
        sel = uqresid_out == idx
        resname = mol_out.resname[sel][0]
        resid = mol_out.resid[sel][0]
        insertion = mol_out.insertion[sel][0]
        chain = mol_out.chain[sel][0]
        segid = mol_out.segid[sel][0]
        curr_data = [old_resn, resname, resid, insertion, chain, segid]
        for propkadata in pka_df:
            if (
                propkadata["res_num"] == resid
                and propkadata["ins_code"].strip() == insertion.strip()
                and propkadata["chain_id"] == chain
            ):
                curr_data += [propkadata["pKa"], propkadata["buried"]]
                break
        data.append(curr_data)

    df = pd.DataFrame(
        data=data,
        columns=[
            "resname",
            "protonation",
            "resid",
            "insertion",
            "chain",
            "segid",
            "pKa",
            "buried",
        ],
    )
    return df


def _list_modifications(df):
    for _, row in df[df.resname != df.protonation].iterrows():
        if row.resname in ["HOH", "WAT"]:
            continue

        old_resn = row.resname
        new_resn = row.protonation
        ch = row.chain
        seg = row.segid
        rid = row.resid
        ins = row.insertion.strip()
        logger.info(
            f"Modified residue {ch}:{seg}:{rid}{ins} from {old_resn} to {new_resn}"
        )


def _warn_pk_close_to_ph(df, pH, tol=1.0):
    dubious = df[abs(df.pKa - pH) < tol]
    if len(dubious):
        logger.warning(
            f"Dubious protonation state: the pKa of {len(dubious)} residues is within {tol:.1f} units of pH {pH:.1f}."
        )
        for _, dr in dubious.iterrows():
            logger.warning(
                f"Dubious protonation state:    {dr.resname:4s} {dr.resid:>4d}{dr.insertion.strip()} {dr.chain:1s} (pKa={dr.pKa:5.2f})"
            )


def _warn_buried_residues(df, mol_out, hydrophobicThickness, maxBuried=0.75):
    ht = hydrophobicThickness / 2.0
    count = 0
    for _, row in df[df.buried > maxBuried].iterrows():
        sele = (
            (mol_out.chain == row.chain)
            & (mol_out.resid == row.resid)
            & (mol_out.insertion == row.insertion)
        )
        mean_z = np.mean(mol_out.coords[sele, 2, 0])
        if -ht < mean_z < ht:
            count += 1

    if count:
        logger.warning(
            f"Predictions for {count} residues may be incorrect because they are "
            + f"exposed to the membrane ({-ht:.1f}<z<{ht:.2f} and buried<{maxBuried:.1f}%)."
        )


def _get_pka_plot(df, outname, pH=7.4, figSizeX=13, dpk=1.0, font_size=12):
    """Internal function to build the protonation diagram"""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patheffects as PathEffects

    # Shading
    Xe = np.array([[1, 0], [1, 0]])

    # Shading colors http://matplotlib.org/examples/pylab_examples/custom_cmap.html
    neutral_grey = (0.7, 0.7, 0.7)
    my_red = (0.98, 0.41, 0.29)
    my_blue = (0.42, 0.68, 0.84)
    grey_red = LinearSegmentedColormap.from_list("grey_red", [neutral_grey, my_red])
    grey_blue = LinearSegmentedColormap.from_list("grey_blue", [neutral_grey, my_blue])
    eps = 0.01  # Tiny overprint to avoid very thin white lines
    outline = [PathEffects.withStroke(linewidth=2, foreground="w")]

    # Color for pk values
    pKa_color = "black"
    pKa_fontsize = 8
    dtxt = 0  # Displacement

    # Or we could change the figure size, which scales axes
    # http://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    plt.rc("font", family="monospace")
    plt.rc("font", size=font_size)  # controls default text sizes
    plt.rc("axes", titlesize=font_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=font_size)  # legend fontsize
    plt.rc("figure", titlesize=font_size)  # fontsize of the figure title

    # Constants
    acidicResidues = ["ASP", "GLU", "TYR", "C-"]

    df = df[df.pKa < 99]

    # Format residue labels
    labels = []
    pKas = []
    restypes = []
    for _, row in df.iterrows():
        dub = "(!)" if abs(row.pKa - pH) < dpk else ""
        labels.append(
            f"{dub} {row.chain}:{row.resid}{row.insertion.strip()}-{row.resname}"
        )
        pKas.append(row.pKa)
        restypes.append("neg" if row.resname in acidicResidues else "pos")

    N = len(df)
    xmin, xmax = xlim = 0, 14
    ymin, ymax = ylim = -1, N

    width = 0.8  # Of each band

    # So, arbitrarily, 40 residues are square
    sizePerBand = figSizeX * (N / 40)
    figsize = (figSizeX, sizePerBand)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, xlim=xlim, ylim=ylim, autoscale_on=False)

    ax.xaxis.tick_top()
    ax.set_xlabel("pKa")
    ax.xaxis.set_label_position("top")

    ax.yaxis.set_ticks(range(N))
    ax.yaxis.set_ticklabels(labels)
    ax.invert_yaxis()

    for i in range(N):
        left = xmin
        right = xmax
        top = i + width / 2
        bottom = i - width / 2
        pk = pKas[i]
        restype = restypes[i]

        if restype == "neg":
            ax.imshow(
                Xe * 0,
                interpolation="none",
                cmap=grey_blue,
                vmin=0,
                vmax=1,
                extent=(left, pk - dpk, bottom, top),
                alpha=1,
            )
            ax.imshow(
                np.fliplr(Xe),
                interpolation="bicubic",
                cmap=grey_blue,
                vmin=0,
                vmax=1,
                extent=(pk - dpk - eps, pk + dpk, bottom, top),
                alpha=1,
            )
            ax.imshow(
                1 + Xe * 0,
                interpolation="none",
                cmap=grey_blue,
                vmin=0,
                vmax=1,
                extent=(pk + dpk - eps, right, bottom, top),
                alpha=1,
            )
            ax.text(
                pk - dtxt,
                i,
                " {:.2f} ".format(pk),
                color=pKa_color,
                fontsize=pKa_fontsize,
                horizontalalignment="right",
                zorder=30,
                path_effects=outline,
                weight="bold",
            )
        else:
            ax.imshow(
                1 + Xe * 0,
                interpolation="none",
                cmap=grey_red,
                vmin=0,
                vmax=1,
                extent=(left, pk - dpk, bottom, top),
                alpha=1,
            )
            ax.imshow(
                Xe,
                interpolation="bicubic",
                cmap=grey_red,
                vmin=0,
                vmax=1,
                extent=(pk - dpk - eps, pk + dpk, bottom, top),
                alpha=1,
            )
            ax.imshow(
                Xe * 0,
                interpolation="none",
                cmap=grey_red,
                vmin=0,
                vmax=1,
                extent=(pk + dpk - eps, right, bottom, top),
                alpha=1,
            )
            ax.text(
                pk + dtxt,
                i,
                " {:.2f} ".format(pk),
                color=pKa_color,
                fontsize=pKa_fontsize,
                horizontalalignment="left",
                zorder=30,
                path_effects=outline,
                weight="bold",
            )
        ax.add_line(
            Line2D([pk, pk], [bottom, top], linewidth=3, color="white", zorder=2)
        )

        # ax.add_line(Line2D([pk,pk], [bottom,top], linewidth=3, color='blue'))

    # Shaded vertical band at pH
    ax.axvline(x=pH - dpk, linewidth=2, color="black", alpha=0.2, linestyle="dashed")
    ax.axvline(x=pH + dpk, linewidth=2, color="black", alpha=0.2, linestyle="dashed")
    ax.axvline(x=pH, linewidth=3, color="black", alpha=0.5)
    ax.text(
        pH - dpk,
        ymax,
        " 90% protonated",
        rotation=90,
        horizontalalignment="right",
        verticalalignment="bottom",
        style="italic",
        path_effects=outline,
    )
    ax.text(
        pH + dpk,
        ymax,
        " 10% protonated",
        rotation=90,
        horizontalalignment="left",
        verticalalignment="bottom",
        style="italic",
        path_effects=outline,
    )

    ax.set_aspect("auto")
    plt.savefig(outname, dpi=300)
    plt.close(fig)


class _TestPreparation(unittest.TestCase):
    def test_proteinPrepare(self):
        _, d = proteinPrepare(Molecule("3PTB"), returnDetails=True)
        assert d.protonation[d.resid == 40].iloc[0] == "HIE"
        assert d.protonation[d.resid == 57].iloc[0] == "HIP"
        assert d.protonation[d.resid == 91].iloc[0] == "HID"

    # @unittest.skipUnless(os.environ.get("HTMD_LONGTESTS") == "yes", "Too long")
    def test_proteinPrepareLong(self):
        from moleculekit.home import home
        from moleculekit.util import assertSameAsReferenceDir
        import tempfile

        pdbids = ["3PTB", "1A25", "1GZM", "1U5U"]
        for pdb in pdbids:
            mol = Molecule(pdb)
            mol.filter("protein")
            with tempfile.TemporaryDirectory() as tmpdir:
                mol_op, prepData = proteinPrepare(
                    mol, returnDetails=True, plotpka=os.path.join(tmpdir, "plot.png")
                )

                mol_op.write(os.path.join(tmpdir, f"{pdb}-prepared.pdb"))
                prepData.to_csv(
                    os.path.join(tmpdir, f"{pdb}-prepared.csv"), float_format="%.2f"
                )
                compareDir = home(dataDir=os.path.join("test-proteinprepare", pdb))
                assertSameAsReferenceDir(compareDir, tmpdir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
    import doctest

    doctest.testmod()
