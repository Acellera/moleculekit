# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import subprocess
import inspect
import threading
import platform
import queue
import time
import os
from moleculekit.util import string_to_tempfile
import numpy as np
import tempfile
import logging

logger = logging.getLogger(__name__)


_viewers = np.empty(0, dtype=object)


def _enqueue_output(obj, vmd, queue):
    while vmd.poll() is None:
        _ = vmd.stdout.readline()
    # print("READIN: [" + line.decode("ascii") + "]", end="" );
    # 			queue.put(line.decode("ascii"))
    # 	print( "Process finished" )
    vmd.stdout.close()
    obj.done = True


class VMD:
    """Please do not directly call this class constructor. Use the `viewer` or `getCurrentViewer` function instead.

    Parameters
    ----------
    vmd : str
    host : str
    dispdev : str
    """

    def __init__(self, vmd=None, host=None, dispdev="win"):
        self.done = False
        vmd = getVMDpath(vmd=vmd)

        args = [vmd]
        if host:
            args.append("--host")
            args.append(host)
        args.append("--dispdev")
        args.append(dispdev)
        self.vmd = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
            shell=False,
        )
        self.queue = queue.Queue()
        self.thread = threading.Thread(
            target=_enqueue_output, args=(self, self.vmd, self.queue)
        )
        self.thread.daemon = True
        self.thread.start()
        # 	self.vmd.stdin.write( b"chan configure stdout -buffering none\n" )
        time.sleep(2)
        self.send("display reposition 500 1000")
        self.send("display resize 800 800")
        self.send("menu main on")
        self.send("display depthcue off")
        self.send("axes location Off")
        # self.send('color Display Background white')
        self.send("display projection Orthographic")

    def send(self, command):
        """Send a tcl command to VMD

        Parameters
        ----------
        command : str
            The tcl command which to send
        """
        if self.done:
            return
        # print( command )
        fn = string_to_tempfile(command, "tcl")
        # 		print(fn)

        cc = f"if {{ [ catch {{ source {fn}}} ] }} {{ unlink {fn} }} else {{ unlink {fn} }}\n"
        self.vmd.stdin.write(cc.encode("ascii"))
        self.vmd.stdin.flush()

        while os.path.isfile(fn) and not self.done:
            time.sleep(0.01)

    def loadMol(self, mol, name=None):
        """Load a :class:`Molecule <moleculekit.molecule.Molecule>` object into VMD

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            The Molecule to load into VMD
        name : str
            The name to give to the Molecule
        """
        # Getting name of variable passed to method
        """
        caller = inspect.currentframe().f_back
        try:
            value = caller.f_locals[mol]
        except KeyError:
            value = caller.f_globals[mol]
        if name is None:
            name = value
        # -----------------------------------------
        """
        mol.view(name=name, viewerhandle=self)

    def rep(self, mode, sel="resname MOL", color=0):
        """Modify representations for the top molecule in VMD

        Parameters
        ----------
        mode : str ('protein', 'ligand')
            If set to 'protein', it will show a single conformation of the protein with sequence coloring
            If set to 'ligand', it will show all ligands in the specified `color` and the protein in teal
        sel : str
            Atom selection string for the ligand mode.
            See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
        color : int
            Color for the ligand. Use color numbers of VMD
        """
        if mode == "ligand":
            # Protein representation
            self.send("mol modstyle 0 top NewCartoon")
            self.send('mol modselect 0 top "protein"')
            # self.send('mol modcolor 0 top Index')
            # Ligand representation
            self.send("set rep [molinfo top get numreps]")
            self.send("mol color ColorID " + str(color))
            self.send("mol representation Lines 1.000000")
            self.send("mol selection " + sel)
            self.send("mol material Opaque")
            self.send("mol addrep top")
            self.send("set end [molinfo top get numframes]")
            self.send('mol drawframes top $rep "0:$end"')
        elif mode == "protein":
            # Protein representation
            self.send("mol modstyle 0 top NewCartoon")
            self.send('mol modselect 0 top "protein"')
            self.send("mol modcolor 0 top Index")
        else:
            raise ValueError(
                "Invalid mode. Choose between " "ligand" " and " "protein" ""
            )

    def completed(self):
        """Check if the viewer has been closed

        Returns
        -------
        comp : bool
            Returns True if the VMD viewer has closed
        """
        return self.done

    def copy(self):
        return None

    def render(
        self,
        outfile,
        renderer="TachyonInternal",
        resolution=None,
        aasamples=None,
        skylight=None,
        tachyon=None,
        convert=None,
        trim=False,
    ):
        """Renders the current VMD scene into a file.

        Parameters
        ----------
        outfile : str
            File to which to render image
        renderer : ('TachyonInternal', 'tachyon', 'snapshot')
            Which renderer to use
        resolution : tuple
            X,Y resolution of the output image i.e. (1920, 1080). Only used with the renderer='tachyon' option.
        aasamples : int
            Number of anti-aliasing samples. Only used with the renderer='tachyon' option.
        skylight : float
            Add a skylight. Only used with the renderer='tachyon' option.
        tachyon : str
            Path to tachyon renderer executable. Only used with the renderer='tachyon' option.
        convert : bool
            Attempts to convert the image to the datatype of the `outfile` extension
        trim : bool
            Trims the whitespace of the image
        """
        import shutil

        outfile = os.path.abspath(outfile)
        outname, ext = os.path.splitext(outfile)

        if (renderer.lower() != "tachyon") and (
            resolution is not None
            or aasamples is not None
            or skylight is not None
            or tachyon is not None
        ):
            raise AttributeError(
                "resolution, aasamples, skylight and tachyon parameters only accepted with the "
                "renderer='tachyon' option."
            )

        if renderer.lower() == "tachyon":
            tmpext = ".psd"
            if tachyon is None:
                tachyon = shutil.which("tachyon", mode=os.X_OK)
            if tachyon is None:
                raise FileNotFoundError(
                    "Could not find `tachyon` executable, or no execute permissions are given. Try using renderer='snapshot' instead."
                )
            rendercommand = f"render Tachyon {outname}"
        elif renderer.lower() == "snapshot":
            tmpext = ".tga"
            rendercommand = f"render snapshot {outname}{tmpext}"
        elif renderer.lower() == "tachyoninternal":
            tmpext = ".tga"
            rendercommand = f"render TachyonInternal {outname}{tmpext}"

        self.send(rendercommand)
        if renderer == "tachyon":
            os.system(
                f"{tachyon} -res {resolution[0]} {resolution[1]} -aasamples {aasamples} -add_skylight {skylight} {outname} -format PSD48 -o {outname}.psd"
            )
        logger.debug(rendercommand)
        if not os.path.exists(outname + tmpext):
            raise RuntimeError(
                f"Rendering failed to produce image with following command: {rendercommand}"
            )
        if os.path.exists(outname):
            os.remove(outname)

        if ext != tmpext:
            if convert is None:
                convert = shutil.which("convert", mode=os.X_OK)
            if convert is None:
                raise FileNotFoundError(
                    f"Could not find `convert` executable, or no execute permissions are given. You can find the temporary render file in {outname + tmpext}"
                )

            os.system(f"{convert} {outname}{tmpext} {outname}{ext}")
            if trim:
                os.system(f"{convert} {outname}{ext} -trim {outname}{ext}")
            os.remove(outname + tmpext)

    def movie_reps(self):
        """Adds some default settings and representations for creating movies with Tachyon in VMD"""
        render_conf = r"""display depthcue   on
display shadows on
display ambientocclusion on
display aoambient 0.920000
display aodirect 0.400000
display dof on
color Display {Background} white

proc clear_reps { } {
    set nreps [molinfo top get numreps]
    for {set i 0} { $i < $nreps } { incr i } {
        mol delrep 0 top
    }
}
clear_reps

mol representation VDW 1.000000 12.000000
mol color ColorID 8
mol selection {protein and noh}
mol material Opaque
mol addrep top

mol representation VDW 1.000000 12.000000
mol color Name
mol selection {not protein and not resname ACE NME and not ion and not water}
mol material HardPlastic
mol addrep top
"""
        self.send(render_conf)

    def close(self):
        self.send("exit")

    def __del__(self):
        self.close()


def getVMDpath(vmd=None):
    sys = platform.system()
    if not vmd:
        if sys == "Linux" or sys == "Darwin":
            vmd = os.path.join(os.path.dirname(inspect.getfile(VMD)), "vmd_wrapper")
        elif sys == "Windows":
            vmd = os.path.join(os.path.dirname(inspect.getfile(VMD)), "vmd_wrapper.bat")
        else:
            raise OSError("Don't know how to run VMD on platform [" + sys + "]")
    if (not vmd) or (not os.access(vmd, os.X_OK)):
        raise OSError("Cannot find VMD. Specify the location with 'vmd=' argument")
    return vmd


def getCurrentViewer(dispdev="win"):
    """Get the handle to the current molecular viewer

    Parameters
    ----------
    dispdev : str, ('win', 'text', 'cave', 'caveforms', 'none')
        Specify the type of graphical display to use. The possible display devices include:
        win: a standard graphics display window.
        text: do not provide any graphics display window.
        cave: use the CAVE virtual environment for display, forms are disabled.
        caveforms: use the CAVE virtual environment for display and with forms enabled. This is useful with -display machine:0 for remote display of the forms when the CAVE uses the local screen.

    Returns
    -------
    viewer : :class:`VMD` object
        Returns a handle to the current viewer
    """
    global _viewers
    # Check for dead viewers and remove them
    todrop = np.empty(0, dtype=int)
    for i in range(len(_viewers)):
        if _viewers[i].completed():
            todrop = np.append(todrop, i)
    _viewers = np.delete(_viewers, todrop)
    # Creating a new viewer if none exist or otherwise returning the last viewer
    if len(_viewers) == 0:
        _viewers = np.append(_viewers, VMD(dispdev=dispdev))
    return _viewers[-1]


def viewer(dispdev="win"):
    """Start a new molecular viewer

    Returns
    -------
    viewer : :class:`VMD` object
        Returns a handle to a new viewer
    """
    global _viewers
    # Check for dead viewers and remove them
    todrop = np.empty(0, dtype=int)
    for i in range(len(_viewers)):
        if _viewers[i].completed:
            todrop = np.append(todrop, i)
    _viewers = np.delete(_viewers, todrop)
    # Creating a new viewer
    handle = VMD(dispdev=dispdev)
    _viewers = np.append(_viewers, handle)
    return handle


def _tempfilename():
    return os.path.join(
        tempfile._get_default_tempdir(), next(tempfile._get_candidate_names())
    )
