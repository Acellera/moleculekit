# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import moleculekit.vmdviewer
import io
from moleculekit.vmdviewer import getCurrentViewer
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule


class VMDGraphicObject(object):
    """A superclass from which VMD graphic objects (e.g. the convex hull) inherit.

    Subclasses build a tcl script that draws a graphic primitive (a box,
    sphere, cylinder, convex hull, isosurface, ...) and send it to the current
    VMD viewer. Each object is assigned a unique id so that it can later be
    removed with the :meth:`delete` method.
    """

    counter = 1

    def __init__(self, data):
        """Create a graphic object. Not meant to be called directly by the user.

        Parameters
        ----------
        data : object
            The underlying data used to build the graphic (e.g. coordinates,
            a box definition or a convex hull). Stored on the instance for
            reference.
        """
        self.n = self.counter
        self.data = data
        self.script = io.StringIO()
        self.valid = True
        VMDGraphicObject.counter += 1

    def delete(self):
        """Undisplay and delete a graphic object."""
        if not self.valid:
            raise Exception("The object has been deleted already.")

        n = self.n
        vmd = moleculekit.vmdviewer.getCurrentViewer()
        cmd = """
            set htmd_tmp $htmd_graphics_mol({0:d})
            foreach i $htmd_graphics({0:d}) {{ graphics $htmd_tmp delete $i }}
            unset htmd_graphics({0:d})
            unset htmd_graphics_mol({0:d})
        """.format(
            n
        )
        vmd.send(cmd)
        self.valid = False

    def _remember(self, s):
        # We can't get data back from VMD, so let it remember what's to be deleted
        n = self.n
        self.script.write(f"lappend htmd_graphics({n:d}) [{s:s}]\n")

    @staticmethod
    def tq(v: np.ndarray) -> str:
        """Quote a numpy 3-vector as a tcl list.

        Parameters
        ----------
        v : np.ndarray
            The vector to format.

        Returns
        -------
        quoted : str
            The vector formatted as a brace-quoted tcl list, e.g. ``"{ 1 2 3 }"``.
        """
        return "{ " + str(v).strip("[]") + " }"


class VMDConvexHull(VMDGraphicObject):
    def __init__(self, mol: "Molecule", style: str = "", preamble: str = "", solid: bool = False):
        """Display the convex hull of the given molecule.

        For preamble and color, see http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node129.html

        The function returns an instance of VMDGraphicsObject. To delete it, use the delete() method.

        Parameters
        ----------
        mol: :class:`Molecule <moleculekit.molecule.Molecule>` object
            The object of which to show the hull (only 1 frame)
        style: str
            Style for wireframe lines
        preamble: str
            Commands (material, color) to be prefixed to the output.
            E.g.: "draw color red; graphics top materials on; graphics top material Transparent".
            Note that this affects later preamble-less commands.
        solid: bool
            Solid or wireframe

        Examples
        --------
        >>> from moleculekit.vmdgraphics import VMDConvexHull
        >>> m=Molecule("3PTB")
        >>> m.view()
        >>> mf=m.copy()
        >>> _ = mf.filter("protein ")
        >>> gh=VMDConvexHull(mf)  # doctest: +ELLIPSIS
        >>> gh.delete()
        """

        from scipy.spatial import ConvexHull

        if mol.coords.shape[2] != 1:
            raise Exception("Only one frame is supported")

        cc = mol.coords[:, :, 0]
        hull = ConvexHull(cc)

        super().__init__(hull)
        self.script.write(preamble + "\n")

        for i in range(hull.nsimplex):
            v1, v2, v3 = hull.simplices[i, :]
            c1s = VMDGraphicObject.tq(cc[v1, :])
            c2s = VMDGraphicObject.tq(cc[v2, :])
            c3s = VMDGraphicObject.tq(cc[v3, :])
            if solid:
                self._remember(f"draw triangle {c1s} {c2s} {c3s}")
            else:
                self._remember(f"draw line {c1s} {c2s} {style}")
                self._remember(f"draw line {c1s} {c3s} {style}")
                self._remember(f"draw line {c2s} {c3s} {style}")
                self.script.write("\n")

        self.script.write(f"set htmd_graphics_mol({self.n}) [molinfo top]")
        cmd = self.script.getvalue()
        vmd = getCurrentViewer()
        vmd.send(cmd)


class VMDBox(VMDGraphicObject):
    def __init__(self, box, color: str = "red"):
        """Displays a box in VMD as lines of its edges.

        The function returns an instance of VMDGraphicsObject. To delete it, use the delete() method.

        Parameters
        ----------
        box : list
            The min and max positions of the edges. Given as [xmin, xmax, ymin, ymax, zmin, zmax]
        color : str
            Color of the lines
        """
        super().__init__(box)
        xmin, xmax, ymin, ymax, zmin, zmax = box
        mi = [xmin, ymin, zmin]
        ma = [xmax, ymax, zmax]

        self._remember("draw materials off")
        self._remember(f"draw color {color}")

        self._remember(
            f'draw line "{mi[0]} {mi[1]} {mi[2]}" "{ma[0]} {mi[1]} {mi[2]}"\n'
        )
        self._remember(
            f'draw line "{mi[0]} {mi[1]} {mi[2]}" "{mi[0]} {ma[1]} {mi[2]}"\n'
        )
        self._remember(
            f'draw line "{mi[0]} {mi[1]} {mi[2]}" "{mi[0]} {mi[1]} {ma[2]}"\n'
        )

        self._remember(
            f'draw line "{ma[0]} {mi[1]} {mi[2]}" "{ma[0]} {ma[1]} {mi[2]}"\n'
        )
        self._remember(
            f'draw line "{ma[0]} {mi[1]} {mi[2]}" "{ma[0]} {mi[1]} {ma[2]}"\n'
        )

        self._remember(
            f'draw line "{mi[0]} {ma[1]} {mi[2]}" "{ma[0]} {ma[1]} {mi[2]}"\n'
        )
        self._remember(
            f'draw line "{mi[0]} {ma[1]} {mi[2]}" "{mi[0]} {ma[1]} {ma[2]}"\n'
        )

        self._remember(
            f'draw line "{mi[0]} {mi[1]} {ma[2]}" "{ma[0]} {mi[1]} {ma[2]}"\n'
        )
        self._remember(
            f'draw line "{mi[0]} {mi[1]} {ma[2]}" "{mi[0]} {ma[1]} {ma[2]}"\n'
        )

        self._remember(
            f'draw line "{ma[0]} {ma[1]} {ma[2]}" "{ma[0]} {ma[1]} {mi[2]}"\n'
        )
        self._remember(
            f'draw line "{ma[0]} {ma[1]} {ma[2]}" "{mi[0]} {ma[1]} {ma[2]}"\n'
        )
        self._remember(
            f'draw line "{ma[0]} {ma[1]} {ma[2]}" "{ma[0]} {mi[1]} {ma[2]}"\n'
        )
        self.script.write("\n")
        self.script.write(f"set htmd_graphics_mol({self.n}) [molinfo top]")
        cmd = self.script.getvalue()
        vmd = getCurrentViewer()
        vmd.send(cmd)


class VMDSphere(VMDGraphicObject):
    def __init__(self, xyz, color: str = "red", radius: float = 1):
        """Displays a sphere in VMD.

        The function returns an instance of VMDGraphicsObject. To delete it, use the delete() method.

        Parameters
        ----------
        xyz : list
            The center of the sphere
        color : str
            Color of the sphere
        radius : float
            The radius of the sphere
        """
        super().__init__(xyz)
        # self._remember('draw materials off')
        self._remember(f"draw color {color}")

        self._remember(f'draw sphere "{xyz[0]} {xyz[1]} {xyz[2]}" radius {radius}\n')
        self.script.write("\n")
        self.script.write(f"set htmd_graphics_mol({self.n}) [molinfo top]")
        cmd = self.script.getvalue()
        vmd = getCurrentViewer()
        vmd.send(cmd)


class VMDCylinder(VMDGraphicObject):
    def __init__(self, start, end, color: str = "red", radius: float = 1):
        """Displays a cylinder in VMD.

        The function returns an instance of VMDGraphicsObject. To delete it, use the delete() method.

        Parameters
        ----------
        start : list
            The starting coordinates of the cylinder
        end : list
            The ending coordinates of the cylinder
        color : str
            Color of the cylinder
        radius : float
            The radius of the cylinder
        """
        super().__init__([start, end])
        # self._remember('draw materials off')
        self._remember(f"draw color {color}")
        self._remember(
            "draw cylinder {{ {} }} {{ {} }} radius {}\n".format(
                " ".join(map(str, start)), " ".join(map(str, end)), radius
            )
        )
        self.script.write("\n")
        self.script.write(f"set htmd_graphics_mol({self.n}) [molinfo top]")
        cmd = self.script.getvalue()
        vmd = getCurrentViewer()
        vmd.send(cmd)


class VMDText(VMDGraphicObject):
    def __init__(self, text: str, xyz, color: str = "red"):
        """Displays a text in VMD.

        The function returns an instance of VMDGraphicsObject. To delete it, use the delete() method.

        Parameters
        ----------
        text : str
            The text
        xyz : list
            The position of the text
        color : str
            Color of the text
        """
        super().__init__(xyz)
        self._remember("draw materials off")
        self._remember(f"draw color {color}")

        self._remember(f'draw text "{xyz[0]} {xyz[1]} {xyz[2]}" "{text}"\n')
        self.script.write("\n")
        self.script.write(f"set htmd_graphics_mol({self.n}) [molinfo top]")
        cmd = self.script.getvalue()
        vmd = getCurrentViewer()
        vmd.send(cmd)


class VMDIsosurface(VMDGraphicObject):
    def __init__(
        self, arr: np.ndarray, vecMin: np.ndarray, vecRes: np.ndarray, color: int = 8, isovalue: float = 0.5, name: str | None = None, draw: str = "solid"
    ):
        """Displays an isosurface in VMD

        The function returns an instance of VMDGraphicsObject. To delete it, use the delete() method.

        Parameters
        ----------
        arr: np.ndarray
            3D array with volumetric data.
        vecMin: np.ndarray
            3D vector denoting the minimal corner of the grid
        vecRes: np.ndarray
            3D vector denoting the resolution of the grid in each dimension
        color: int
            The VMD ColorID to be used for the isosurface
        isovalue: float
            The isovalue at which to draw the isosurface
        name: str
            A name for the representation
        draw: str ('solid', 'wireframe')
            Drawing style for the isosurface
        """
        super().__init__(arr)
        from moleculekit.util import tempname, writeVoxels
        import os

        filename = tempname(suffix=".cube")
        writeVoxels(arr, filename, vecMin, vecRes)

        vmd = getCurrentViewer()

        drawmapping = {"solid": 0, "wireframe": 1}

        vmd.send(
            "mol new {} type cube first 0 last -1 step 1 waitfor 1 volsets {{0 }}".format(
                filename
            )
        )
        vmd.send(
            f"mol modstyle top top Isosurface {isovalue} 0 2 {drawmapping[draw]} 2 1"
        )
        vmd.send(f"mol modcolor top top ColorID {color}")
        vmd.send(f"set htmd_graphics_mol({self.n}) [molinfo top]")
        if name is not None:
            vmd.send(f"mol rename top {{{name}}}")

        if os.path.exists(filename):
            os.unlink(filename)

    def delete(self):
        """Undisplay and delete the isosurface from the viewer."""
        vmd = getCurrentViewer()
        vmd.send(f"mol delete htmd_graphics_mol({self.n})")


class VMDLabels(VMDGraphicObject):
    count = 0

    def __init__(
        self,
        mol: "Molecule",
        selection: str | np.ndarray,
        molid: str = "top",
        textsize: float = 0.5,
        textcolor: str = "green",
    ):
        """Displays labels on atoms in VMD.

        The function returns an instance of VMDGraphicsObject. To delete it, use the delete() method.

        Parameters
        ----------
        mol : :class:`Molecule <moleculekit.molecule.Molecule>` object
            The molecule whose atoms are labelled
        selection : str or np.ndarray
            Atom selection (string, boolean mask, or integer index array) defining which atoms to label
        molid : str
            The VMD molecule id on which to draw the labels
        textsize : float
            The size of the label text
        textcolor : str
            The color of the label text

        Examples
        --------
        >>> from moleculekit.molecule import Molecule  # doctest: +ELLIPSIS
        >>> from moleculekit.vmdgraphics import VMDLabels
        >>> mol = Molecule('3ptb')
        >>> mol.view()
        >>> x = VMDLabels(mol, 'resid 40')
        >>> y = VMDLabels(mol, 'resid 50')
        >>> y.delete()
        >>> x.delete()
        """
        # TODO: After deleting labels it breaks. Better don't delete or fix the code
        super().__init__(None)
        idx = mol.atomselect(selection, indexes=True)
        cmd = """label textsize {s}
color Labels Atoms {c}
set molnum [molinfo {molid}]
set i {start}
foreach n [list {idx}] {{
    label add Atoms $molnum/$n
    label textformat Atoms $i {{%a}}
    incr i
}}""".format(
            s=textsize,
            c=textcolor,
            sel=selection,
            molid=molid,
            idx=" ".join(map(str, idx)),
            start=VMDLabels.count,
        )
        self.labelidx = list(range(VMDLabels.count, VMDLabels.count + len(idx)))
        VMDLabels.count += len(idx)
        vmd = getCurrentViewer()
        vmd.send(cmd)
        # print(cmd)

    def delete(self):
        """Undisplay and delete the atom labels from the viewer."""
        vmd = getCurrentViewer()
        for i in self.labelidx[::-1]:
            vmd.send(f"label delete Atoms {i}")
            VMDLabels.count -= 1
