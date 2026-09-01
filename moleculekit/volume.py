"""Volumetric data: densities, maps and grids, drawn alongside molecules.

A Volume is the non-atomic half of a figure: an electron density, a docking
grid, an electrostatic potential. It carries its own representations the way a
Molecule carries ``mol.reps``, so a scene of a protein and a density is a list
of two objects rather than one merged thing.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

#: Isosurface colour used when a representation names none.
DEFAULT_VOLUME_COLOR = "#3465a4"


class _VolumeRepresentation:
    """One isosurface of a volume.

    Parameters
    ----------
    isovalue : float
        The value the surface is drawn at, in the units of the data.
    color : str
        SVG colour name or ``#rrggbb``.
    opacity : float
        0 is fully transparent, 1 fully opaque.
    wireframe : bool
        Draw the surface as a mesh rather than solid.
    """

    def __init__(self, isovalue=None, color=None, opacity=None, wireframe=False):
        self.isovalue = isovalue
        self.color = DEFAULT_VOLUME_COLOR if color is None else color
        self.opacity = 1.0 if opacity is None else float(opacity)
        self.wireframe = bool(wireframe)


class VolumeRepresentations:
    """The isosurfaces drawn for a Volume.

    Mirrors :class:`moleculekit.representations.Representations`, so a volume
    is styled the same way a molecule is.

    Parameters
    ----------
    vol : Volume
        The volume these representations belong to.

    Examples
    --------
    >>> vol = Volume("density.cube")                    # doctest: +SKIP
    >>> vol.reps.add(isovalue=0.05, color="#66ccff", opacity=0.6)  # doctest: +SKIP
    >>> vol.reps.add(isovalue=0.15, color="#ff6600")    # doctest: +SKIP
    """

    def __init__(self, vol: "Volume"):
        self.replist = []
        self._vol = vol

    def add(self, isovalue=None, color=None, opacity=None, wireframe=False):
        """Add an isosurface.

        Parameters
        ----------
        isovalue : float
            The value to draw the surface at, in the units of the data. None
            picks a value from the data itself, see :meth:`Volume.suggest_isovalue`.
        color : str
            SVG colour name or ``#rrggbb``.
        opacity : float
            0 is fully transparent, 1 fully opaque.
        wireframe : bool
            Draw a mesh rather than a solid surface.
        """
        if isovalue is None:
            isovalue = self._vol.suggest_isovalue()
        self.replist.append(_VolumeRepresentation(isovalue, color, opacity, wireframe))

    def remove(self, index: int | None = None):
        """Remove one isosurface, or all of them.

        Parameters
        ----------
        index : int or None
            Which to remove. None removes all.
        """
        if index is None:
            self.replist = []
        else:
            del self.replist[index]

    def __str__(self):
        out = ""
        for i, r in enumerate(self.replist):
            out += (
                f"rep {i}: isovalue={r.isovalue}, color='{r.color}', "
                f"opacity={r.opacity}, wireframe={r.wireframe}\n"
            )
        return out

    def __repr__(self):
        return self.__str__()


class Volume:
    """Volumetric data on a regular grid, drawable beside molecules.

    Parameters
    ----------
    filename : str or None
        A ``.cube`` or rDock ``.grd`` file. None builds an empty Volume to
        fill in from arrays.
    data : np.ndarray or None
        A 3D array, when building from memory rather than a file.
    origin : array_like or None
        Cartesian coordinates of the first grid point, in Angstrom, which is
        what a cube file stores.
    spacing : array_like or None
        Grid step along x, y and z, in Angstrom.

    Examples
    --------
    >>> vol = Volume("density.cube")                    # doctest: +SKIP
    >>> vol.reps.add(isovalue=0.05, color="#66ccff", opacity=0.6)  # doctest: +SKIP
    >>> render([mol, vol], "figure.png")                # doctest: +SKIP
    """

    def __init__(self, filename=None, data=None, origin=None, spacing=None):
        self.data = None if data is None else np.asarray(data, dtype=np.float32)
        self.origin = None if origin is None else np.asarray(origin, dtype=np.float64)
        self.spacing = None if spacing is None else np.asarray(spacing, dtype=np.float64)
        self.viewname = None
        self.reps = VolumeRepresentations(self)
        if filename is not None:
            self.read(filename)

    def read(self, filename: str):
        """Read volumetric data from a file.

        Parameters
        ----------
        filename : str
            A ``.cube`` or rDock ``.grd`` file.

        Raises
        ------
        RuntimeError
            If the extension is not one this can read, or if a ``.grd`` file
            describes a grid this cannot represent.
        """
        extension = os.path.splitext(filename)[1].lower()
        if extension == ".cube":
            from moleculekit.util import readCube

            data, meta = readCube(filename)
            # readCube reports the grid in Bohr, as the format stores it.
            bohr = 0.52917725
            self.data = np.asarray(data, dtype=np.float32)
            self.origin = np.asarray(list(meta["org"]), dtype=np.float64) * bohr
            self.spacing = np.array(
                # Each axis vector's own component: the step along that axis.
                [
                    np.asarray(list(meta[f"{axis}vec"]))[i] * bohr
                    for i, axis in enumerate("xyz")
                ],
                dtype=np.float64,
            )
        elif extension == ".grd":
            self._read_grd(filename)
        else:
            raise RuntimeError(
                f"Cannot read {extension!r} as volumetric data. "
                "Supported: .cube, .grd"
            )
        self.viewname = os.path.basename(filename)

    @property
    def shape(self):
        """The grid dimensions."""
        return None if self.data is None else self.data.shape

    def suggest_isovalue(self, quantile: float = 0.999) -> float:
        """A value worth drawing a surface at, taken from the data.

        A density's useful contour depends on its units and normalisation, so
        rather than guess a constant this picks a high quantile: the surface
        then encloses the densest part of the map whatever the scale.

        Parameters
        ----------
        quantile : float, optional
            Fraction of grid points to leave outside the surface.

        Returns
        -------
        isovalue : float
            The suggested value.
        """
        return float(np.quantile(self.data, quantile))

    def _read_grd(self, filename: str):
        """Read an rDock grid.

        The header is five lines: a title, the Fortran format the values are
        written in, the unit cell, the cell's grid sampling, and then the axis
        order followed by the first and last grid index along each axis. The
        indices are absolute, so they are what places the grid in space.

        Parameters
        ----------
        filename : str
            An rDock ``.grd`` file.

        Raises
        ------
        RuntimeError
            If the cell is not orthogonal, or the values are not stored with
            x varying fastest: either would put the data somewhere other than
            where this reads it.
        """
        with open(filename) as fh:
            lines = fh.read().split("\n")

        cell = np.array(lines[2].split(), dtype=np.float64)
        if not np.allclose(cell[3:], 90.0):
            raise RuntimeError(
                f"{filename} has cell angles {cell[3:].tolist()}. Only "
                "orthogonal grids are supported."
            )
        sampling = np.array(lines[3].split(), dtype=int)
        header = np.array(lines[4].split(), dtype=int)
        if header[0] != 1:
            raise RuntimeError(
                f"{filename} stores its values in axis order {header[0]}, and "
                "only 1 (x fastest) is supported."
            )
        first, last = header[1::2], header[2::2]

        self.spacing = cell[:3] / sampling
        # Absolute grid indices, so the first point's index times the step is
        # where the grid sits.
        self.origin = first * self.spacing
        values = np.fromstring(" ".join(lines[5:]), dtype=np.float32, sep=" ")
        # x fastest, as the axis order says, and our own arrays are [x][y][z].
        extent = last - first + 1
        self.data = values.reshape(extent[::-1]).transpose(2, 1, 0)

    def to_ccp4(self) -> bytes:
        """Serialise to a CCP4/MRC map, the format the viewer parses.

        Returns
        -------
        ccp4 : bytes
            The volume as a CCP4 map file's contents.
        """
        import struct

        nx, ny, nz = self.data.shape
        sx, sy, sz = self.spacing
        head = bytearray(1024)

        def put_i(word, *values):
            struct.pack_into(f"<{len(values)}i", head, word * 4, *values)

        def put_f(word, *values):
            struct.pack_into(f"<{len(values)}f", head, word * 4, *values)

        put_i(0, nx, ny, nz)  # sampled extent, fast axis first
        put_i(3, 2)  # MODE 2: float32 values
        # The unit cell is one voxel larger than the sampled extent: an extent
        # that fills its cell exactly is what marks a map periodic, and the
        # viewer then wraps the isosurface around the box edges. A density or a
        # docking grid is not a crystal.
        put_i(7, nx + 1, ny + 1, nz + 1)
        put_f(10, (nx + 1) * sx, (ny + 1) * sy, (nz + 1) * sz)
        put_f(13, 90.0, 90.0, 90.0)  # cell angles
        put_i(16, 1, 2, 3)  # axis order: x fast, y medium, z slow
        put_f(19, self.data.min(), self.data.max(), self.data.mean())
        put_i(22, 1)  # spacegroup P 1
        put_i(27, 20140)  # MRC2014 format version
        # The origin words, rather than the integer NCSTART/NRSTART/NSSTART
        # grid offsets: a grid's first point need not sit a whole number of
        # voxels away from the Cartesian origin.
        put_f(49, *self.origin)
        head[52 * 4 : 52 * 4 + 4] = b"MAP "
        head[53 * 4 : 53 * 4 + 4] = bytes((0x44, 0x41, 0x00, 0x00))  # little endian
        put_f(54, self.data.std())

        # x fastest, as the axis order above promises: our array is [x][y][z].
        values = np.ascontiguousarray(self.data.transpose(2, 1, 0), dtype="<f4")
        return bytes(head) + values.tobytes()

    def to_cube(self) -> str:
        """Serialise to Gaussian cube text.

        Returns
        -------
        cube : str
            The volume as a cube file's contents.
        """
        import tempfile

        from moleculekit.util import writeCube

        path = tempfile.mktemp(suffix=".cube")
        try:
            # writeCube takes the grid's minimal corner and writes the first
            # grid point, half a voxel further along. origin here is that first
            # grid point, as the format and readCube both mean it, so the shift
            # is undone to keep a read/write round trip in the same place.
            writeCube(self.data, path, self.origin - 0.5 * self.spacing, self.spacing)
            with open(path) as fh:
                return fh.read()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def __str__(self):
        if self.data is None:
            return "Volume with no data"
        return (
            f"Volume {self.shape[0]}x{self.shape[1]}x{self.shape[2]} "
            f"spacing {np.round(self.spacing, 3).tolist()} "
            f"origin {np.round(self.origin, 3).tolist()}"
        )

    def __repr__(self):
        return self.__str__()
