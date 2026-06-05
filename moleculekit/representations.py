from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule


class Representations:
    """Class that stores representations for Molecule.

    Parameters
    ----------
    mol : Molecule
        The Molecule object for which the representations are stored.

    Examples
    --------
    >>> from moleculekit.molecule import Molecule
    >>> mol = tryp.copy()
    >>> mol.reps.add('protein', 'NewCartoon')
    >>> print(mol.reps)                     # doctest: +NORMALIZE_WHITESPACE
    rep 0: sel='protein', style='NewCartoon', color='Name'
    >>> mol.view() # doctest: +SKIP
    >>> mol.reps.remove() # doctest: +SKIP
    """

    def __init__(self, mol: "Molecule"):
        self.replist = []
        self._mol = mol
        return

    def append(self, reps: "Representations"):
        """Append the representations of another Representations object.

        Parameters
        ----------
        reps : :class:`Representations` object
            The Representations object whose representations will be appended
            to this one.

        Raises
        ------
        RuntimeError
            If `reps` is not a Representations object.
        """
        if not isinstance(reps, Representations):
            raise RuntimeError("You can only append Representations objects.")
        self.replist += reps.replist

    def add(
        self,
        sel: str | np.ndarray | None = None,
        style: str | None = None,
        color: "str | int | None" = None,
        frames: list | None = None,
        opacity: float | None = None,
    ):
        """Adds a new representation for Molecule.

        Parameters
        ----------
        sel : str or np.ndarray
            Atom selection (string, boolean mask, or integer index array) for the representation.
            See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
        style : str
            Representation style. See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node55.html>`__.
        color : str or int
            Coloring mode (str) or ColorID (int).
            See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node85.html>`__.
        frames : list
            List of frames to visualize with this representation. If None it will visualize the current frame only.
        opacity : float
            Opacity of the representation. 0 is fully transparent and 1 is fully opaque.
        """
        self.replist.append(_Representation(sel, style, color, frames, opacity))

    def remove(self, index: int | None = None):
        """Removed one or all representations.

        Parameters
        ----------
        index : int
            The index of the representation to delete. If none is given it deletes all.
        """
        if index is None:
            self.replist = []
        else:
            del self.replist[index]

    def list(self):
        """Print all currently stored representations.

        Prints, for each representation, its index, atom selection, style and
        color. Equivalent to printing the Representations object directly.
        """
        print(self)

    def __str__(self):
        s = ""
        for i, r in enumerate(self.replist):
            s += f"rep {i}: sel='{r.sel}', style='{r.style}', color='{r.color}'\n"
        return s

    def _translateNGL(self, rep):
        styletrans = {
            "newcartoon": "cartoon",
            "licorice": "hyperball",
            "lines": "line",
            "vdw": "spacefill",
            "cpk": "ball+stick",
        }
        colortrans = {
            "name": "element",
            "index": "residueindex",
            "chain": "chainindex",
            "secondary structure": "sstruc",
            "colorid": "color",
        }
        hexcolors = {
            0: "#0000ff",
            1: "#ff0000",
            2: "#333333",
            3: "#ff6600",
            4: "#ffff00",
            5: "#4c4d00",
            6: "#b2b2cc",
            7: "#33cc33",
            8: "#ffffff",
            9: "#ff3399",
            10: "#33ccff",
        }
        try:
            selidx = "@" + ",".join(
                map(str, self._mol.atomselect(rep.sel, indexes=True))
            )
        except Exception:
            return None
        if rep.style.lower() in styletrans:
            style = styletrans[rep.style.lower()]
        else:
            style = rep.style
        if isinstance(rep.color, int):
            color = hexcolors[rep.color]
        elif rep.color.lower() in colortrans:
            color = colortrans[rep.color.lower()]
        else:
            color = rep.color
        return _Representation(sel=selidx, style=style, color=color)

    def _translateMolstar(self, rep):
        """Translate a VMD-flavored representation to a plain dict for the
        inline molstar scene IR: resolved atom indices, an MVS rep type, and a
        color (a {"theme": name} dict or a uniform hex/SVG string). Returns
        None if the selection matches no atoms."""
        styletrans = {
            "newcartoon": "cartoon",
            "cartoon": "cartoon",
            "licorice": "ball_and_stick",
            "cpk": "ball_and_stick",
            "vdw": "spacefill",
            "lines": "line",
        }
        themetrans = {
            "name": "element-symbol",
            "element": "element-symbol",
            "chain": "chain-id",
            "secondary structure": "secondary-structure",
            "resname": "residue-name",
            "index": "residue-id",
        }
        hexcolors = {
            0: "#0000ff", 1: "#ff0000", 2: "#333333", 3: "#ff6600",
            4: "#ffff00", 5: "#4c4d00", 6: "#b2b2cc", 7: "#33cc33",
            8: "#ffffff", 9: "#ff3399", 10: "#33ccff",
        }
        indices = [int(i) for i in self._mol.atomselect(rep.sel, indexes=True)]
        if not indices:
            return None
        style = styletrans.get(rep.style.lower(), "ball_and_stick")
        if isinstance(rep.color, int):
            color = hexcolors.get(rep.color, "#808080")
        elif rep.color.lower() in themetrans:
            color = {"theme": themetrans[rep.color.lower()]}
        else:
            color = rep.color
        out = {"atom_indices": indices, "type": style, "color": color}
        if rep.opacity is not None and rep.opacity != 1:
            out["opacity"] = float(rep.opacity)
        return out

    def _repsVMD(self, viewer):
        colortrans = {"secondary structure": "Structure"}
        if len(self.replist) > 0:
            viewer.send("mol delrep 0 top")
            for rep in self.replist:
                if isinstance(rep.color, str) and rep.color.lower() in colortrans:
                    color = colortrans[rep.color.lower()]
                else:
                    color = rep.color
                viewer.send(f"mol selection {rep.sel}")
                viewer.send(f"mol representation {rep.style}")
                if isinstance(rep.color, str) and not rep.color.isnumeric():
                    viewer.send(f"mol color {color}")
                else:
                    viewer.send(f"mol color ColorID {color}")

                viewer.send("mol addrep top")

    def _repsNGL(self, viewer):
        if len(self.replist) > 0:
            reps = []
            for r in self.replist:
                r2 = self._translateNGL(r)
                if r2 is not None:
                    reps.append(
                        {
                            "type": r2.style,
                            "params": {"sele": r2.sel, "color": r2.color},
                        }
                    )
            if reps != []:
                viewer.representations = reps


class _Representation:
    """Class that stores a representation for Molecule

    Parameters
    ----------
    sel : str
        Atom selection for the representation.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    style : str
        Representation style. See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node55.html>`__.
    color : str or int
        Coloring mode (str) or ColorID (int).
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node85.html>`__.
    frames : list
        List of frames to visualize with this representation. If None it will visualize the current frame only.
    opacity : float
        Opacity of the representation. 0 is fully transparent and 1 is fully opaque.

    Examples
    --------
    >>> r = _Representation(sel='protein', style='NewCartoon', color='Index')
    >>> r = _Representation(sel='resname MOL', style='Licorice')
    >>> r = _Representation(sel='ions', style='VDW', color=1)
    """

    def __init__(self, sel=None, style=None, color=None, frames=None, opacity=None):
        self.sel = "all" if sel is None else sel
        self.style = "Lines" if style is None else style
        self.color = "Name" if color is None else color
        self.frames = frames
        self.opacity = 1 if opacity is None else opacity
