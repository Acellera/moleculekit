from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from moleculekit.molecule import Molecule


#: Representation styles, and the Mol* representation each one draws as. The
#: names are VMD's, since that is the vocabulary ``mol.reps`` has always used.
MOLSTAR_STYLES = {
    "newcartoon": "cartoon",
    "cartoon": "cartoon",
    "licorice": "ball_and_stick",
    "cpk": "ball_and_stick",
    "vdw": "spacefill",
    "lines": "line",
    "surf": "molecular_surface",
    "quicksurf": "gaussian_surface",
    "points": "point",
    "labels": "label",
    "formalcharges": "formal_charge",
    # Mol*'s own names for the same things, so a representation can be written
    # in either vocabulary. Normalisation drops the hyphens, so "ball-and-stick"
    # and "ballandstick" arrive here the same.
    "ballandstick": "ball_and_stick",
    "spacefill": "spacefill",
    "line": "line",
    "molecularsurface": "molecular_surface",
    "gaussiansurface": "gaussian_surface",
    "point": "point",
    "label": "label",
    "atomlabel": "label",
    "putty": "putty",
    "backbone": "backbone",
    "ellipsoid": "ellipsoid",
}

#: The VMD representation each Mol* style name corresponds to. VMD's own names
#: are sent as written, so only the Mol* spellings need translating.
VMD_STYLES = {
    "ballandstick": "CPK",
    "spacefill": "VDW",
    "line": "Lines",
    "molecularsurface": "Surf",
    "gaussiansurface": "QuickSurf",
    "point": "Points",
}

#: The VMD coloring method each colouring mode corresponds to. Modes VMD has no
#: equivalent for (hydrophobicity, molecule type) are sent as written and VMD
#: rejects them, which is what it did before any of these were accepted here.
VMD_COLORS = {
    "secondarystructure": "Structure",
    "elementsymbol": "Name",
    "chainid": "Chain",
    "residuename": "ResName",
    "sequenceid": "ResID",
    "atomid": "Index",
    "uncertainty": "Beta",
}

#: Colouring modes, and the Mol* colour theme each one selects. Keys are
#: matched with spaces, hyphens and underscores removed, so "Secondary
#: Structure" and "SecondaryStructure" are the same mode. Our own docstrings
#: used the second spelling, which fell through to being read as a colour name
#: and drew a garbage uniform colour.
MOLSTAR_THEMES = {
    "name": "element-symbol",
    "element": "element-symbol",
    "chain": "chain-id",
    "secondarystructure": "secondary-structure",
    "resname": "residue-name",
    # Not residue-id, which is no theme Mol* knows: it fell back to the
    # default and coloured by chain instead, silently.
    "index": "sequence-id",
    "hydrophobicity": "hydrophobicity",
    "moleculetype": "molecule-type",
    "atomid": "atom-id",
    # Mol*'s own theme names. Several already match after normalisation
    # ("secondary-structure", "molecule-type", "atom-id"); these are the rest.
    "elementsymbol": "element-symbol",
    "chainid": "chain-id",
    "residuename": "residue-name",
    "sequenceid": "sequence-id",
    "elementindex": "element-index",
    "entityid": "entity-id",
    "polymerid": "polymer-id",
    "modelindex": "model-index",
    "structureindex": "structure-index",
    "illustrative": "illustrative",
    # B factor. Mol* calls it uncertainty (it doubles as pLDDT for predicted
    # structures), VMD calls it Beta.
    "beta": "uncertainty",
    "uncertainty": "uncertainty",
    "occupancy": "occupancy",
}

#: Styles VMD has no representation for, skipped rather than sent to it.
_STYLES_WITHOUT_VMD = (
    "labels",
    "label",
    "atomlabel",
    "formalcharges",
    "putty",
    "backbone",
    "ellipsoid",
)
#: The same for NGL, which does draw a backbone but none of the rest.
_STYLES_WITHOUT_NGL = (
    "labels",
    "label",
    "atomlabel",
    "formalcharges",
    "putty",
    "ellipsoid",
)


def _normalize(name: str) -> str:
    """Fold a style or colour name to its lookup key."""
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


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

    def _notify(self, event: str, index=None, rep=None):
        """Tell any registered viewer backend that this list changed.

        A viewer already on screen has to hear about a representation the
        moment it is added, where the renderer walks the list once when it
        builds its scene. Translating is deferred to the backend registry, so a
        scene being built pays nothing for this.

        Parameters
        ----------
        event : str
            ``added``, ``updated`` or ``removed``.
        index : int or None
            Which representation, or None when all were removed.
        rep : _Representation or None
            The representation itself, for the two events that have one.
        """
        from moleculekit.viewer.backends import notify

        def params():
            described = self._translateMolstar(rep)
            if described is not None:
                # Beyond what a scene needs: the selection, which a viewer
                # following a trajectory re-evaluates per frame and resolved
                # indices cannot express, and the two flags a live viewer acts
                # on but a single rendered image cannot.
                described["sel"] = rep.sel
                described["visibility"] = rep.visibility
                described["update_sel_every_frame"] = rep.update_sel_every_frame
            return described

        notify(event, self._mol, index, params)

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
        for rep in reps.replist:
            self.replist.append(rep)
            self._notify("added", len(self.replist) - 1, rep)

    def add(
        self,
        sel: str | np.ndarray | None = None,
        style: str | None = None,
        color: "str | int | None" = None,
        frames: list | None = None,
        opacity: float | None = None,
        size: float | None = None,
        c_atom_color: "str | None" = None,
        size_theme: str | None = None,
        label_fields: "str | list | None" = None,
        label_style: dict | None = None,
        visibility: bool | None = None,
        update_sel_every_frame: bool | None = None,
    ):
        """Adds a new representation for Molecule.

        Parameters
        ----------
        sel : str or np.ndarray
            Atom selection (string, boolean mask, or integer index array) for the representation.
            See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
        style : str
            Representation style, in either vocabulary. VMD's ``NewCartoon``,
            ``Cartoon``, ``Licorice``, ``CPK``, ``VDW``, ``Lines``, ``Surf``,
            ``QuickSurf``, ``Points``, ``Putty``, ``Labels`` and ``FormalCharges``, or
            Mol*'s ``cartoon``, ``ball-and-stick``, ``spacefill``, ``line``,
            ``molecular-surface``, ``gaussian-surface``, ``point``, ``putty``,
            ``backbone``, ``ellipsoid`` and
            ``atom-label`` (also ``label``) for the same things. Spacing, case
            and hyphens are ignored, and anything else is rejected rather than
            drawn as something it is not. ``Labels`` writes each atom's name
            beside it and ``FormalCharges`` writes ``+1``/``-1`` on atoms
            carrying one, both on top of another representation that draws the
            atoms; neither reaches VMD or NGL. A name from neither list is passed
            to the VMD backend as written, so VMD's own styles still work
            there.
        color : str or int
            Coloring mode (str) or ColorID (int), in either vocabulary. VMD's
            ``Name``, ``Element``, ``Chain``, ``ResName``, ``Index``,
            ``Secondary Structure``, ``Hydrophobicity``, ``Molecule Type``,
            ``Atom ID``, ``Beta`` and ``Occupancy``, or Mol*'s
            ``element-symbol``, ``chain-id``, ``residue-name``,
            ``sequence-id``, ``secondary-structure``, ``hydrophobicity``,
            ``molecule-type``, ``atom-id``, ``uncertainty`` (the B factor,
            which VMD calls ``Beta``), ``occupancy``, ``element-index``,
            ``entity-id``, ``polymer-id``, ``model-index``, ``structure-index``
            and ``illustrative``. Any SVG colour name or
            ``#rrggbb`` string gives a uniform colour, as does a VMD ColorID.
        frames : list
            List of frames to visualize with this representation. If None it will visualize the current frame only.
        opacity : float
            Opacity of the representation. 0 is fully transparent and 1 is fully opaque.
        size : float
            Scales the drawn size: stick and sphere radius, surface probe, point
            size, label text. Each style keeps its own sensible size at 1.
        c_atom_color : str
            Colour for carbon atoms only, leaving nitrogen, oxygen and the rest
            their element colours. Takes an SVG colour name, a ``#rrggbb``
            string, or one of ``chain-id``, ``entity-id``, ``model-index``,
            ``structure-index``. Ignored unless the representation is coloured
            by element, which is what it modifies.
        size_theme : str
            How sizes are decided before ``size`` scales them: ``physical`` for
            atomic radii, ``uniform`` for one size everywhere, or
            ``uncertainty`` for the B factor. Each style picks a sensible one,
            so this is only worth setting to override it.
        label_fields : str or list
            What a ``Labels`` representation writes beside each atom: any
            per-atom fields of the molecule, such as ``name``, ``element``,
            ``resname``, ``resid``, ``chain`` or ``index``, joined by spaces.
            Without it the label is the atom name alone.
        label_style : dict
            Cosmetics for a ``Labels`` representation: ``border_width``,
            ``border_color``, ``bg_color``, ``bg_opacity``, ``bg_margin``,
            ``offset_x``, ``offset_y`` and ``offset_z``. Keys left out keep
            their defaults, and any other key is rejected.
        visibility : bool
            Whether to draw this representation. A hidden one keeps its place
            in the list, so it can be switched back on by index.
        update_sel_every_frame : bool
            Whether an interactive viewer re-evaluates the selection on every
            trajectory frame, which is what a coordinate-dependent selection
            such as ``within 5 of resname BEN`` needs to follow the structure.
            A rendered image is one frame, so this does not reach it.
        """
        self.replist.append(
            _Representation(
                sel,
                style,
                color,
                frames,
                opacity,
                size,
                c_atom_color,
                size_theme,
                label_fields,
                label_style,
                visibility,
                update_sel_every_frame,
            )
        )
        self._notify("added", len(self.replist) - 1, self.replist[-1])

    def update(
        self,
        index: int,
        sel: "str | np.ndarray | None" = None,
        style: str | None = None,
        color: "str | int | None" = None,
        frames: list | None = None,
        opacity: float | None = None,
        size: float | None = None,
        c_atom_color: "str | None" = None,
        size_theme: str | None = None,
        label_fields: "str | list | None" = None,
        label_style: dict | None = None,
        visibility: bool | None = None,
        update_sel_every_frame: bool | None = None,
    ):
        """Change one representation in place, leaving the rest of it alone.

        Only what is given is changed, so recolouring a representation does not
        cost its size or its selection. Its position in the list is kept, which
        is what makes it addressable by index at all: removing and re-adding
        would move it to the end and renumber everything after it.

        Parameters
        ----------
        index : int
            Which representation to change, as listed by ``print(mol.reps)``.
        sel : str or np.ndarray
            New atom selection. See :meth:`add`.
        style : str
            New style. See :meth:`add`.
        color : str or int
            New colouring. See :meth:`add`.
        frames : list
            New frames to visualize. See :meth:`add`.
        opacity : float
            New opacity. See :meth:`add`.
        size : float
            New size scaling. See :meth:`add`.
        c_atom_color : str
            New carbon colour. See :meth:`add`.
        size_theme : str
            New size theme. See :meth:`add`.
        label_fields : str or list
            New label fields. See :meth:`add`.
        label_style : dict
            New label cosmetics. See :meth:`add`.
        visibility : bool
            Whether to draw it. See :meth:`add`.
        update_sel_every_frame : bool
            Whether a viewer re-evaluates the selection per frame. See
            :meth:`add`.

        Examples
        --------
        >>> mol = tryp.copy()
        >>> mol.reps.add("protein", "NewCartoon", "Secondary Structure")
        >>> mol.reps.update(0, color="Chain")
        >>> mol.reps.update(0, visibility=False)      # keep it, stop drawing it
        """
        rep = self.replist[index]
        changes = {
            "sel": sel,
            "style": style,
            "color": color,
            "frames": frames,
            "opacity": opacity,
            "size": size,
            "c_atom_color": c_atom_color,
            "size_theme": size_theme,
            "label_fields": label_fields,
            "label_style": label_style,
            "update_sel_every_frame": update_sel_every_frame,
        }
        for name, value in changes.items():
            if value is not None:
                setattr(rep, name, value)
        if visibility is not None:
            rep.visibility = bool(visibility)
        self._notify("updated", index, rep)

    def addDefaults(self):
        """Add the representations a viewer draws when none are set.

        Adding any representation replaces the automatic scene wholesale, so
        colouring one ligand otherwise costs the cartoon and everything else.
        This writes that scene out as ordinary entries to edit, reorder or
        remove. They describe the molecule as it is when this is called, so
        call it again after adding or removing atoms.

        Examples
        --------
        >>> mol = tryp.copy()
        >>> mol.reps.addDefaults()
        >>> mol.reps.remove(1)          # drop the waters, ligand and ions
        >>> mol.reps.add("resname BEN", "VDW", color=0)
        """
        from moleculekit.viewer.molstar.scene import default_representations

        for sel, style, color in default_representations(self._mol):
            self.add(sel, style, color)

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
        self._notify("removed", index)

    def list(self):
        """Print all currently stored representations.

        Prints, for each representation, its index, atom selection, style and
        color. Equivalent to printing the Representations object directly.
        """
        print(self)

    def __str__(self):
        s = ""
        for i, r in enumerate(self.replist):
            hidden = "" if r.visibility else ", hidden"
            s += f"rep {i}: sel='{r.sel}', style='{r.style}', color='{r.color}'{hidden}\n"
        return s

    def _translateNGL(self, rep):
        if _normalize(rep.style) in _STYLES_WITHOUT_NGL:
            return None
        styletrans = {
            "newcartoon": "cartoon",
            "cartoon": "cartoon",
            "licorice": "hyperball",
            "lines": "line",
            "line": "line",
            "vdw": "spacefill",
            "spacefill": "spacefill",
            "cpk": "ball+stick",
            "ballandstick": "ball+stick",
            "surf": "surface",
            "molecularsurface": "surface",
            "quicksurf": "surface",
            "gaussiansurface": "surface",
            "points": "point",
            "point": "point",
            "backbone": "backbone",
        }
        colortrans = {
            "name": "element",
            "element": "element",
            "elementsymbol": "element",
            "index": "residueindex",
            "sequenceid": "residueindex",
            "chain": "chainindex",
            "chainid": "chainindex",
            "resname": "resname",
            "residuename": "resname",
            "secondarystructure": "sstruc",
            "hydrophobicity": "hydrophobicity",
            "moleculetype": "moleculetype",
            "atomid": "atomindex",
            "entityid": "entityindex",
            "modelindex": "modelindex",
            "beta": "bfactor",
            "uncertainty": "bfactor",
            "occupancy": "occupancy",
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
        style = styletrans.get(_normalize(rep.style), rep.style)
        if isinstance(rep.color, int):
            color = hexcolors[rep.color]
        else:
            color = colortrans.get(_normalize(rep.color), rep.color)
        return _Representation(sel=selidx, style=style, color=color, size=rep.size)

    def _translateMolstar(self, rep):
        """Translate a VMD-flavored representation to a plain dict for the
        inline molstar scene IR: resolved atom indices, an MVS rep type, and a
        color (a {"theme": name} dict or a uniform hex/SVG string). Returns
        None if the selection matches no atoms."""
        styletrans = MOLSTAR_STYLES
        themetrans = MOLSTAR_THEMES
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
        indices = [int(i) for i in self._mol.atomselect(rep.sel, indexes=True)]
        if not indices:
            return None
        style_key = _normalize(rep.style)
        if style_key not in styletrans:
            # Silently drawing something else is worse than not drawing: an
            # unmapped style used to come out as ball-and-stick, so asking for
            # a surface produced sticks and looked like the surface had simply
            # not worked.
            raise ValueError(
                f"Unknown representation style {rep.style!r}. Use one of "
                f"{sorted(s.title() for s in styletrans)}."
            )
        style = styletrans[style_key]
        if isinstance(rep.color, int):
            color = hexcolors.get(rep.color, "#808080")
        elif _normalize(rep.color) in themetrans:
            color = {"theme": themetrans[_normalize(rep.color)]}
        else:
            color = rep.color
        if rep.c_atom_color is not None and isinstance(color, dict):
            # Only the element theme has a carbon to recolour. A representation
            # coloured any other way keeps that colour, which is how pmview
            # behaves when both are given.
            if color.get("theme") == "element-symbol":
                carbon = _normalize(rep.c_atom_color)
                color = dict(color)
                color["carbon"] = (
                    {"theme": themetrans[carbon]}
                    if carbon in themetrans
                    else {"uniform": rep.c_atom_color}
                )
        out = {"atom_indices": indices, "type": style, "color": color}
        if rep.size_theme is not None:
            if rep.size_theme.lower() not in ("physical", "uniform", "uncertainty"):
                raise ValueError(
                    f"Unknown size_theme {rep.size_theme!r}. Use 'physical', "
                    "'uniform' or 'uncertainty'."
                )
            out["size_theme"] = rep.size_theme.lower()
        if rep.label_fields is not None:
            fields = rep.label_fields
            out["label_fields"] = [fields] if isinstance(fields, str) else list(fields)
        if rep.label_style is not None:
            out["label_style"] = dict(rep.label_style)
        if rep.opacity is not None and rep.opacity != 1:
            out["opacity"] = float(rep.opacity)
        if rep.size is not None:
            out["size_factor"] = float(rep.size)
        return out

    def _repsVMD(self, viewer):
        if len(self.replist) > 0:
            viewer.send("mol delrep 0 top")
            for rep in self.replist:
                if _normalize(rep.style) in _STYLES_WITHOUT_VMD:
                    continue
                if isinstance(rep.color, str):
                    color = VMD_COLORS.get(_normalize(rep.color), rep.color)
                else:
                    color = rep.color
                style = VMD_STYLES.get(_normalize(rep.style), rep.style)
                if rep.size is not None:
                    # VMD takes the size as the representation's first argument.
                    style = f"{style} {float(rep.size)}"
                viewer.send(f"mol selection {rep.sel}")
                viewer.send(f"mol representation {style}")
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
                    params = {"sele": r2.sel, "color": r2.color}
                    if r2.size is not None:
                        params["radiusScale"] = float(r2.size)
                    reps.append({"type": r2.style, "params": params})
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
        Representation style. See :meth:`Representations.add` for the names.
    color : str or int
        Coloring mode (str) or ColorID (int). See :meth:`Representations.add`
        for the names.
    size : float
        Scales the drawn size: stick and sphere radius, surface probe, point
        size, label text.
    c_atom_color : str
        Colour for carbon atoms only. See :meth:`Representations.add`.
    size_theme : str
        How sizes are decided before ``size`` scales them. See
        :meth:`Representations.add`.
    label_fields : str or list
        Fields a ``Labels`` representation writes. See
        :meth:`Representations.add`.
    label_style : dict
        Cosmetics for a ``Labels`` representation. See
        :meth:`Representations.add`.
    visibility : bool
        Whether the representation is drawn. See :meth:`Representations.add`.
    update_sel_every_frame : bool
        Whether the selection is re-evaluated per frame. See
        :meth:`Representations.add`.
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

    def __init__(
        self,
        sel=None,
        style=None,
        color=None,
        frames=None,
        opacity=None,
        size=None,
        c_atom_color=None,
        size_theme=None,
        label_fields=None,
        label_style=None,
        visibility=None,
        update_sel_every_frame=None,
    ):
        self.sel = "all" if sel is None else sel
        self.style = "Lines" if style is None else style
        self.color = "Name" if color is None else color
        self.frames = frames
        self.opacity = 1 if opacity is None else opacity
        self.size = size
        self.c_atom_color = c_atom_color
        self.size_theme = size_theme
        self.label_fields = label_fields
        self.label_style = label_style
        self.visibility = True if visibility is None else bool(visibility)
        self.update_sel_every_frame = update_sel_every_frame
