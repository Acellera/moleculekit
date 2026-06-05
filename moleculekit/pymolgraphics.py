def draw_cylinder(
    objname: str, xyz1, xyz2, rgb1, rgb2=None, alpha: float = 1, radius: float = 2
) -> str:
    """Draw a cylinder as a CGO object in PyMOL.

    Parameters
    ----------
    objname : str
        The name to give to the created PyMOL object.
    xyz1 : list
        The starting coordinates of the cylinder as ``[x, y, z]``.
    xyz2 : list
        The ending coordinates of the cylinder as ``[x, y, z]``.
    rgb1 : list
        The color of the starting cap as ``[red, green, blue]``.
    rgb2 : list
        The color of the ending cap as ``[red, green, blue]``. If None the
        same color as `rgb1` is used.
    alpha : float
        The opacity of the cylinder.
    radius : float
        The radius of the cylinder.

    Returns
    -------
    objname : str
        The name of the created PyMOL object.
    """
    from pymol import cmd

    if rgb2 is None:
        rgb2 = rgb1
    if len(xyz1) != 3 or len(xyz2) != 3:
        raise RuntimeError("xyz arguments need to be a list of 3 values [x, y, z]")
    if len(rgb1) != 3 or len(rgb2) != 3:
        raise RuntimeError(
            "rgb arguments need to be a list of 3 values [red, green, blue]"
        )
    cmd.load_cgo([25.0, alpha, 9.0, *xyz1, *xyz2, radius, *rgb1, *rgb2], objname)
    return objname


def group_objects(groupname: str, objects: list):
    """Group several PyMOL objects together under a single name.

    Parameters
    ----------
    groupname : str
        The name of the group to create or add to.
    objects : list
        A list of names of the PyMOL objects to add to the group.
    """
    from pymol import cmd

    cmd.group(groupname, " ".join(objects), "add")
