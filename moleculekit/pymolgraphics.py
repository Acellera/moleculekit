def draw_cylinder(objname, xyz1, xyz2, rgb1, rgb2=None, alpha=1, radius=2):
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


def group_objects(groupname, objects):
    from pymol import cmd

    cmd.group(groupname, " ".join(objects), "add")
