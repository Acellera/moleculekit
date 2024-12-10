# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import sys


def getOpenBabelProperties(mol):
    import subprocess
    import os
    from moleculekit.util import tempname

    pdbfile = tempname(suffix=".pdb")
    outfile = tempname(suffix=".csv")
    mol.write(pdbfile)

    obabelcli = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "obabel_cli.py"
    )

    try:
        output = subprocess.check_output(
            [sys.executable, obabelcli, pdbfile, outfile],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        print(
            "Failed to call getOpenBabelProperties with error",
            exc.returncode,
            exc.output,
        )
    else:
        if len(output):
            print(output)

    os.remove(pdbfile)

    atoms = []
    with open(outfile, "r") as f:
        for line in f:
            pieces = line.split(",")
            pieces[0] = int(pieces[0])
            pieces[2] = int(pieces[2])
            pieces[5] = float(pieces[5])
            atoms.append(pieces)

    atoms.sort(key=lambda x: x[0])  # Sort by index

    return atoms


def openbabelConvert(input_file, input_format, output_format, extra_args=()):
    """
    Converts the file from the input format to the output format specified. It uses the openbabel features

    Parameters
    ----------
    input_file: str
        The path of the input file to convert
    input_format: str
        The input file format
    output_format: str
        The output file format

    Returns
    -------
    outfile: str
        The output file generated
    """
    import subprocess
    from moleculekit.util import tempname

    input_format = input_format[1:] if input_format.startswith(".") else input_format
    output_format = (
        output_format[1:] if output_format.startswith(".") else output_format
    )

    outfile = tempname(suffix=f".{output_format}")
    try:
        output = subprocess.check_output(
            [
                "obabel",
                f"-i{input_format}",
                input_file,
                f"-o{output_format}",
                f"-O{outfile}",
                *extra_args,
            ],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            "Failed to call openbabel with error. You might need to install openbabel with `conda install openbabel -c conda-forge`",
            exc.returncode,
            exc.output,
        )
    else:
        print(output)

    return outfile
