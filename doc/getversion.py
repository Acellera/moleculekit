# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import subprocess
import sys


def _run(cmd):
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL
        ).decode("utf8")
    except subprocess.CalledProcessError:
        return ""


if sys.argv[1] == "tag":
    output = _run("git describe --tags")
    tag = output.split("-")[0]
    print(tag)

if sys.argv[1] == "branch":
    output = _run("git rev-parse --abbrev-ref HEAD")
    if output.startswith("master"):
        print("latest")
    elif output.startswith("rel-"):
        print("stable")
