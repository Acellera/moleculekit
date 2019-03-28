# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from subprocess import call


def makeMajorRelease(version, message):
    relname = 'rel-{}'.format(version)
    call(['git', 'checkout', 'master'])
    call(['git', 'fetch'])
    call(['git', 'pull'])
    call(['git', 'checkout', '-b', relname])
    call(['git', 'tag', version, '-m', message])
    call(['git', 'push', '--tags', 'origin', relname])
    call(['git', 'checkout', 'master'])
    call(['git', 'tag', version, '-m', message])
    call(['git', 'push', '--tags'])


