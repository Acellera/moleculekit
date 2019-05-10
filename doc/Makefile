# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
GITTAG       = $(shell python getversion.py tag)
GITBRANCH    = $(shell python getversion.py branch)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

rst:
	$(SPHINXAPIDOC) --force --separate -o $(SOURCEDIR) ../moleculekit/ >/dev/null
	echo "$(GITTAG)" > source/version
	@echo
	@echo "Build finished. The RST files are on source/"

clean:
	rm -f source/moleculekit.*
	rm source/version

publish:
	ssh www.acellera.com 'rm -rf ~/software.acellera.com/source/moleculekit/*'
	scp -r source/* www.acellera.com:~/software.acellera.com/source/moleculekit/
	ssh www.acellera.com 'chmod -R g+rwX software.acellera.com/'

