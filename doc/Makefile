# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
TUTORIALSDIR  = source/tutorials
GITTAG       = $(shell python getversion.py tag)
GITBRANCH    = $(shell python getversion.py branch)
IPYNB_FILES  = $(shell ls ../tutorials/*.ipynb)

TUT_RST_FILES := $(addprefix $(TUTORIALSDIR)/,$(notdir $(IPYNB_FILES:.ipynb=.rst)))


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html: Makefile rst
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Rule for building .rst from .ipynb
$(TUTORIALSDIR)/%.rst: ../tutorials/%.ipynb
	mkdir -p $(TUTORIALSDIR)
	jupyter-nbconvert $< --to rst --output-dir=$(TUTORIALSDIR)/

writeversion:
	echo "$(GITTAG)" > source/version

tutorials: writeversion $(TUT_RST_FILES)

rst: writeversion tutorials
	$(SPHINXAPIDOC) --force --separate -o $(SOURCEDIR) ../moleculekit/ >/dev/null
	echo "$(GITTAG)" > source/version
	@echo
	@echo "Build finished. The RST files are on source/"

clean:
	rm -f source/moleculekit.*
	rm -f source/version
	rm -rf $(TUTORIALSDIR)
	rm -rf build

publish:
	ssh software.acellera.com 'rm -rf /var/www/software.acellera.com/source/moleculekit/*'
	scp -r source/* software.acellera.com:/var/www/software.acellera.com/source/moleculekit/
	ssh software.acellera.com 'chmod -R g+rwX /var/www/software.acellera.com/source/moleculekit/'

