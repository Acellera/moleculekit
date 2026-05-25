# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
REFERENCEDIR  = $(SOURCEDIR)/reference
GITTAG       = $(shell python getversion.py tag)

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html rst writeversion clean publish

html: Makefile rst
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

writeversion:
	echo "$(GITTAG)" > source/version

rst: writeversion
	mkdir -p $(REFERENCEDIR)
	$(SPHINXAPIDOC) --force --separate -o $(REFERENCEDIR) ../moleculekit/ >/dev/null
	@echo "RST files written to $(REFERENCEDIR)/"

clean:
	rm -f $(REFERENCEDIR)/moleculekit.*
	rm -f $(SOURCEDIR)/version
	rm -f $(SOURCEDIR)/llms-full.txt
	rm -rf build

publish:
	ssh software.acellera.com 'rm -rf /var/www/software.acellera.com/source/moleculekit/*'
	scp -r source/* software.acellera.com:/var/www/software.acellera.com/source/moleculekit/
	ssh software.acellera.com 'chmod -R g+rwX /var/www/software.acellera.com/source/moleculekit/'
