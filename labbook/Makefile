##
##  ISG@UP Makefile for Lab Books
##  Copyright (C) 2014-2022, University of Pretoria.
##

TEXFLAGS="-shell-escape"

SOURCES = $(wildcard *.tex *.cls)

all: labbook.pdf

labbook.pdf: $(SOURCES)
	pdflatex $(TEXFLAGS) labbook
	if [ -f isglabbook.ist ]; then \
	  makeindex -s isglabbook.ist -o labbook.ind labbook.idx ; \
	fi
	bibtex labbook
	pdflatex $(TEXFLAGS) labbook
	if [ -f isglabbook.ist ]; then \
	  makeindex -s isglabbook.ist -o labbook.ind labbook.idx ; \
	fi
	pdflatex $(TEXFLAGS) labbook

## Housecleaning
clean:
	rm -f *.aux *.log *.toc *.lof *.lot *.bbl *.blg *.dvi *.ps *.idx *.brf
	rm -f *.ind *.ilg *.gld *.glb *.glo *.bak *.ent *.out labbook.pdf
	rm -f *.fls *.synctex.gz *.fdb_latexmk
	rm -f *converted-to.pdf

## End of File

