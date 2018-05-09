LATEXMK		= latexmk
LATEXMKOPT	= -bibtex -pdf
CONTINUOUS	= -pvc

LATEX		= pdflatex
LATEXOPT	= --shell-escape
NONSTOP		= --interaction=nonstopmode

JOBNAME		= Silva_JB-TCC

DEBUG		= '\def\DEBUG{} \input{%S}'


all: monografia clean

monografia: principal.tex
	$(LATEXMK) $(LATEXMKOPT) -pdflatex="$(LATEX) $(LATEXOPT) %O %S" -jobname=$(JOBNAME) principal.tex

debug: principal.tex
	$(LATEXMK) $(LATEXMKOPT) $(CONTINUOUS) -pdflatex="$(LATEX) $(LATEXOPT) $(NONSTOP) %O $(DEBUG)" -jobname=$(JOBNAME) principal.tex

clean:
	$(LATEXMK) -c -jobname=$(JOBNAME) principal.tex
	find . -name \*.nlo -type f -delete
	find . -name \*.brf -type f -delete

cleanall:
	$(LATEXMK) -C -bibtex -jobname=$(JOBNAME) principal.tex
	find . -name \*.nlo -type f -delete
	find . -name \*.brf -type f -delete

.PHONY: all clean cleanall
