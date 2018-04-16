LATEXMK		= latexmk
LATEXMKOPT	= -bibtex -pdf
CONTINUOUS	= -pvc

LATEX		= pdflatex
LATEXOPT	= --shell-escape
NONSTOP		= --interaction=nonstopmode

JOBNAME		= Silva_JB-TCC

DEBUG		= '\def\DEBUG{} \input{%S}'


all: monografia clean

monografia: main.tex
	$(LATEXMK) $(LATEXMKOPT) -pdflatex="$(LATEX) $(LATEXOPT) %O %S" -jobname=$(JOBNAME) main.tex

debug: main.tex
	$(LATEXMK) $(LATEXMKOPT) $(CONTINUOUS) -pdflatex="$(LATEX) $(LATEXOPT) $(NONSTOP) %O $(DEBUG)" -jobname=$(JOBNAME) main.tex

clean:
	$(LATEXMK) -c -jobname=$(JOBNAME) main.tex

cleanall:
	$(LATEXMK) -C -bibtex -jobname=$(JOBNAME) main.tex

.PHONY: all clean cleanall
