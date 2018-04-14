LATEXMK		= latexmk
LATEXMKOPT	= -bibtex -pdf
CONTINUOUS	= -pvc

LATEX		= pdflatex
LATEXOPT	= --shell-escape
NONSTOP		= --interaction=nonstopmode

JOBNAME		= TCC-Julio_Batista_Silva

DEBUG		= '\def\DEBUG{} \input{%S}'


all: apresentacao clean

apresentacao: apresentacao.tex
	$(LATEXMK) $(LATEXMKOPT) -pdflatex="$(LATEX) $(LATEXOPT) %O %S" -jobname=$(JOBNAME) apresentacao.tex

debug: apresentacao.tex
	$(LATEXMK) $(LATEXMKOPT) $(CONTINUOUS) -pdflatex="$(LATEX) $(LATEXOPT) $(NONSTOP) %O $(DEBUG)" -jobname=$(JOBNAME) apresentacao.tex

clean:
	$(LATEXMK) -c -jobname=$(JOBNAME) apresentacao.tex
	rm -rf *.nav *.snm

cleanall:
	$(LATEXMK) -C -bibtex -jobname=$(JOBNAME) apresentacao.tex
	rm -rf *.nav *.snm

.PHONY: all clean cleanall
