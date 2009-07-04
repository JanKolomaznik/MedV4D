
################################################################################

DATA=data

PDF_IMAGES= $(DATA)/automaticAlignOfSmallData.pdf $(DATA)/ITKFilter.pdf $(DATA)/newDesign.pdf\
	    $(DATA)/portingCycle.pdf $(DATA)/resultingFilter.pdf $(DATA)/SPEEmbedding.pdf\
	    $(DATA)/cellLayout.pdf $(DATA)/logo.pdf $(DATA)/originalHierarchy.pdf $(DATA)/PPUCentricModel.pdf\
	    $(DATA)/streamingModel.pdf $(DATA)/computationProcess.pdf         $(DATA)/multipleDMAList.pdf\
	    $(DATA)/pipelineModel.pdf      $(DATA)/RCClientCycle.pdf    $(DATA)/sparsefield.pdf\
	    $(DATA)/flooding.pdf $(DATA)/narrowBands.pdf $(DATA)/RCServerCycle.pdf    $(DATA)/speedFunction.pdf


EPS_IMAGES=     cellLayout.eps

all: dvi bib

dvi: *.tex #${EPS_IMAGES}
	latex diplomka.tex

ps: dvi
	dvips -t a4 -o diplomka.ps diplomka.dvi

pdf: ps
	ps2pdf diplomka.ps

pdf2: *.tex ${PDF_IMAGES} bib
	pdflatex diplomka.tex

cover: bookCover.tex
	pdflatex bookCover.tex

clean:
	rm *.log *.aux *.tex.aux *.lot *.out *.toc *.pdf *.dvi *.lof *.bbl. *.blg *.tex~
	data/*.pdf

tmp:
	mkdir $(TMP_DIR) 2>/dev/null && true

#pictures through pdf & pdf latex
################################################################################

#%.pdf: imgs/schemes/%.svg
#	inkscape -f $< -T -A $@

%.pdf: $(DATA)/%.eps
	epstopdf $<

#diagram
################################################################################

shape_based_diag.pdf: shape_based_diag.tex
	latex shape_based_diag.tex
	dvips -o shape_based_diag.ps shape_based_diag.dvi
	ps2pdf shape_based_diag.ps

#conversion through png to eps
################################################################################

%.eps: %.png
	gimp -c -i --batch-interpreter "plug_in_script_fu_eval" -b \
                      "(let* ((img (car (file-png-load 1 \"$<\" \"$<\"))) \
                              (lay (gimp-image-flatten img)) \
                             ) \
                             (file-eps-save 1 img 3 \"$@\" \"$@\" 0 0 0 0 1 1 0 1 0 2))" \
                      "(gimp-quit 0))"
	rm $<

#images from directory
################################################################################

%.png: imgs/schemes/%.svg
	inkscape -f $< -D -d 600 -e $@

%.png: imgs/3d/%.png
	cp $< .

%.png: imgs/atlas/%.png
	cp $< .

%.png: imgs/misc/%.png
	cp $< .

%.png: imgs/snakes/%.png
	cp $< .

#bib
################################################################################

bib: references.bib
	bibtex diplomka

#graphs
################################################################################

plot: plotdata/bsplinebase.gnuplot plotdata/regionstatistics.gnuplot plotdata/benchmarks.gnuplot plotdata/levelsets.gnuplot plotdata/esnake.gnuplot
	cd plotdata && gnuplot bsplinebase.gnuplot
	cd plotdata && gnuplot regionstatistics.gnuplot
	cd plotdata && gnuplot esnake.gnuplot
	cd plotdata && gnuplot levelsets.gnuplot
	cd plotdata && gnuplot benchmarks.gnuplot



