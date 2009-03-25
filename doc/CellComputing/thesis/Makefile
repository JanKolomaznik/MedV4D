################################################################################
#   Makefile aplikace 'segmentation'
#   sou��st diplomov� pr�ce 'M��en� objemu v 3D datech'
#
#   Autor: V�clav 'Vajicek' Kraj��ek
#

#images
################################################################################

DATA=data

PDF_IMAGES=     bspline_example.pdf

EPS_IMAGES=     cellLayout.eps

all: dvi bib

dvi: *.tex #${EPS_IMAGES}
	latex diplomka.tex

ps: dvi
	dvips -t a4 -o diplomka.ps diplomka.dvi

pdf: ps
	ps2pdf diplomka.ps

pdf2: *.tex ${PDF_IMAGES}
	pdflatex diplomka.tex

cover: bookCover.tex
	pdflatex bookCover.tex

clean:
	rm *.log *.aux *.tex.aux *.lot *.out *.toc *.pdf *.dvi *.lof

#pictures through pdf & pdf latex
################################################################################

trainingSet.pdf: trainingSet.eps
	epstopdf $<

registration.pdf: registration.eps
	epstopdf $<

%.pdf: imgs/schemes/%.svg
	inkscape -f $< -T -A $@

%.pdf: %.eps
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

bib: diplomka_references.bib
	bibtex diplomka

#graphs
################################################################################

plot: plotdata/bsplinebase.gnuplot plotdata/regionstatistics.gnuplot plotdata/benchmarks.gnuplot plotdata/levelsets.gnuplot plotdata/esnake.gnuplot
	cd plotdata && gnuplot bsplinebase.gnuplot
	cd plotdata && gnuplot regionstatistics.gnuplot
	cd plotdata && gnuplot esnake.gnuplot
	cd plotdata && gnuplot levelsets.gnuplot
	cd plotdata && gnuplot benchmarks.gnuplot


 
