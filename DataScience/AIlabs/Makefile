buildPresentation:
	cd presentation/ ; pdflatex ailabs ; pdflatex -halt-on-error ailabs

presentation:
	evince presentation/ailabs.pdf

clean:
	find . -regex '.*\.\(aux\|bbl\|log\|blg\|toc\|out\|snm\|nav\|o\)' -delete
