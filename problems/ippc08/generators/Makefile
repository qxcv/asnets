SUBDIRS = Applis RapportStageLaTeX

default:
	@for i in $(SUBDIRS); do		\
        echo "make all in $$i...";		\
        (cd $$i; make); done

clean:
	@for i in $(SUBDIRS); do		\
        echo "make all in $$i...";		\
        (cd $$i; make clean); done
	rm -f *~
