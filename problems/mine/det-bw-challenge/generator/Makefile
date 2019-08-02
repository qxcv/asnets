CC = gcc

#MATHLIB = /usr/lib/libm.a
MATHLIB = -lm

CFLAGS = -O2 $(DEBUGFLAG) $(DATADEF)

PROGRAMS = bwkstates

OPTOBS = optimal.o gn1lib.o utils.o mshs.o

all: $(PROGRAMS)

optimal: $(OPTOBS)
	$(CC) $(CFLAGS) $(OPTOBS) -o optimal

bwkstates: bwkstates.o
	$(CC) $(CFLAGS) bwkstates.o -o bwkstates $(MATHLIB)

clean:
	/bin/rm -f *.o core $(PROGRAMS)

$(OPTOBS): setlists.h
optimal.o gn1lib.o: optimal.h
