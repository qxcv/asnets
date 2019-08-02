all: ex-blocksworld bwstates

ex-blocksworld: ex-blocksworld.cc
	g++ -o ex-blocksworld ex-blocksworld.cc

bwstates: bwstates.c
	gcc -o bwstates bwstates.c -lm

clean:
	rm -f ex-blocksworld bwstates *~
