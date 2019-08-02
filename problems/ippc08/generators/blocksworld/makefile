all: blocksworld bwstates

blocksworld: blocksworld.cc
	g++ -o blocksworld blocksworld.cc

bwstates: bwstates.c
	gcc -o bwstates bwstates.c -lm

clean:
	rm -f blocksworld bwstates *~
