all: bwstates 2pddl-typed 2pddl-untyped
.PHONY: all

bwstates: bwstates.c bwstates.h
	$(CC) -o bwstates bwstates.c -lm

2pddl-typed: 2pddl-typed.c
	$(CC) -o 2pddl-typed 2pddl-typed.c -lm

2pddl-untyped: 2pddl-untyped.c
	$(CC) -o 2pddl-untyped 2pddl-untyped.c -lm
