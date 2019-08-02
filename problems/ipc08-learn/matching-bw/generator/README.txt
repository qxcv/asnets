The matching-bw generator is shell script that calls three executables
that you must compile: 

Compile these executables via the commands:

  cc bwstates.c -o bwstates -lm
  cc 2pddl-typed.c -o 2pddl-typed
  cc 2pddl-untyped.c -o 2pddl-untyped

You also need to make sure that the shell script is executable via:

  chmod +x matching-bw-generator.sh

Then to generate a matching-bw problem with base name "bname" and size
"n" you can call:

  matching-bw-generator.sh bname n

This will create two files: bname-typed.pddl and bname-untyped.pddl 

which are the typed and untyped versions of the files


