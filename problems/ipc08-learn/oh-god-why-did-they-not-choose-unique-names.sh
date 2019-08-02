#!/bin/bash

set -e

# So apparently the PDDL files in each of these subdirectories don't have unique
# names, which wreaks havoc with any script that assumes a problem's name
# uniquely identifies that problem. This script rectifies that problem by
# actually giving all PDDL files a unique name.

find . -type f -name '*.pddl' | while read pddl_file
do
    # first remove junk (leading non-alphanumeric characters and the extension)
    sans_ext="$(echo "$pddl_file" | sed 's/\(.pddl$\|^[^[:alnum:]]\+\)//g')"
    # replace slashes & other non-alphanumeric chars with dashes
    alnum_only="$(echo "$sans_ext" | sed 's/[^[:alnum:]]\+/-/g')"
    # replace duplicate '{bootstrap,target}-{typed,untyped}' with single one
    simplified="$(echo "$alnum_only" | sed 's/\(-\?\(\(target\|bootstrap\)-\(typed\|untyped\)\)\)\+/\1/g')"
    # now touch the actual file & replace "(define (problem *)" with "(define
    #(problem *-$simplfied)"
    sed -E -e 's/\s*\(define\s+\(problem\s+([^)]+)\)/(define (problem \1-'"$simplified"')/' \
        -i "$pddl_file"
done
