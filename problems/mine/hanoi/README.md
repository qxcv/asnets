This dir contains an impl of the Towers of Hanoi problem from
https://github.com/SoarGroup/Domains-Planning-Domain-Definition-Language

I hacked the PDDL files to add types, since my parser can't handle untyped
stuff. I also don't have a generator for the domain, but I suspect that won't
really matter (instances larger than size 8 will take more than 300 steps to
solve anyway, which would necessitate a special configuration with a higher
action count limit).
