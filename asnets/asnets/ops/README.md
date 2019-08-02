# Custom TensorFlow ops

For performance reasons, some parts of the actual ASNet network are implemented
as custom TensorFlow ops in C++. This directory contains implementations &
wrappers for those parts. See the main `setup.py` file for ASNets for info on
how they're built; if you don't install ASNets with `setup.py` then they won't
be built by default!
