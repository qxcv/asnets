# Generalised policy trainer (Action Schema Networks)

## Installation

Most of this ASNet implementation is written in Python, with a few simple C++
extensions. In Ubuntu 18.04 (with Python 3.6), you can install all of the
necessary OS-level dependencies with the following command:

```sh
sudo apt install python3-numpy python3-dev python3-pip python3-wheel \
  python3-venv flex bison build-essential autoconf libtool git \
  libboost-all-dev cmake
```

Installation of the Python components is easiest to carry out in
a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/).
Creating a virtualenv will allow you to sandbox ASNet's Python dependencies,
preventing them from spreading to the rest of your system. The following set of
commands should create an appropriate environment and install all dependencies:

```sh
cd /path/to/this/dir/for/asnets
# make virtual environment for packages in "asnet-env" dir
python3 -m venv asnet-env
# Activate so that we can use packages. You will need to do this each time you
# want to run ASNets in a new shell; you can see whether the environment has
# been activated by looking at the beginning of your shell prompt, which (in
# bash) should start with "(asnet-env)" so long as you're in the environment.
source asnet-env/bin/activate
# sometimes necessary to avoid "invalid command: 'bdist_wheel'"
pip install --upgrade pip
# install all dependencies & install ASNets package
pip install -e .
```

## Running manually

`run_asnets.py` is the entry point for the trainer. `python run_asnets.py
--help` will explain the options which can be supplied to the trainer. The
following example shows how to train a two-layer network on a set of exploding
blocksworld problems:

```sh
# Train on p01-p06; the `{1,2,3,4,5,6}*.pddl` Bash syntax
# is a concise way of including all relevant PDDL files (except domain)
CUDA_VISIBLE_DEVICES="" ./run_asnets -m actprop \
   -O 'num_layers=2,hidden_size=16' --supervised \
   ../problems/ippc08/domains/ex-blocksworld/domain.pddl \
   ../problems/ippc08/domains/ex-blocksworld/p0{1,2,3,4,5,6}*.pddl
```

`CUDA_VISIBLE_DEVICES=""` was included to prevent `run_asnets.py` from using a
GPU; the GPU is much slower than the CPU for this network!

Here's a quick rundown of the most important options:

- `--supervised` trains the network in supervised mode. Omitting this flag will
  make the program train in RL mode (which doesn't work right now).
- `-m actprop -O 'num_layers=2,hidden_size=16'` controls the network
  architecture. You can also use `-m simple -O num_layers=2,hidden_size=16` to
  train a fully connected network (although I'm not sure whether this still
  works right now).
- `-p <problem name>` can be used to restrict the problems used for training.
  For instance, `-p problem1 -p problem2 -p problem3` will force `run_asnets.py`
  to only train on the problems named `problem1`, `problem2`, and `problem3`, even
  if the PDDL files listed at the end of the `run_asnets.py` invocation also
  contain other problems. By default (i.e. without `-p` flags), `run_asnets.py`
  trains on all problems in all PDDL files it is given.
- `--resume-from <path to snapshot>` can initialise a network's weights from
  some given path. Usuaully network weights are saved in `snapshots/<some unique
  identifier>/<name>.pkl`. This option is quite useful for training on one set
  of problems and testing on another.
- `--no-train` will skip training and only evaluate the network on the given
  problems. This is helpful in conjunction with `--resume-from`.

## Running a complete experiment

The `experiments/` subdirectory contains a collection of network/trainer
configurations and sets of PDDL problems to train and test on. For example,
`experiments/actprop_3l.py` configures a three-layer action proposition network
with a certain amount of training time, a certain size for hidden units, etc.,
while `experiments/ex_blocksworld.py` describes a subset of exploding
blocksworld problems to train on, and a set to test on. You could test the
`actprop_3l` trainer configuration and the `ex_blocksworld` problem
configuration with the following comment:

```
# experiments.actprop_3l is the Python module path for experiments/actprop_3l.py;
# likewise for the second argument.
CUDA_VISIBLE_DEVICES="" ./run_experiment experiments.actprop_3l experiments.ex_blocksworld
```

This will train the network for a fixed amount of time (it was 2hrs originally,
but check the configuration in `actprop_3l.py` for the most recent value), then
tests it. All results and intermediate working are written to
`experiments-results/P<something>` (the subdirectory name should be reasonably
obvious).

The `collate_results.py` script in `asnets/scripts` can merge the results
produced by `run_experiment.py` into `.json` files that are easy to interpret
for the other tools in `asnets/scripts`.
