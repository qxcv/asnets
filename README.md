# Implementation of ASNets

This repository contains the code used in the AAAI'18 paper
[Action Schema Networks: Generalised Policies with Deep Learning](https://arxiv.org/abs/1709.04271).
The abstract of that paper explains the idea:

> In this paper, we introduce the Action Schema Network (ASNet): a neural network architecture for learning generalised policies for probabilistic planning problems. By mimicking the relational structure of planning problems, ASNets are able to adopt a weight-sharing scheme which allows the network to be applied to any problem from a given planning domain. This allows the cost of training the network to be amortised over all problems in that domain. Further, we propose a training method which balances exploration and supervised training on small problems to produce a policy which remains robust when evaluated on larger problems. In experiments, we show that ASNet's learning capability allows it to significantly outperform traditional non-learning planners in several challenging domains. 

This repository is structured as follows:

- `deepfpg/` contains our implementation and experiment files. The main entry point is
  [`deepfpg/fpg.py`](https://github.com/qxcv/asnets/blob/master/deepfpg/fpg.py). Consult
  [`deepfpg/README.md`](https://github.com/qxcv/asnets/blob/master/deepfpg/README.md) for
  instructions on installing and running the code.
- `models/` contains the trained models for our AAAI'18 paper. You can use those
  models by passing the appropriate command line flags to
  [`deepfpg/fpg.py`](https://github.com/qxcv/asnets/blob/master/deepfpg/fpg.py).
  For instance,
  `-m actprop -O num_layers=2,hidden_size=16 --resume-from models/ttw-adm/snapshot_31_1.000000.pkl`
  will load one of the two-layer networks which we trained for Triangle Tireworld.
- `problems/` includes all problems that we used to train + test the network, plus some problems
  which might be helpful for further research or debugging.

If you use this code in an academic publication, we'd appreciate it if you cited the following paper:

```bibtex
@inproceedings{toyer2018action,
  title={Action Schema Networks: Generalised Policies with Deep Learning},
  author={Toyer, Sam and Trevizan, Felipe and Thi{\'e}baux, Sylvie and Xie, Lexing},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2018}
}
```

Comments & queries can go to [Sam Toyer](mailto:sam@qxcv.net).
