# Implementation of ASNets

**Update August 2019:** This repository has been updated with a new copy of the
ASNets code. The new code is substantially cleaner and includes new
functionality described in a journal article that is currently under review. The
old code and saved models for reproducing the AAAI paper are still available in
the `aaai18` branch.

This repository contains the code used in the AAAI'18 paper [Action Schema
Networks: Generalised Policies with Deep
Learning](https://arxiv.org/abs/1709.04271), as well as a follow-up journal
article that is currently under review. The abstract of the AAAI paper explains
the idea:

> In this paper, we introduce the Action Schema Network (ASNet): a neural
> network architecture for learning generalised policies for probabilistic
> planning problems. By mimicking the relational structure of planning problems,
> ASNets are able to adopt a weight-sharing scheme which allows the network to
> be applied to any problem from a given planning domain. This allows the cost
> of training the network to be amortised over all problems in that domain.
> Further, we propose a training method which balances exploration and
> supervised training on small problems to produce a policy which remains robust
> when evaluated on larger problems. In experiments, we show that ASNet's
> learning capability allows it to significantly outperform traditional
> non-learning planners in several challenging domains.

A longer, informal explanation can be found in [a blog post on
ASNets](http://cm.cecs.anu.edu.au/post/asnets/) that discusses the AAAI'18
submission.

This repository is structured as follows:

- `asnets/` contains our implementation and experiment files. Consult
  [`asnets/README.md`](https://github.com/qxcv/asnets/blob/master/asnets/README.md)
  for instructions on installing and running the code.
- `problems/` includes all problems that we used to train + test the network,
  plus some problems which might be helpful for further research or debugging.
- `materials/` contains papers and slides relevant to ASNets. They may be
  helpful for understanding this code, or ASNets more generally.

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
