# Metric Residual Networks for Sample Efficient Goal-Conditioned Reinforcement Learning

This repo contains the official implementation for Metric Residual Networks, and other
neural architectures including monolithic critic, Bilinear Value Network (BVN),
and Poisson Quasimetric Embedding (PQE), on 12-GCRL problems.

![Alt Text](https://github.com/Cranial-XIX/MRN/blob/main/misc/gcrl_env.pdf)
![Alt Text](https://github.com/Cranial-XIX/MRN/blob/main/misc/mrn.pdf)


## 1. Dependencies
Coming soon!

## 2. Code structure
The code structure is listed in below. Note that we provide MHER, GCSL, and WGCSL
implementation in PyTorch as well for the convenience of future research, though
they are not used in our paper.
```
 └─run_all.sh (the script to reproduce all results using different critics)
 └─run.sh     (the script to run with a specific critic architecture)
 └─main.py    (the main file to run all code)
 └─plot.py    (plotting utils to make figures in the paper)
 └─src
   └─model.py (include different critic architectures, and the actor architecture)
   └─agent
     └─base.py  (base class for goal-conditioned agent)
     └─her.py   (the Hindsight Experience Replay agent)
     └─ddpg.py  (DDPG agent)
     └─mher.py  (M-HER agent)
     └─gcsl.py  (GCSL agent)
     └─wgcsl.py (WGCSL agent)
 ```

## 2. To reproduce results in the paper
```
./run_all.sh
```

## 3. Citations
If you find our work interesting or the repo useful, please consider citing [this paper](https://arxiv.org/abs/2208.08133.pdf):
```
@article{liu2022metric,
  title={Metric Residual Networks for Sample Efficient Goal-conditioned Reinforcement Learning},
  author={Liu, Bo and Feng, Yihao and Liu, Qiang and Stone, Peter},
  journal={arXiv preprint arXiv:2208.08133},
  year={2022}
}
```
