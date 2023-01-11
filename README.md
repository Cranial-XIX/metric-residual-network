# Metric Residual Networks for Sample Efficient Goal-Conditioned Reinforcement Learning

This repo contains the official implementation for Metric Residual Networks, and other
neural architectures for the goal-conditioned reinforcement learning (GCRL) critic network.

### 12 GCRL environments
<p align="center">
<img src="https://github.com/Cranial-XIX/metric-residual-network/blob/master/misc/gcrl_env.png" width="800">
<p>

### Implemented Critic Networks
<p align="center">
<img src="https://github.com/Cranial-XIX/metric-residual-network/blob/master/misc/gcrl.png" width="800">
<p>

| Critic Architecture |
| --- |
| Monolithic Network |
| [Deep/Wide Norms (DN/WN)](https://arxiv.org/pdf/2002.05825.pdf) |
| [Bilinear Value Network (BVN)](https://arxiv.org/pdf/2204.13695.pdf) |
| [Poisson Quasimetric Embedding (PQE)](https://arxiv.org/pdf/2206.15478.pdf) |
| [Metric Residual Network (MRN)](https://arxiv.org/abs/2208.08133.pdf) |

**update 2022/12/6:** Thank [@SsnL](https://github.com/SsnL) for pointing out the bug that the metric part should be l2-norm instead of square loss. (See [this paper](https://arxiv.org/abs/2211.15120))

## 1. Dependencies
Create conda environment.
```
conda create -n metric-residual-network python=3.7.4
conda activate metric-residual-network
```
Install PyTorch
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Download [mujoco200](https://www.roboti.us/download.html). Then install pip requirements:
```
pip install -r requirements.txt
```

## 2. Code structure
The code structure is listed in below. Note that we provide 

|             Method                      | Paper |
| --------------------------------------- | ----- |
| Model-based Hindsight Experience Replay | [MHER](https://arxiv.org/pdf/2107.00306.pdf) |
| Goal-Conditioned Supervised Learning    | [GCSL](https://openreview.net/pdf?id=rALA0Xo6yNJ) |
| Weighted Goal-Conditioned Supervised Learning | [WGCSL](https://arxiv.org/pdf/2202.04478.pdf) |

implementation in PyTorch as well for the convenience of future research, though
they are not used in our paper.
```
metric-residual-network
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

## 3. Logs and checkpoints of trained models
For the convenience of future research, you can download all pretrained checkpoints/logs from [here](https://drive.google.com/file/d/1SjzP4Oezt-CZVQ_5BPx5NoI0UIi6YEnl/view?usp=sharing) and 
unzip and put it under this repo. Then you can use plot.py to reproduce 
the 2 main figures about the results on the 12 Fetch environments.

## 4. Citations
If you find our work interesting or the repo useful, please consider citing [this paper](https://arxiv.org/abs/2208.08133.pdf):
```
@article{liu2022metric,
  title={Metric Residual Networks for Sample Efficient Goal-conditioned Reinforcement Learning},
  author={Liu, Bo and Feng, Yihao and Liu, Qiang and Stone, Peter},
  journal={arXiv preprint arXiv:2208.08133},
  year={2022}
}
```
