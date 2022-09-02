import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import sys
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


fig, axs = plt.subplots(3, 4, figsize=(20, 10))


envs = [
    "FetchReach", 
    'FetchPush',
    "FetchSlide", 
    "FetchPick", 
    'HandManipulateBlockRotateZ',
    'HandManipulateBlockRotateParallel',
    'HandManipulateBlockRotateXYZ',
    'HandManipulateBlockFull',
    'HandManipulateEggRotate',
    'HandManipulateEggFull',
    'HandManipulatePenRotate',
    'HandManipulatePenFull',
]

xlims = [
    25,
    50,
    50,
    50,
    50,
    100,
    100,
    100,
    50,
    100,
    50,
    100
]

if sys.argv[1] == "main":
    templates = [
        "her_(-)rew_monolithic_lr0.001_sd{}.pt",
        "her_(-)rew_bilinear_emb16_lr0.001_sd{}.pt",
        "her_(-)rew_dn_emb16_lr0.001_sd{}.pt",
        "her_(-)rew_wn-maxavg_emb16_lr0.001_sd{}.pt",
        "her_(-)rew_pqe_emb16_lr0.001_sd{}.pt",
        "her_(-)rew_asym-max_emb16_lr0.001_sd{}.pt",
    ]
    methods = [
        "monolithic",
        "BVN",
        "DN",
        "WN",
        "PQE",
        "MRN (ours)",
    ]
    colors = ["C0", "C1", "C2", "C3", "C4", "C9", "C8"]
elif sys.argv[1] == "ablation":
    templates = [
        "her_(-)rew_sym_emb16_lr0.001_sd{}.pt",
        "her_(-)rew_max_emb16_lr0.001_sd{}.pt",
        "her_(-)rew_asym-max-sag-latent_emb16_lr0.001_sd{}.pt",
        "her_(-)rew_asym-max_emb16_lr0.001_sd{}.pt",
    ]
    methods = [
         "MRN (Sym Only)",
         "MRN (Asym Only)",
         "MRN (w/ SAG for e2)",
         "MRN (ours)",
    ]
    colors = ["C6", "C8", "C5", "C9"]
elif sys.argv[1] == "ablation2":
    templates = [
        "her_(-)rew_sym_emb16_lr0.002_sd{}.pt",
        "her_(-)rew_max_emb16_lr0.002_sd{}.pt",
        "her_(-)rew_asym-max_emb16_lr0.001_sd{}.pt",
    ]
    methods = [
         "MRN (Sym Only)",
         "MRN (Asym Only)",
         "MRN (ours)",
    ]
    colors = ["C6", "C8", "C9"]

seeds = [100, 200, 300, 400, 500]

def smooth(x, delta=2):
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i-delta):min(n, i+delta)].mean()
    return b

for i,(env,xlim) in enumerate(zip(envs, xlims)):
    i1 = i // 4
    i2 = i % 4

    success = {}
    for j, (method, tmp) in enumerate(zip(methods, templates)):
        success[method] = []
        for seed in seeds:
            try:
                filename = f"./results/{env}_{tmp.format(seed)}"
                res = torch.load(filename)
                s = np.array(res['stats']['successes'])
                s = smooth(s)
                success[method].append(s)
            except:
                print("[error] ", env, method, seed)
                continue
        if len(success[method]) > 0:
            min_len = min([len(x) for x in success[method]])
            print(env, method, min_len)
            s = np.stack([x[:min_len] for x in success[method]])
            axs[i1, i2].plot(s.mean(0), color=colors[j], linewidth=5.0, label=method)
            if len(success[method]) > 1:
                axs[i1, i2].fill_between(np.arange(s.shape[1]), s.mean(0) - s.std(0), s.mean(0) + s.std(0), color=colors[j], alpha=0.3)

    env_title = env if "HandManipulate" not in env else env.replace("HandManipulate", "")
    axs[i1, i2].set_title(env_title, fontsize=20)
    axs[i1, i2].set_xlim(0, xlim)

    axs[i1, i2].set_ylim(0, 1.05)

    axs[i1, i2].spines['top'].set_visible(False)
    axs[i1, i2].spines['right'].set_visible(False)
    axs[i1, i2].spines['bottom'].set_visible(True)
    axs[i1, i2].spines['left'].set_visible(True)

    axs[i1, i2].spines['bottom'].set_edgecolor('black')
    axs[i1, i2].spines['left'].set_edgecolor('black')

    if i2 != 0:
        axs[i1, i2].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs[i1, i2].set_yticklabels(["", "", "", "", "", ""])
        axs[i1, i2].set_yticks(np.linspace(0, 1, 21), minor=True)
        axs[i1, i2].tick_params(which = 'both', direction = 'out')
    else:
        axs[i1, i2].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs[i1, i2].set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        axs[i1, i2].set_yticks(np.linspace(0, 1, 21), minor=True)
        axs[i1, i2].tick_params(which = 'both', direction = 'out')

    if i1 == 2:
        axs[i1, i2].set_xlabel("Epoch", fontsize=19)

    if i2 == 0:
        axs[i1, i2].set_ylabel("Success Rate", fontsize=19)

    if i1 == 0 and i2 == 0:
        axs[i1, i2].legend(fontsize=18, loc='lower right')

    axs[i1, i2].grid()

plt.tight_layout()
plt.savefig(f"{sys.argv[1]}.pdf")
plt.close()
