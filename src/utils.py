import os
import numpy as np
import gym
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import matplotlib.pyplot as plt

from gym.core import Wrapper
from gym.spaces import Dict, Box
from mpi4py import MPI

################################################################################
#
# Preprocessing
#
################################################################################


class Normalizer(object):
    """
    A helper class for online normalizing observations / goals.
    It keeps the (sum_i X_i) and (sum_i X_i^2) and count for computing variance.
    """
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        # local stats
        self.local_sum   = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)

        # global stats
        self.total_sum   = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)

        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std  = np.ones(self.size, np.float32)

        # thread locker
        self.lock = threading.Lock()
    
    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            # reset
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0

        # synrc the stats
        sync_sum, sync_sumsq, sync_count = self.sync(
                local_sum, local_sumsq, local_count)

        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count

        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(
            np.square(self.eps), 
            (self.total_sumsq/self.total_count) - \
                    np.square(self.total_sum/self.total_count)
        ))

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    # average across the cpu's data
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)

    def normalize_goal(self, v, goal_idx, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean[goal_idx]) / (self.std[goal_idx]), -clip_range, clip_range)

    # for partially unnormalize a tensor (since we have clipping)
    def unnormalize(self, v):
        return v * self.std + self.mean

    def unnormalize_goal(self, v, goal_idx):
        return v * self.std[goal_idx] + self.mean[goal_idx]


def numpy2torch(v, unsqueeze=False, cuda=False):
    if v.dtype == np.float32 or v.dtype == np.float64:
        v_tensor = torch.tensor(v).float()
    elif v.dtype == np.int32 or v.dtype == np.int64:
        v_tensor = torch.LongTensor(v)
    else:
        raise Exception(f"[error] unknown type {v.dtype}")

    if unsqueeze:
        v_tensor = v_tensor.unsqueeze(0)
    if cuda:
        v_tensor = v_tensor.cuda()
    return v_tensor

def first_nonzero(arr, axis, invalid_val=-1):
    mask = (arr != 0)
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

################################################################################
#
# Plotting utils
#
################################################################################

def plot(ax, S, goal, A=None, quiver=True):
    for i in range(S.shape[0]):
        states = S[i]
        if A is not None:
            actions = A[i]
        x = states[:, 0]
        y = states[:, 1]
        num_states = len(states)
        plasma_cm = plt.get_cmap('plasma')
        for i, state in enumerate(states):
            color = plasma_cm(float(i) / num_states)
            ax.plot(state[0], state[1],
                    marker='o', color=color, markersize=5,
                    )

        if A is not None:
            actions_x = actions[:, 0]
            actions_y = actions[:, 1]

        if quiver:
            #ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
            #          scale_units='xy', angles='xy', scale=1, width=0.001)

            if A is not None:
                ax.quiver(x[:-1], y[:-1], actions_x, actions_y, scale_units='xy',
                          angles='xy', scale=1, color='r',
                          width=0.001, )

    boundary_dist = 4
    if goal is not None:
        ax.plot(goal[0], goal[1], marker='*', color='g', markersize=15)
    ax.set_ylim(-boundary_dist-1, boundary_dist+1)
    ax.set_xlim(-boundary_dist - 1, boundary_dist + 1)


def plot_state_action(ax, S, A, goal=None):
    ax.scatter(S[:,0], S[:,1], marker='o')
    ax.quiver(S[:,0], S[:,1], A[:,0], A[:,1],
              scale_units='xy',
              angles='xy',
              scale=1,
              color='r',
              width=0.005)

    boundary_dist = 4
    if goal is not None:
        ax.plot(goal[0], goal[1], marker='*', color='g', markersize=15)
    ax.set_ylim(-boundary_dist-1, boundary_dist+1)
    ax.set_xlim(-boundary_dist - 1, boundary_dist + 1)


################################################################################
#
# MPI utils
#
################################################################################

def sync_networks(network):
    comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode='params')
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params_or_grads(network, flat_params, mode='params')

def sync_grads(network, scale_grad_by_procs=False):
    flat_grads = _get_flat_params_or_grads(network, mode='grads')
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    if scale_grad_by_procs:
        global_grads /= comm.Get_size()
    _set_flat_params_or_grads(network, global_grads, mode='grads')
    return np.linalg.norm(global_grads)

def _get_flat_params_or_grads(network, mode='params'):
    li = []
    for p in network.parameters():
        if mode == 'params':
            li.append(p.data.cpu().numpy().flatten())
        else:
            if p.grad is not None:
                li.append(p.grad.cpu().numpy().flatten())
            else:
                zeros = torch.zeros_like(p.data).cpu().numpy().flatten()
                li.append(zeros)
    return np.concatenate(li)

def _set_flat_params_or_grads(network, flat_params, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    pointer = 0
    for param in network.parameters():
        if not (attr == 'grad' and param.grad == None):
            getattr(param, attr).copy_(
                    torch.tensor(
                        flat_params[pointer:pointer + param.data.numel()]
                    ).view_as(param.data))
        pointer += param.data.numel()


################################################################################
#
# Environment utils
#
################################################################################


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


DEFAULT_ENV_PARAMS = {
    'Point2DLargeEnv-v1':{
         'wgcsl_baw_delta': 0.15,
    },
    'Point2D-FourRoom-v1':{
        'wgcsl_baw_delta': 0.15,
    },
    'SawyerReachXYZEnv-v1':{
        'wgcsl_baw_delta': 0.15,
    },
    'FetchReach-v1': {
        'wgcsl_baw_delta': 0.15,
    },
    'Reacher-v2': {
        'wgcsl_baw_delta': 0.15,
    },
    'SawyerDoor-v0':{
        'wgcsl_baw_delta': 0.15,
        },
    'FetchPush-v1':{
        'wgcsl_baw_delta': 0.01,
        },
    'FetchSlide-v1':{
        'wgcsl_baw_delta': 0.01,
        },
    'FetchPickAndPlace-v1':{
        'wgcsl_baw_delta': 0.01,
        },
    'HandReach-v0':{
        'wgcsl_baw_delta': 0.01,
        }
}
