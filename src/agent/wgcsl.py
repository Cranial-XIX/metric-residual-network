import copy
import numpy as np
import time
import torch

from src.model import *
from src.replay_buffer import ReplayBuffer
from src.utils import *
from src.sampler import Sampler
from src.agent.gcsl import GCSL


class Advque:
    def __init__(self, size=50000):
        self.size = size 
        self.current_size = 0
        self.que = np.zeros(size)
        self.idx = 0
    
    def update(self, values):
        values = values.reshape(-1)
        l = len(values)

        if self.idx + l <= self.size:
            idxes = np.arange(self.idx, self.idx+l)
        else:
            idx1 = np.arange(self.idx, self.size)
            idx2 = np.arange(0, self.idx+l -self.size)
            idxes = np.concatenate((idx1, idx2))
        self.que[idxes] = values.reshape(-1)

        self.idx = (self.idx + l) % self.size 
        self.current_size = min(self.current_size+l, self.size)

    def get(self, threshold):
        return np.percentile(self.que[:self.current_size], threshold)


class WGCSL(GCSL):
    """
    Goal-conditioned supervised learning agent
    """
    def __init__(self, args, env):
        super().__init__(args, env)

        critic_map = {
            'monolithic': CriticMonolithic,
            'bilinear': CriticBilinear,
            'l2': CriticL2,
        }
        self.critic = critic_map[args.critic](args)
        sync_networks(self.critic)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        if self.args.cuda:
            self.critic.cuda()
            self.critic_target.cuda()
        self.critic_optim  = torch.optim.Adam(self.critic.parameters(),
                                              lr=self.args.lr_critic)

        self.adv_que = Advque()

        def sample_func(S, A, AG, G, size):
            return self.sampler.sample_wgcsl_transitions(
                    S, A, AG, G, size, args,
                    self._get_Q, self.env.compute_reward, self.adv_que)

        self.sample_func = sample_func
        self.buffer = ReplayBuffer(args, self.sample_func)

    def _get_Q(self, S, G):
        with torch.no_grad():
            S, G = self._preproc_inputs(S, G)
            A_ = self.actor(S, G)
        Q = self.critic_target(S, A_, G).detach().cpu().numpy()
        return Q

    def _update(self):
        transition = self.buffer.sample(self.args.batch_size)
        S  = transition['S']
        NS = transition['NS']
        A  = transition['A']
        G  = transition['G']
        R  = transition['R']
        W  = transition['W']
        # S: (batch, dim_state)
        # A: (batch, dim_action)
        # G: (batch, dim_goal)
        # W: (batch, 1)
        R = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda)
        A = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        W = numpy2torch(W, unsqueeze=False, cuda=self.args.cuda)
        S, G = self._preproc_inputs(S, G)
        NS, _ = self._preproc_inputs(NS)

        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            NA = self.actor_target(NS, G)
            NQ = self.critic_target(NS, NA, G).detach()
            # WGCSL uses reward in range {0, 1}
            clip_return = 1 / (1 - self.args.gamma)
            if self.args.negative_reward:
                target = (R + self.args.gamma * NQ).detach().clamp_(-clip_return, 0)
            else:
                target = (R + self.args.gamma * NQ).detach().clamp_(0, clip_return)

        Q = self.critic(S, A, G)
        critic_loss = F.mse_loss(Q, target)

        A_ = self.actor(S, G)
        actor_loss = ((A_ - A).pow(2) * W).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optim.step()
        return actor_loss.item(), critic_loss.item()

    def learn(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            successes = []
            hitting_times = []
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
                'critic_losses': [],
            }

        # put something to the buffer first
        self.prefill_buffer()
        if self.args.cuda:
            n_scales = (self.args.max_episode_steps * self.args.rollout_n_episodes // (self.args.n_batches*2)) + 1
        else:
            n_scales = 1

        for epoch in range(self.args.n_epochs):
            AL, CL = [], []
            for _ in range(self.args.n_cycles):
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)

                for _ in range(n_scales): # scale up for single thread
                    for _ in range(self.args.n_batches):
                        a_loss, c_loss = self._update()
                        AL.append(a_loss); CL.append(c_loss)

                    self._soft_update(self.actor_target, self.actor)
                    self._soft_update(self.critic_target, self.critic)

            global_success_rate, global_hitting_time = self.eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL); CL = np.array(CL)
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                stats['critic_losses'].append(CL.mean())
                print(f"[info] epoch {epoch:3d} success rate {global_success_rate:6.4f} | "+\
                        f"time {(t1-t0)/60:6.4f} min")
                self.save_model(stats)
