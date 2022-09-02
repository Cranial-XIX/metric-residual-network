import copy
import numpy as np
import time
import torch

from src.model import *
from src.replay_buffer import ReplayBuffer
from src.utils import *
from src.sampler import Sampler
from src.agent.base import Agent


################################################################################
#
# Supervised learning agents (GCSL / WGCSL)
#
################################################################################


class GCSL(Agent):
    """
    Goal-conditioned supervised learning agent
    """
    def __init__(self, args, env):
        super().__init__(args, env)
        self.sample_func = self.sampler.sample_gcsl_transitions
        self.buffer = ReplayBuffer(args, self.sample_func)

    def _update(self):
        transition = self.buffer.sample(self.args.batch_size)
        S  = transition['S']
        A  = transition['A']
        G  = transition['G']
        # S: (batch, dim_state)
        # A: (batch, dim_action)
        # G: (batch, dim_goal)
        A = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        S, G = self._preproc_inputs(S, G)

        A_ = self.actor(S, G)
        actor_loss = (A_ - A).pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optim.step()
        return actor_loss.item()

    def learn(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            successes = []
            hitting_times = []
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
            }

        # put something to the buffer first
        self.prefill_buffer()
        if self.args.cuda:
            n_scales = (self.args.max_episode_steps * self.args.rollout_n_episodes // (self.args.n_batches*2)) + 1
        else:
            n_scales = 1

        for epoch in range(self.args.n_epochs):
            AL = []
            for _ in range(self.args.n_cycles):
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)

                for _ in range(n_scales): # scale up for single thread
                    for _ in range(self.args.n_batches):
                        al = self._update()
                        AL.append(al)

            global_success_rate, global_hitting_time = self.eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL)
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                print(f"[info] epoch {epoch:3d} success rate {global_success_rate:6.4f} | "+\
                        f"time {(t1-t0)/60:6.4f} min")
                self.save_model(stats)
