import copy
import numpy as np
import time
import torch

from src.model import *
from src.replay_buffer import ReplayBuffer
from src.utils import *
from src.sampler import Sampler
from src.agent.her import HER


class MHER(HER):
    """
    Model-based Hindsight Experience Replay agent
    """
    def __init__(self, args, env):
        super().__init__(args, env)
        self.dynamics = EnsembleSingleStepDynamics(args)
        sync_networks(self.dynamics)
        if self.args.cuda:
            self.dynamics.cuda()

        self.dynamics_optim = torch.optim.Adam(self.dynamics.parameters(),
                                               lr=self.args.lr_transition)

        def sample_func(S, A, AG, G, size):
            return self.sampler.sample_mher_transitions(
                    S, A, AG, G, size,
                    self._imaginary_rollout,
                    self.args.goal_idx)

        self.sample_func = sample_func
        self.buffer = ReplayBuffer(args, self.sample_func)

    def _imaginary_rollout(self, s, g, n_steps=10, act_noise=0.2):
        s, g = self._preproc_inputs(s, g)
        max_action = self.args.max_action

        s_list = [s.cpu().numpy()]

        with torch.no_grad():
            for i in range(n_steps):
                a = self.actor(s, g).detach()
                if act_noise > 0:
                    a += act_noise * max_action * torch.randn_like(a)
                    a = torch.clip(a, -max_action, max_action)
                s = self.dynamics(s, a).detach()
                s_list.append(s.cpu().numpy())
        s_list = np.stack(s_list).transpose(1, 0, 2)
        return s_list

    def _update(self):
        transition = self.buffer.sample(self.args.batch_size)
        S   = transition['S']
        NS  = transition['NS']
        A   = transition['A']
        G   = transition['G']
        R   = transition['R']
        # S/NS: (batch, dim_state)
        # A: (batch, dim_action)
        # G: (batch, dim_goal)
        A  = numpy2torch(A, unsqueeze=False, cuda=self.args.cuda)
        R  = numpy2torch(R, unsqueeze=False, cuda=self.args.cuda)
        S,   G = self._preproc_inputs(S, G)
        NS,  _ = self._preproc_inputs(NS)

        # 1. update critic
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            NA = self.actor_target(NS, G)
            NQ = self.critic_target(NS, NA, G).detach()
            clip_return = 1 / (1 - self.args.gamma)
            if self.args.negative_reward:
                target = (R + self.args.gamma * NQ).detach().clamp_(-clip_return, 0)
            else:
                target = (R + self.args.gamma * NQ).detach().clamp_(0, clip_return)

        Q = self.critic(S, A, G)
        critic_loss = F.mse_loss(Q, target)

        # 2. update actor
        A_ = self.actor(S, G)
        actor_loss = - self.critic(S, A_, G).mean()
        actor_loss += self.args.action_l2 * (A_ / self.args.max_action).pow(2).mean()

        # 3. update dynamics model
        for _ in range(self.args.n_dynamics_updates):
            dynamics_loss = self.dynamics.loss_fn(S, A, NS)
            self.dynamics_optim.zero_grad()
            dynamics_loss.backward()
            sync_grads(self.dynamics)
            self.dynamics_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optim.step()
        return actor_loss.item(), critic_loss.item(), dynamics_loss.item()

    def learn(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            t0 = time.time()
            stats = {
                'successes': [],
                'hitting_times': [],
                'actor_losses': [],
                'critic_losses': [],
                'dynamics_losses': [],
            }

        # put something to the buffer first
        self.prefill_buffer()
        if self.args.cuda:
            n_scales = (self.args.max_episode_steps * self.args.rollout_n_episodes // (self.args.n_batches*2)) + 1
        else:
            n_scales = 1

        for epoch in range(self.args.n_epochs):
            AL, CL, DL = [], [], []
            for _ in range(self.args.n_cycles):
                (S, A, AG, G), success = self.collect_rollout()
                self.buffer.store_episode(S, A, AG, G)
                self._update_normalizer(S, A, AG, G)

                for _ in range(n_scales): # scale up for single thread
                    for _ in range(self.args.n_batches):
                        a_loss, c_loss, d_loss = self._update()
                        AL.append(a_loss); CL.append(c_loss); DL.append(d_loss)

                    self._soft_update(self.actor_target, self.actor)
                    self._soft_update(self.critic_target, self.critic)

            global_success_rate, global_hitting_time = self.eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                t1 = time.time()
                AL = np.array(AL); CL = np.array(CL); DL = np.array(DL)
                stats['successes'].append(global_success_rate)
                stats['hitting_times'].append(global_hitting_time)
                stats['actor_losses'].append(AL.mean())
                stats['critic_losses'].append(CL.mean())
                stats['dynamics_losses'].append(DL.mean())
                print(f"[info] epoch {epoch:3d} success rate {global_success_rate:6.4f} | "+\
                        f" actor loss {AL.mean():6.4f} | critic loss {CL.mean():6.4f} | "+\
                        f" dynamics loss {DL.mean():6.4f} | "+\
                        f"time {(t1-t0)/60:6.4f} min")
                self.save_model(stats)
