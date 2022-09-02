import numpy as np


class Sampler(object):
    """
    Helper class to sample transitions for learning.
    Methods like sample_her_transitions will relabel part of trajectories.
    """
    def __init__(self, args, env_reward_func):
        self.relabel_rate = args.relabel_rate
        plus = 1.0 if not args.negative_reward else 0.0
        # make reward to {-1, 0} instead of {0, 1} if negative reward
        self.reward_func = lambda ag, g, c: env_reward_func(ag, g, c) + plus

        if args.negative_reward:
            self.achieved_func = lambda ag, g: env_reward_func(ag, g, None) + 1.0
        else:
            self.achieved_func = lambda ag, g: env_reward_func(ag, g, None)

        self.global_threshold = 80

    def get_closest_goal_state(self, G, t1, t2):
        # S: (batch, T+1, dim_state)
        # G: (batch, T+1, dim_goal)
        # t1: (batch,) the starting state timestep
        # t2: (batch,) the goal state timestep
        # return: max_{t1 <= t <= t2} phi(S_t) = G_t2
        ts = []
        for i in range(G.shape[0]):
            t = t2[i]
            while self.achieved_func(G[i, t], G[i, t2[i]]) > 0.5 and t > t1[i]:
                t -= 1
            ts.append(t+1)
        return np.array(ts).astype(np.int32)

    def sample_ddpg_transitions(self, S, A, AG, G, size):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'NG': NAG_,
        }
        return transition

    def sample_her_transitions(self, S, A, AG, G, size):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        # determine which time step to sample
        her_idx = np.where(np.random.uniform(size=size) < self.relabel_rate)
        future_offset = (np.random.uniform(size=size) * (T - t)).astype(int)
        future_t = (t + 1 + future_offset)[her_idx]
        her_AG = AG[epi_idx[her_idx], future_t]

        #tt = self.get_closest_goal_state(AG[epi_idx[her_idx]], t[her_idx], future_t)
        #GS = S_.copy()
        #GS[her_idx] = S[epi_idx[her_idx], tt].copy()
        mask = np.zeros((size,))
        mask[her_idx] = 1.0

        G_[her_idx] = her_AG
        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'NG': NAG_,
            #'GS': GS,
            'mask': mask,
        }
        return transition

    def sample_mher_transitions(self, S, A, AG, G, size, get_imaginary_rollout, goal_idx):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        S_list = get_imaginary_rollout(S_, G_) # (size, n_steps, dim_state)

        idx = np.where(np.random.uniform(size=size) < 1.0)
        future_offset = (np.random.uniform(size=size) * (T - t)).astype(int)
        future_t = (t + 1 + future_offset)[idx]
        future_AG = AG[epi_idx[idx], future_t]

        # imaginary relabel
        relabel_idx = (np.random.uniform(size=size) < 0.8)
        step_idx = np.random.randint(S_list.shape[1], size=size)
        last_state = S_list[np.arange(size), step_idx] # (size, dim_state)
        imaginary_goal = last_state[..., goal_idx.numpy()][relabel_idx]
        G_[relabel_idx] = imaginary_goal

        her_idx = (np.random.uniform(size=size) < 0.4)
        G_[relabel_idx & her_idx] = future_AG[relabel_idx & her_idx]
        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)

        transition = {
            'S'  : S_,
            'NS' : NS_,
            'A'  : A_,
            'G'  : G_,
            'R'  : R_,
        }
        return transition

    def sample_gcsl_transitions(self, S, A, AG, G, size):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        # determine which time step to sample
        her_idx = np.where(np.random.uniform(size=size) < self.relabel_rate)
        future_offset = (np.random.uniform(size=size) * (T - t)).astype(int)
        future_t = (t + 1 + future_offset)[her_idx]
        her_AG = AG[epi_idx[her_idx], future_t]

        G_[her_idx] = her_AG
        transition = {
            'S' : S_,
            'A' : A_,
            'G' : G_,
        }
        return transition

    def sample_wgcsl_transitions(self, S, A, AG, G, size,
                                 args, q_func, r_func, advque):

        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        # determine which time step to sample
        her_idx = np.where(np.random.uniform(size=size) < self.relabel_rate)
        future_offset = (np.random.uniform(size=size) * (T - t)).astype(int)
        future_t = (t + 1 + future_offset)[her_idx]
        her_AG = AG[epi_idx[her_idx], future_t]

        G_[her_idx] = her_AG
        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)

        W_ = pow(args.gamma, future_offset).reshape(-1, 1)
        adv = args.gamma * q_func(NS_, G_) - \
                q_func(S_, G_) + R_
        advque.update(adv)
        self.global_threshold = min(self.global_threshold + \
                args.wgcsl_baw_delta, args.wgcsl_baw_max)
        threshold = advque.get(self.global_threshold)

        W_ *= np.clip(np.exp(adv), 0, args.wgcsl_adv_clip)

        positive = adv.copy()
        positive[adv >= threshold] = 1
        positive[adv < threshold] = 0.05
        W_ *= positive
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'W' : W_,
        }
        return transition
