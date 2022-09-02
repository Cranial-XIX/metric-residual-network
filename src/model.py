import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


################################################################################
#
# Policy Network
#
################################################################################


class Actor(nn.Module):
    """
    The policy network
    """
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal

        self.net = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_action),
            nn.Tanh()
        )

    def forward(self, s, g):
        x = torch.cat([s, g], -1)
        actions = self.max_action * self.net(x)
        return actions


################################################################################
#
# Critic Networks
#
################################################################################


class CriticMonolithic(nn.Module):
    """
    Monolithic Action-value Function Network (Q)
    """
    def __init__(self, args):
        super(CriticMonolithic, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal

        self.net = nn.Sequential(
            nn.Linear(dim_state+dim_goal+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, 1)
        )

    def forward(self, s, a, g):
        x = torch.cat([s, a/self.max_action, g], -1)
        q_value = self.net(x)
        return q_value


class CriticBilinear(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """
    def __init__(self, args):
        super(CriticBilinear, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed

        self.f = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed)
        )
        self.phi = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed)
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        ff = self.f(x1)
        pp = self.phi(x2)
        q_value = ff.unsqueeze(1).bmm(pp.unsqueeze(-1)).view(-1, 1)
        return q_value


class CriticL2(nn.Module):
    """
    L2-norm Action-value Function
    Q(s, a, g) = ||f(s, a) - phi(s, g)||^2
    """
    def __init__(self, args):
        super(CriticL2, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed

        self.f = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed)
        )
        self.phi = nn.Sequential(
            nn.Linear(dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed)
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        ff = self.f(x1)
        pp = self.phi(g)
        q_value = -(ff - pp).pow(2).mean(-1, keepdims=True)
        return q_value


class CriticAsym(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """
    def __init__(self, args):
        super(CriticAsym, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1-sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = (F.softmax(res, -1) * res).sum(-1, keepdims=True)
        dist = dist_s + dist_a
        return -dist

class CriticAsymNew(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """
    def __init__(self, args):
        super(CriticAsymNew, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_new_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed
        self.dim_embed = args.dim_embed
        self.gamma = args.gamma

        self.rew = nn.Sequential(
            nn.Linear(dim_state+dim_action+dim_goal, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1))

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        x3 = torch.cat([s, a/self.max_action, g], -1)
        r = self.rew(x3)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1-sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = (F.softmax(res, -1) * res).sum(-1, keepdims=True)
        dist = dist_s + dist_a
        return r - dist

    def sep_forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        x3 = torch.cat([s, a/self.max_action, g], -1)
        r = self.rew(x3)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1-sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = (F.softmax(res, -1) * res).sum(-1, keepdims=True)
        dist = dist_s + dist_a
        return r.detach() -dist, r


class CriticSym(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """
    def __init__(self, args):
        super(CriticSym, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dim_hidden, 450),
            nn.ReLU(inplace=True),
            nn.Linear(450, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        dist_s = (sym1-sym2).pow(2).mean(-1, keepdims=True)
        return -dist_s


class CriticSoftmax(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """
    def __init__(self, args):
        super(CriticSoftmax, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        res = F.relu(asym1 - asym2)
        dist_a = (F.softmax(res, -1) * res).sum(-1, keepdims=True)
        return -dist_a

class CriticMax(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """
    def __init__(self, args):
        super(CriticMax, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, 450),
            nn.ReLU(inplace=True),
            nn.Linear(450, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        res = F.relu(asym1 - asym2)
        #dist_a = (F.softmax(res, -1) * res).sum(-1, keepdims=True)
        dist_a = res.max(-1)[0].view(-1, 1)
        return -dist_a


class CriticAsymMax(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """
    def __init__(self, args):
        super(CriticAsymMax, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1-sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = res.max(-1)[0].view(-1, 1)
        dist = dist_s + dist_a
        return -dist

class CriticAsymMaxSAG(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """
    def __init__(self, args):
        super(CriticAsymMaxSAG, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, a/self.max_action, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1-sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = res.max(-1)[0].view(-1, 1)
        dist = dist_s + dist_a
        return -dist

class CriticAsymMaxSAGLatent(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """
    def __init__(self, args):
        super(CriticAsymMaxSAGLatent, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed
        self.dim_embed = args.dim_embed

        dh = 140
        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dh),
            nn.ReLU(inplace=True),
        )
        self.g_emb = nn.Linear(dim_goal, dh)
        self.phi_emb = nn.Sequential(
            nn.Linear(dh*2, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dh),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dh, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dh, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        #x2 = torch.cat([s, a/self.max_action, g], -1)
        fh = self.f_emb(x1)
        x2 = torch.cat([fh, self.g_emb(g)], -1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1-sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = res.max(-1)[0].view(-1, 1)
        dist = dist_s + dist_a
        return -dist

class CriticAsymLSE(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """
    def __init__(self, args):
        super(CriticAsymLSE, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1-sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        #dist_a = res.max(-1)[0].view(-1, 1)
        dist_a = (res.exp().sum(-1, keepdims=True)+1e-4).log()
        dist = dist_s + dist_a
        return -dist


class CriticModel(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """
    def __init__(self, args):
        super(CriticModel, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_model_hidden = args.dim_model_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_model_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_model_hidden, dim_model_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_model_hidden, dim_state),
        )

        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_model_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_model_hidden, dim_model_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_model_hidden, dim_state),
        )

        self.sym = nn.Sequential(
            nn.Linear(dim_state, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_state, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, ns, gs):
        sym1 = self.sym(ns)
        sym2 = self.sym(gs)
        asym1 = self.asym(ns)
        asym2 = self.asym(gs)
        dist_s = (sym1-sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = (F.softmax(res, -1) * res).sum(-1, keepdims=True)
        dist = dist_s + dist_a
        return -dist

    def forward_model(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        ns = self.f_emb(x1) + s
        gs = self.phi_emb(x2)
        return ns, gs


################################################################################
#
# Deep Norm Critic
#
################################################################################

class ConstrainedLinear(nn.Linear):
  def forward(self, x):
    return F.linear(x, torch.min(self.weight ** 2, torch.abs(self.weight)))


class MaxAvgGlobalActivation(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.alpha = nn.Parameter(-torch.ones(1))
    if args.cuda:
        self.alpha.cuda()

  def forward(self, x):
    alpha = torch.sigmoid(self.alpha)
    return alpha * x.max(dim=-1)[0] + (1 - alpha) * x.mean(dim=-1)


class ConcaveActivation(nn.Module):
  def __init__(self, num_features, concave_activation_size, args):
    super().__init__()
    assert concave_activation_size > 1

    self.bs_nonzero = nn.Parameter(1e-3 * torch.randn((1, num_features, concave_activation_size - 1)) - 1)
    self.bs_zero    = torch.zeros((1, num_features, 1))
    self.ms = nn.Parameter(1e-3 * torch.randn((1, num_features, concave_activation_size)))

    if args.cuda:
        self.bs_nonzero.cuda()
        self.bs_zero = self.bs_zero.cuda()
        self.ms.cuda()

  def forward(self, x):
    bs = torch.cat((F.softplus(self.bs_nonzero), self.bs_zero), -1)
    ms = 2 * torch.sigmoid(self.ms)
    x = x.unsqueeze(-1)

    x = x * ms + bs
    return x.min(-1)[0]


class ReduceMetric(nn.Module):
  def __init__(self, mode, args):
    super().__init__()
    if mode == 'avg':
      self.forward = self.avg_forward
    elif mode == 'max':
      self.forward = self.max_forward
    elif mode == 'maxavg':
      self.maxavg_activation = MaxAvgGlobalActivation(args)
      self.forward = self.maxavg_forward
    elif mode == 'softmax':
        self.forward = self.softmax_forward
    else:
      raise NotImplementedError

  def maxavg_forward(self, x):
    return self.maxavg_activation(x)

  def max_forward(self, x):
    return x.max(-1)[0]

  def avg_forward(self, x):
    return x.mean(-1)

  def softmax_forward(self, x):
    return (F.softmax(x, -1) * x).sum(-1)


class WideNormCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed))

        self.symmetric = False
        num_features = dim_embed
        self.num_components = 62
        concave_activation_size = 8
        if 'softmax' in args.critic:
            mode = 'softmax'
        elif 'maxavg' in args.critic:
            mode = 'maxavg'
        elif 'max' in args.critic:
            mode = 'max'
        else:
            mode = 'avg'
        self.component_size = args.dim_embed

        output_size = self.component_size * self.num_components
        if not self.symmetric:
            num_features = num_features * 2
            self.f = ConstrainedLinear(num_features, output_size)
        else:
            self.f = nn.Linear(num_features, output_size)
        self.concave_activation = ConcaveActivation(self.num_components, concave_activation_size, args) if concave_activation_size else nn.Identity()
        self.reduce_metric = ReduceMetric(mode, args)

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        x = self.phi(self.f_emb(x1))
        y = self.phi(self.phi_emb(x2))

        h = x - y
        if not self.symmetric:
            h = torch.cat((F.relu(h), F.relu(-h)), -1)
        h = torch.reshape(self.f(h), (-1, self.num_components, self.component_size))
        h = torch.norm(h, dim=-1)
        h = self.concave_activation(h)
        dist = self.reduce_metric(h).view(-1, 1)
        return -dist


class DeepNormCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed))

        if 'softmax' in args.critic:
            mode = 'softmax'
        elif 'maxavg' in args.critic:
            mode = 'maxavg'
        elif 'max' in args.critic:
            mode = 'max'
        else:
            mode = 'avg'
        layers = [162, 162]
        self.symmetric = False
        concave_activation_size = 8
        num_features = dim_embed
        self.Us = nn.ModuleList([nn.Linear(num_features, layers[0], bias=False)])
        self.Ws = nn.ModuleList([])

        for in_features, out_features in zip(layers[:-1], layers[1:]):
          self.Us.append(nn.Linear(num_features, out_features, bias=False))
          self.Ws.append(ConstrainedLinear(in_features, out_features, bias=False))

        self.activation = nn.ReLU()
        self.concave_activation = ConcaveActivation(layers[-1], concave_activation_size, args) if concave_activation_size else nn.Identity()
        self.reduce_metric = ReduceMetric(mode, args)

    def _asym_fwd(self, h):
        h1 = self.Us[0](h)
        for U, W in zip(self.Us[1:], self.Ws):
            h1 = self.activation(W(h1) + U(h))
        return h1

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        x = self.phi(self.f_emb(x1))
        y = self.phi(self.phi_emb(x2))

        h = x - y
        if self.symmetric:
            h = self._asym_fwd(h) + self._asym_fwd(-h)
        else:
            h = self._asym_fwd(h)
        h = self.concave_activation(h)
        dist = self.reduce_metric(h).view(-1, 1)
        return -dist

################################################################################
#
# PQE https://openreview.net/pdf?id=y0VvIg25yk
#
################################################################################

#from pqe import PQE
from src.pqe import PQE

class CriticPQE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal
        dim_embed  = args.dim_embed

        self.discounted_pqe = PQE(num_quasipartition_mixtures=16,
                                  num_poisson_processes_per_quasipartition=4)

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.Sequential(
            nn.Linear(dim_hidden, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 16*4))

    def forward(self, s, a, g):
        x1 = torch.cat([s, a/self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        zx = self.encoder(self.f_emb(x1))
        zy = self.encoder(self.phi_emb(x2))
        pred = self.discounted_pqe(zx, zy)
        return -pred.view(-1, 1)
