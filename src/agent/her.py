import copy
import numpy as np
import time
import torch

from src.model import *
from src.replay_buffer import ReplayBuffer
from src.utils import *
from src.sampler import Sampler
from src.agent.ddpg import DDPG


class HER(DDPG):
    """
    Hindsight Experience Replay agent
    """
    def __init__(self, args, env):
        super().__init__(args, env)
        self.sample_func = self.sampler.sample_her_transitions
        self.buffer = ReplayBuffer(args, self.sample_func)
