import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting

    parser.add_argument('--env-name', type=str, default='FetchReach', help='the environment name')

    parser.add_argument('--agent', type=str, default='ddpg', choices=[
        'ddpg', 'her', 'gcsl', 'wgcsl', 'mher'
    ], help='the agent name')
    parser.add_argument('--critic', type=str, default='monolithic', choices=[
        'monolithic', 'bilinear', 'l2', 'asym', 'dn', 'wn', 'asym-max',
        'asym-lse', 'max', 'sym', 'wn-softmax', 'wn-max', 'wn-maxavg',
        'dn-softmax', 'dn-max', 'softmax', 'asym-max-sag',
        'asym-max-sag-latent', 'pqe', 'asym-new',
    ], help='the critic type')

    parser.add_argument('--n-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--n-ensembles', type=int, default=3, help='number of ensembles in dynamics model')
    parser.add_argument('--n-dynamics-updates', type=int, default=2, help='number of ensembles in dynamics model')
    parser.add_argument('--seed', type=int, default=123, help='random seed')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--relabel-rate', type=int, default=0.8, help='ratio to be replace')
    parser.add_argument('--dim-hidden', type=int, default=256, help='hidden dimension of neural networks')
    parser.add_argument('--dim-model-hidden', type=int, default=176, help='hidden dimension of neural networks')
    parser.add_argument('--dim-critic-hidden', type=int, default=176, help='hidden dimension of critic networks')
    parser.add_argument('--dim-new-hidden', type=int, default=174, help='hidden dimension of critic networks')
    parser.add_argument('--dim-embed', type=int, default=16, help='hidden dimension of embeddings')

    parser.add_argument('--dynamics-coef', type=float, default=0.0001, help='dynamics coefficient')
    parser.add_argument('--loss-scale', type=float, default=20.0, help='loss scale')

    parser.add_argument('--negative-reward', action='store_true', help='if reward is {0, 1} or {-1, 0}')
    parser.add_argument('--terminate', action='store_true', help='whether terminate at goal')

    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--save-dir', type=str, default='./results/', help='the path to save the models')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=1024, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor') # 0.001
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic') # 0.001
    parser.add_argument('--lr-transition', type=float, default=0.001, help='the learning rate of the transition model')
    parser.add_argument('--polyak', type=float, default=0.9, help='the average coefficient')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--rollout-n-episodes', type=int, default=20, help='the rollouts per mpi') # 2
    parser.add_argument('--n-init-episodes', type=int, default=0, help='number of initial random episodes')
    parser.add_argument('--eval-rollout-n-episodes', type=int, default=100, help='the number of tests')

    parser.add_argument('--wgcsl-adv-clip', type=float, default=10, help='wgcsl clip value')
    parser.add_argument('--wgcsl-baw-delta', type=float, default=0.15, help='wgcsl clip value')
    parser.add_argument('--wgcsl-baw-max', type=float, default=80, help='wgcsl clip value')

    args = parser.parse_args()
    return args
