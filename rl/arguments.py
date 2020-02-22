import argparse
import multiprocessing as mp
import os
import warnings

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Fast Context Adaptation via Meta-Learning (CAVIA)')

    # General
    parser.add_argument('--env-name', type=str,
                        default='AntDir-v1',
                        help='name of the environment')
    parser.add_argument('--cavia', action='store_true', default=False,
                        help='Use CAVIA')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
                        help='use the first-order approximation')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of hidden layers')

    # Inference network
    parser.add_argument('--latent-size', type=int, default=5,
                        help='Dimension of the latent context vector')
    parser.add_argument('--information-bottleneck', action='store_false', default=True,
                        help='False makes latent context deterministic')
    parser.add_argument('--episodes', type=int, default=40,
                        help='Episodes to accumulate in context trajectory')
    parser.add_argument('--context-lr', type=float, default=1e-4,
                        help='learning rate for the context network')
    parser.add_argument('--policy-lr', type=float, default=1e-4,
                        help='learning rate for the policy network')

    # CAVIA
    parser.add_argument('--num-context-params', type=int, default=50,
                        help='number of context parameters')
    parser.add_argument('--halve-test-lr', action='store_true', default=False,
                        help='half LR at test time after one update')
    parser.add_argument('--fast-lr', type=float, default=1.0, 
                        help='learning rate for the 1-step gradient update of CAVIA')

    # Testing
    parser.add_argument('--test-freq', type=int, default=10,
                        help='How often to test multiple updates')
    parser.add_argument('--num-test-steps', type=int, default=5,
                        help='Number of inner loops in the test set')
    parser.add_argument('--test-batch-size', type=int, default=40,
                        help='batch size (number of trajectories) for testing')


    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
                        help='number of rollouts for each individual task ()')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=500,
                        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=40,
                        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
                        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
                        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
                        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
                        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
                        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count(),
                        help='number of workers for trajectories sampling')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--make_deterministic', action='store_true',
                        help='make everything deterministic (set cudnn seed; num_workers=1; '
                             'will slow things down but make them reproducible!)')

    args = parser.parse_args()

    if args.make_deterministic:
        args.num_workers = 1

    # use the GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.output_folder = 'cavia' if args.cavia else 'experiment'

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')

    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    return args
