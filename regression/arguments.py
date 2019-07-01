import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Fast Context Adaptation via Meta-Learning (CAVIA),'
                                                 'Regression experiments')

    parser.add_argument('--task', type=str, default='sine', help='problem setting: sine or celeba')

    parser.add_argument('--n_iter', type=int, default=50000, help='number of meta-iterations')

    parser.add_argument('--tasks_per_metaupdate', type=int, default=25)

    parser.add_argument('--k_meta_train', type=int, default=10, help='data points in task training set (during meta training, inner loop)')
    parser.add_argument('--k_meta_test', type=int, default=10, help='data points in task test set (during meta training, outer loop)')
    parser.add_argument('--k_shot_eval', type=int, default=10, help='data points in task training set (during evaluation)')

    parser.add_argument('--lr_inner', type=float, default=1.0, help='inner-loop learning rate (task-specific)')
    parser.add_argument('--lr_meta', type=float, default=0.001, help='outer-loop learning rate')

    parser.add_argument('--num_inner_updates', type=int, default=1, help='number of inner-loop updates (during training)')

    parser.add_argument('--num_context_params', type=int, default=5, help='number of context parameters (added at first layer)')
    parser.add_argument('--num_hidden_layers', type=int, nargs='+', default=[40, 40])

    parser.add_argument('--first_order', action='store_true', default=False, help='run first-order version')

    parser.add_argument('--maml', action='store_true', default=False, help='run MAML')
    parser.add_argument('--seed', type=int, default=42)

    # commands specific to the CelebA image completion task
    parser.add_argument('--use_ordered_pixels', action='store_true', default=False)

    args = parser.parse_args()

    # use the GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args
