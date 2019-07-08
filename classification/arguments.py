import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Fast Context Adaptation via Meta-Learning (CAVIA),'
                                                 'Clasification experiments.')

    parser.add_argument('--n_iter', type=int, default=60000, help='number of meta-iterations')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--tasks_per_metaupdate', type=int, default=16, help='number of tasks in each batch per meta-update')

    parser.add_argument('--n_way', type=int, default=5, help='number of object classes to learn')
    parser.add_argument('--k_shot', type=int, default=1, help='number of examples per class to learn from')
    parser.add_argument('--k_query', type=int, default=15, help='number of examples to evaluate on (in outer loop)')

    parser.add_argument('--lr_inner', type=float, default=1.0, help='inner-loop learning rate (per task)')
    parser.add_argument('--lr_meta', type=float, default=0.001, help='outer-loop learning rate (used with Adam optimiser)')
    parser.add_argument('--lr_meta_decay', type=float, default=0.9, help='decay factor for meta learning rate')

    parser.add_argument('--num_grad_steps_inner', type=int, default=2, help='number of gradient steps in inner loop (during training)')
    parser.add_argument('--num_grad_steps_eval', type=int, default=2, help='number of gradient updates at test time (for evaluation)')

    parser.add_argument('--first_order', action='store_true', default=False, help='run first order approximation of CAVIA')

    # network architecture

    parser.add_argument('--num_context_params', type=int, default=100, help='number of context parameters')
    parser.add_argument('--context_in', nargs='+', default=[False, False, True, False, False], help='per layer, indicate if context params are added')

    parser.add_argument('--imsize', type=int, default=84, help='downscale images to this size')
    parser.add_argument('--no_max_pool', action='store_true', default=False, help='turn off max pooling in CNN')
    parser.add_argument('--num_filters', type=int, default=32, help='number of filters per conv-layer')
    parser.add_argument('--nn_initialisation', type=str, default='kaiming', help='initialisation type (kaiming, xavier, None)')

    parser.add_argument('--num_film_hidden_layers', type=int, default=0, help='mumber of hidden layers used for FiLM')

    #

    parser.add_argument('--data_path', type=str, default='./data/miniimagenet/', help='folder which contains image data')
    parser.add_argument('--rerun', action='store_true', default=False,
                        help='Re-run experiment (will override previously saved results)')

    args = parser.parse_args()

    # use the GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(args.device))

    return args
