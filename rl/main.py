import datetime
import json
import os
import matplotlib.pyplot as plt
import time

import numpy as np
import scipy.stats as st
import torch
from tensorboardX import SummaryWriter

import utils
from arguments import parse_args
from baseline import LinearFeatureBaseline
from metalearner import MetaLearner
from policies.categorical_mlp import CategoricalMLPPolicy
from policies.normal_mlp import NormalMLPPolicy, CaviaMLPPolicy
from sampler import BatchSampler


def get_returns(episodes_per_task):

    # sum up for each rollout, then take the mean across rollouts
    returns = []
    for task_idx in range(len(episodes_per_task)):
        curr_returns = []
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            # compute returns for individual rollouts
            ret = (episodes[update_idx].rewards * episodes[update_idx].mask).sum(dim=0)
            curr_returns.append(ret)
        # result will be: num_evals * num_updates
        returns.append(torch.stack(curr_returns, dim=1))

    # result will be: num_tasks * num_evals * num_updates
    returns = torch.stack(returns)
    returns = returns.reshape((-1, returns.shape[-1]))

    return returns


def total_rewards(episodes_per_task, interval=False):

    returns = get_returns(episodes_per_task).cpu().numpy()

    mean = np.mean(returns, axis=0)
    conf_int = st.t.interval(0.95, len(mean) - 1, loc=mean, scale=st.sem(returns, axis=0))
    conf_int = mean - conf_int
    if interval:
        return mean, conf_int[0]
    else:
        return mean


def main(args):
    print('starting....')

    utils.set_seed(args.seed, cudnn=args.make_deterministic)

    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
                                            'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
                                            '2DNavigation-v0'])

    # subfolders for logging
    method_used = 'maml' if args.maml else 'cavia'
    num_context_params = str(args.num_context_params) + '_' if not args.maml else ''
    output_name = num_context_params + 'lr=' + str(args.fast_lr) + 'tau=' + str(args.tau)
    output_name += '_' + datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_folder = os.path.join(os.path.join(dir_path, 'logs'), args.env_name, method_used, output_name)
    save_folder = os.path.join(os.path.join(dir_path, 'saves'), output_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # initialise tensorboard writer
    writer = SummaryWriter(log_folder)

    # save config file
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)
    with open(os.path.join(log_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size, num_workers=args.num_workers,
                           device=args.device, seed=args.seed)

    if continuous_actions:
        if not args.maml:
            policy = CaviaMLPPolicy(
                int(np.prod(sampler.envs.observation_space.shape)),
                int(np.prod(sampler.envs.action_space.shape)),
                hidden_sizes=(args.hidden_size,) * args.num_layers,
                num_context_params=args.num_context_params,
                device=args.device
            )
        else:
            policy = NormalMLPPolicy(
                int(np.prod(sampler.envs.observation_space.shape)),
                int(np.prod(sampler.envs.action_space.shape)),
                hidden_sizes=(args.hidden_size,) * args.num_layers
            )
    else:
        if not args.maml:
            raise NotImplementedError
        else:
            policy = CategoricalMLPPolicy(
                int(np.prod(sampler.envs.observation_space.shape)),
                sampler.envs.action_space.n,
                hidden_sizes=(args.hidden_size,) * args.num_layers)

    # initialise baseline
    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))

    # initialise meta-learner
    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau,
                              device=args.device)

    for batch in range(args.num_batches):

        # get a batch of tasks
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)

        # do the inner-loop update for each task
        # this returns training (before update) and validation (after update) episodes
        episodes, inner_losses = metalearner.sample(tasks, first_order=args.first_order)

        # take the meta-gradient step
        outer_loss = metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
                                      cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
                                      ls_backtrack_ratio=args.ls_backtrack_ratio)

        # -- logging

        curr_returns = total_rewards(episodes, interval=True)
        print('   return after update: ', curr_returns[0][1])

        # Tensorboard
        writer.add_scalar('policy/actions_train', episodes[0][0].actions.mean(), batch)
        writer.add_scalar('policy/actions_test', episodes[0][1].actions.mean(), batch)

        writer.add_scalar('running_returns/before_update', curr_returns[0][0], batch)
        writer.add_scalar('running_returns/after_update', curr_returns[0][1], batch)

        writer.add_scalar('running_cfis/before_update', curr_returns[1][0], batch)
        writer.add_scalar('running_cfis/after_update', curr_returns[1][1], batch)

        writer.add_scalar('loss/inner_rl', np.mean(inner_losses), batch)
        writer.add_scalar('loss/outer_rl', outer_loss.item(), batch)

        # -- evaluation

        # evaluate for multiple update steps
        if batch % args.test_freq == 0:
            test_tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
            test_episodes = metalearner.test(test_tasks, num_steps=args.num_test_steps,
                                             batch_size=args.test_batch_size, halve_lr=args.halve_test_lr)
            all_returns = total_rewards(test_episodes, interval=True)
            for num in range(args.num_test_steps + 1):
                writer.add_scalar('evaluation_rew/avg_rew ' + str(num), all_returns[0][num], batch)
                writer.add_scalar('evaluation_cfi/avg_rew ' + str(num), all_returns[1][num], batch)

            print('   inner RL loss:', np.mean(inner_losses))
            print('   outer RL loss:', outer_loss.item())

        # -- save policy network
        with open(os.path.join(save_folder, 'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    args = parse_args()

    main(args)
