import datetime
import json
import os
import matplotlib.pyplot as plt
import time

import numpy as np
import scipy.stats as st
import torch
from tensorboardX import SummaryWriter
from fcm_notifier import FCMNotifier

import utils
from arguments import parse_args
from baseline import LinearFeatureBaseline
from metalearner import MetaLearner
from policies.categorical_mlp import CategoricalMLPPolicy
from policies.normal_mlp import NormalMLPPolicy, CaviaMLPPolicy, FlattenMlp, Mlp
from sampler import BatchSampler

from context import ContextEncoder


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
    print('Starting....')

    utils.set_seed(args.seed, cudnn=args.make_deterministic)

    continuous_actions = (args.env_name in ['AntVel-v1', 
                                            'AntDir-v1',
                                            'AntPos-v0', 
                                            'HalfCheetahVel-v1', 
                                            'HalfCheetahDir-v1',
                                            '2DNavigation-v0'])

    # subfolders for logging
    method_used = 'experiment'
    num_context_params = str(args.num_context_params) + '_' 
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
    notifier = FCMNotifier()

    # save config file
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    with open(os.path.join(log_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSampler(
        args.env_name, 
        batch_size = args.fast_batch_size, 
        num_workers = args.num_workers,
        device = args.device, 
        seed = args.seed
    )

    # instantiate networks
    obs_dim, action_dim, reward_dim = sampler.get_dim()
    latent_dim = args.latent_size
    context_encoder_input_dim = obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if args.information_bottleneck else latent_dim

    if continuous_actions:
        policy = NormalMLPPolicy(
            obs_dim + latent_dim,
            action_dim,
            hidden_sizes=(args.hidden_size,) * args.num_layers
        )

        context_network = Mlp(
            context_encoder_input_dim,
            context_encoder_output_dim,
            hidden_sizes=(args.hidden_size,) * 3
        )

        context = ContextEncoder(
            latent_dim,
            context_network,
            args.information_bottleneck,
            device=args.device
        )

    else:
        raise NotImplementedError

    # initialise baseline
    baseline = LinearFeatureBaseline(obs_dim)

    # initialise meta-learner
    metalearner = MetaLearner(
        sampler,
        policy,
        baseline, 
        context_encoder = context,
        gamma = args.gamma, 
        fast_lr = args.fast_lr, 
        policy_lr = args.policy_lr,
        context_lr = args.context_lr,
        tau = args.tau,
        device = args.device
    )

    for batch in range(args.num_batches):
        # sample a batch of tasks
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)

        # do the inner-loop update for each task
        # this returns training (before update) and validation (after update) episodes
        episodes, inner_losses = metalearner.sample(tasks, episodes=args.episodes, first_order=args.first_order)
        
        # take the meta-gradient step
        outer_loss = metalearner.step(
            episodes, 
            max_kl = args.max_kl, 
            cg_iters = args.cg_iters,
            cg_damping = args.cg_damping, 
            ls_max_steps = args.ls_max_steps,
            ls_backtrack_ratio = args.ls_backtrack_ratio
        )

        # ---------- logging ----------
        curr_returns = total_rewards(episodes, interval=True)
        print('Return after update: ', curr_returns[0][1])
        
        # Tensorboard

        # Actions mean per batch of episodes
        actions_train_fwd = []
        actions_test_fwd = []
        actions_train_back = []
        actions_test_back = []
        for i in range(len(episodes)):
            if tasks[i]['direction'] == 1:
                actions_train_fwd.append(episodes[i][0].actions.mean().item())
                actions_test_fwd.append(episodes[i][1].actions.mean().item())
            else:
                actions_train_back.append(episodes[i][0].actions.mean().item())
                actions_test_back.append(episodes[i][1].actions.mean().item())
                
        # writer.add_scalar('policy/actions_train', episodes[0][0].actions.mean(), batch)
        # writer.add_scalar('policy/actions_test', episodes[0][1].actions.mean(), batch)
        
        writer.add_scalar('policy/actions_train_fwd', np.array(actions_train_fwd).mean(), batch)
        writer.add_scalar('policy/actions_test_fwd', np.array(actions_test_fwd).mean(), batch)
        writer.add_scalar('policy/actions_train_back', np.array(actions_train_back).mean(), batch)
        writer.add_scalar('policy/actions_test_back', np.array(actions_test_back).mean(), batch)

        # Reward 
        writer.add_scalar('running_returns/before_update', curr_returns[0][0], batch)
        writer.add_scalar('running_returns/after_update', curr_returns[0][1], batch)

        # Confidence interval of the mean 
        writer.add_scalar('running_cfis/before_update', curr_returns[1][0], batch)
        writer.add_scalar('running_cfis/after_update', curr_returns[1][1], batch)

        # Loss
        writer.add_scalar('loss/reinforce', np.mean(inner_losses, axis=0)[0], batch)
        writer.add_scalar('loss/kl_divergence', np.mean(inner_losses, axis=0)[1], batch)
        writer.add_scalar('loss/inner_rl', np.mean(inner_losses, axis=0)[2], batch)
        writer.add_scalar('loss/outer_rl', outer_loss.item(), batch)

        # # Inference
        # writer.add_scalar('inference_train/z_means', np.mean(z_train, axis=0)[0], batch)
        # writer.add_scalar('inference_train/z_vars', np.mean(z_train, axis=0)[1], batch)


        notifier.notify(
            title='', 
            returnAfter=round(curr_returns[0][1], 4), 
            T_innerLoss=round(np.mean(inner_losses), 4), 
            T_outerLoss=round(outer_loss.item(), 4)
        )

        # ----- evaluation -----
        # evaluate for multiple update steps
        if batch % args.test_freq == 0:
            test_tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
            test_episodes, z_test = metalearner.test(
                test_tasks, 
                num_steps = args.num_test_steps,
                batch_size = args.test_batch_size, 
                halve_lr = args.halve_test_lr
            )
            all_returns = total_rewards(test_episodes, interval=True)

            for num in range(args.num_test_steps + 1):
                writer.add_scalar('evaluation_rew/avg_rew ' + str(num), all_returns[0][num], batch)
                writer.add_scalar('evaluation_cfi/avg_rew ' + str(num), all_returns[1][num], batch)
                writer.add_scalar('evaluation_z_means/z_mean ' + str(num), np.mean(z_test[num], axis=0)[0], batch)
                writer.add_scalar('evaluation_z_vars/z_var ' + str(num), np.mean(z_test[num], axis=0)[1], batch)

            print('Inner RL loss:', np.mean(inner_losses))
            print('Outer RL loss:', outer_loss.item())
            notifier.notify(
                title='', 
                E_innerLoss=round(np.mean(inner_losses), 4), 
                E_outerLoss=round(outer_loss.item(), 4)
            )

        # ----- save policy network -----
        with open(os.path.join(save_folder, 'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(policy.state_dict(), f)
            
        with open(os.path.join(save_folder, 'context-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(context_network.state_dict(), f)


if __name__ == '__main__':
    args = parse_args()

    main(args)
