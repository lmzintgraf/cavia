import torch
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

from rl_utils.optimization import conjugate_gradient
from rl_utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)

import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


class MetaLearner(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the inner loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, 
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """

    def __init__(self, sampler, policy, baseline, context_encoder=None, 
    policy_lr=3e-4, context_lr=3e-4, gamma=0.95, tau=1.0, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.context_encoder = context_encoder
        self.gamma = gamma
        self.policy_lr = policy_lr
        self.context_lr = context_lr
        self.tau = tau
        self.kl_lambda = 0.1
        self.to(device)
        
        self.context_optimizer = optim.Adam(
            self.context_encoder.network.parameters(),
            lr=self.context_lr,
        )

    def get_z(self, episodes):
        self.context_encoder.sample_z()
        task_z = self.context_encoder.z
        task_z = [z.repeat(episodes.observations.shape[1], 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        task_z = task_z.unsqueeze(0)
        task_z = task_z.repeat(episodes.observations.shape[0],1,1)

        in_ = torch.cat([episodes.observations, task_z], dim=2)

        return in_

    def inner_loss(self, episodes, in_, params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """

        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(in_, params=params)

        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)

        loss = -weighted_mean(log_probs * advantages, dim=0, weights=episodes.mask)

        return loss

    def adapt(self, episodes, context):
        
        losses = []
        for (train_episodes, valid_episodes), context_task in zip(episodes, context):
            
            self.context_encoder.clear_z()
            
            _, task_z = self.context_encoder(train_episodes.observations, context_task, self.policy)
            
            # KL constraint on z if probabilistic
            self.context_optimizer.zero_grad()
            kl_div = self.context_encoder.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div        
            kl_loss.backward(retain_graph=True)
            
            self.baseline.fit(train_episodes)               
            in_ = torch.cat([train_episodes.observations, task_z], dim=2)
            reinforce_loss = self.inner_loss(train_episodes, in_)
            reinforce_loss.backward()
            # print('REINFORCE ', reinforce_loss.item())

            self.context_optimizer.step()
            plot_grad_flow(self.context_encoder.network.named_parameters())

            loss = kl_loss + reinforce_loss
            losses.append((reinforce_loss.item(), kl_loss.item(), loss.item()))

        return losses

    def sample(self, tasks, episodes=None):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """

        episodes = []
        context = []
        z_train = []
        for task in tasks:
            self.sampler.reset_task(task)
            
            # initialize cT = {}
            self.context_encoder.clear_z()
            
            # collect rollout T_train
            train_episodes = self.sampler.sample(
                policy=self.policy, 
                context_encoder=self.context_encoder,
                gamma=self.gamma
            )

            # inner loop
            context_task = self.context_encoder.context
            self.context_encoder.infer_posterior(self.context_encoder.context)

            # collect rollout T_test
            valid_episodes = self.sampler.sample(
                policy=self.policy, 
                context_encoder=self.context_encoder,
                gamma=self.gamma
            )
            
            episodes.append((train_episodes, valid_episodes))
            context.append(context_task)     # TODO context is collect from train and test data?
            z_train.append([self.context_encoder.z_means.data.cpu().numpy().mean(),
                            self.context_encoder.z_vars.data.cpu().numpy().mean()])

        return episodes, context, z_train

    def test(self, tasks, num_steps, batch_size):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.batchsize
        """

        episodes_per_task = []
        z_test = []
        for task in tasks:

            self.sampler.reset_task(task)
            self.context_encoder.clear_z()

            # gather some initial experience and log performance
            test_episodes = self.sampler.sample(
                self.policy, 
                gamma=self.gamma, 
                context_encoder=self.context_encoder,
                batch_size=batch_size
            )

            # initialise list which will log all rollouts for the current task
            curr_episodes = [test_episodes]
            z_step = []
            for i in range(1, num_steps + 1):

                context = self.context_encoder.context
                self.context_encoder.clear_z()
                self.context_encoder.infer_posterior(context)

                # get new rollouts
                test_episodes = self.sampler.sample(
                    self.policy, 
                    gamma=self.gamma, 
                    context_encoder=self.context_encoder,
                    batch_size=batch_size
                )
                
                curr_episodes.append(test_episodes)
                z_step.append([self.context_encoder.z_means.data.cpu().numpy().mean(),
                                self.context_encoder.z_vars.data.cpu().numpy().mean()])
                

            z_test.append(z_step)
            episodes_per_task.append(curr_episodes)

        return episodes_per_task, z_test

    def kl_divergence(self, episodes, context, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), context_task, old_pi in zip(episodes, context, old_pis):

            self.context_encoder.clear_z()
            self.context_encoder.infer_posterior(context_task)

            in_ = self.get_z(valid_episodes)
            pi = self.policy(in_)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, context, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""

        def _product(vector):
            kl = self.kl_divergence(episodes, context)
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def surrogate_loss(self, episodes, context, old_pis=None):
        losses, kls, pis, = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi, context_task in zip(episodes, old_pis, context):
           
            self.context_encoder.clear_z()
            self.context_encoder.infer_posterior(context_task)
           
            with torch.set_grad_enabled(old_pi is None):

                in_ = self.get_z(valid_episodes)
                pi = self.policy(in_)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions)
                             - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages, dim=0, weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
                kls.append(kl)

        return torch.mean(torch.stack(losses, dim=0)), torch.mean(torch.stack(kls, dim=0)), pis

    def step(self, episodes, context, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """ Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4])."""
        
        old_loss, _, old_pis = self.surrogate_loss(episodes, context)
        # print('SURROGATE ', old_loss.item())

        # update context network
        in_loss = self.adapt(episodes, context)

        # this part will take higher order gradients through the inner loop:
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes, context, damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step, self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, context, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            print('no update?')
            vector_to_parameters(old_params, self.policy.parameters())

        return loss, in_loss

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.context_encoder.to(device, **kwargs)
        self.device = device
