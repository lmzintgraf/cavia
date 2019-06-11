import torch
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

from rl_utils.optimization import conjugate_gradient
from rl_utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)


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

    def __init__(self, sampler, policy, baseline, gamma=0.95,
                 fast_lr=0.5, tau=1.0, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)

    def inner_loss(self, episodes, params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)

        loss = -weighted_mean(log_probs * advantages, dim=0, weights=episodes.mask)

        return loss

    def adapt(self, episodes, first_order=False, params=None, lr=None):
        """Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """

        if lr is None:
            lr = self.fast_lr

        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)

        # Get the loss on the training episodes
        loss = self.inner_loss(episodes, params=params)

        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(loss, step_size=lr, first_order=first_order, params=params)

        return params, loss

    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        episodes = []
        losses = []
        for task in tasks:
            self.sampler.reset_task(task)
            self.policy.reset_context()
            train_episodes = self.sampler.sample(self.policy, gamma=self.gamma)
            # inner loop (for CAVIA, this only updates the context parameters)
            params, loss = self.adapt(train_episodes, first_order=first_order)
            # rollouts after inner loop update
            valid_episodes = self.sampler.sample(self.policy, params=params, gamma=self.gamma)
            episodes.append((train_episodes, valid_episodes))
            losses.append(loss.item())

        return episodes, losses

    def test(self, tasks, num_steps, batch_size, halve_lr):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.batchsize
        """
        episodes_per_task = []
        for task in tasks:

            # reset context params (for cavia) and task
            self.policy.reset_context()
            self.sampler.reset_task(task)

            # start with blank params
            params = None

            # gather some initial experience and log performance
            test_episodes = self.sampler.sample(self.policy, gamma=self.gamma, params=params, batch_size=batch_size)

            # initialise list which will log all rollouts for the current task
            curr_episodes = [test_episodes]

            for i in range(1, num_steps + 1):

                # lower learning rate after first update (for MAML, as described in their paper)
                if i == 1 and halve_lr:
                    lr = self.fast_lr / 2
                else:
                    lr = self.fast_lr

                # inner-loop update
                params, loss = self.adapt(test_episodes, first_order=True, params=params, lr=lr)

                # get new rollouts
                test_episodes = self.sampler.sample(self.policy, gamma=self.gamma, params=params, batch_size=batch_size)
                curr_episodes.append(test_episodes)

            episodes_per_task.append(curr_episodes)

        self.policy.reset_context()
        return episodes_per_task

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):

            # this is the inner-loop update
            self.policy.reset_context()
            params, _ = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""

        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):

            # do inner-loop update
            self.policy.reset_context()
            params, _ = self.adapt(train_episodes)

            with torch.set_grad_enabled(old_pi is None):

                # get action values after inner-loop update
                pi = self.policy(valid_episodes.observations, params=params)
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

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        old_loss, _, old_pis = self.surrogate_loss(episodes)
        # this part will take higher order gradients through the inner loop:
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes, damping=cg_damping)
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
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            print('no update?')
            vector_to_parameters(old_params, self.policy.parameters())

        return loss

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device
