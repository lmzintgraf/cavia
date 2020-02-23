import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

from arguments import parse_args

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared

class ContextEncoder(nn.Module):

    def __init__(
        self,
        latent_dim,
        network,
        information_bottleneck,
        device
    ):

        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.network = network
        self.use_ib = information_bottleneck
        self.use_next_obs_in_context = False
        self.kl_lambda = 0.1

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        """Reset q(z|c) to the prior sample a new z from the prior."""

        # reset distribution over z to the prior
        mu = torch.zeros(num_tasks, self.latent_dim).to(device=self.device)
        
        if self.use_ib:
            var = torch.ones(num_tasks, self.latent_dim).to(device=self.device)
        else:
            var = torch.zeros(num_tasks, self.latent_dim).to(device=self.device)
        
        self.z_means = mu
        self.z_vars = var

        # sample a new z from the prior
        self.sample_z()

        # reset the context collected so far
        self.context = None

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        """ Append single transition to the current context."""

        o, a, r, no, d, info = inputs
        r = np.reshape(r, (r.shape[0], 1))

        o = torch.from_numpy(o[None, None, ...]).float().to(device=self.device)
        a = torch.from_numpy(a[None, None, ...]).float().to(device=self.device)
        r = torch.from_numpy(r[None, None, ...]).float().to(device=self.device)
        no = torch.from_numpy(no[None, None, ...]).float().to(device=self.device)

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=3)

        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def infer_posterior(self, context=None):
        """Compute q(z|c) as a function of input context and sample new z from it."""

        if context is None:
            context = self.context
        
        params = self.network(context)
        params = params.view(context.size(0), -1, self.network.output_size)

        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
            
        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def compute_kl_div(self):
        """ compute KL( q(z|c) || r(z) ) """
        prior = torch.distributions.Normal(torch.zeros(self.latent_dim).to(self.device), torch.ones(self.latent_dim).to(self.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum
        
    def get_action(self, policy, obs, w, params=None, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        z = torch.cat(w*[z])
        in_ = torch.cat([obs, z], dim=1)
        return policy(in_, params=params).sample()
    
    def forward(self, obs, context, policy):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)
        self.sample_z()

        task_z = self.z        
        
        task_z = [z.repeat(obs.shape[1], 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        task_z = task_z.unsqueeze(0)
        task_z = task_z.repeat(obs.shape[0],1,1)
        in_ = torch.cat([obs, task_z], dim=2)

        policy_outputs = policy(in_)

        return policy_outputs, task_z
