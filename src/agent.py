import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from utils.agent_utils import *


class Agent(nn.Module):
    def __init__(self, envs, network_params, norm_input=False):
        super().__init__()

        self.action_space = envs.action_space
        
        self.units = network_params.units

        if network_params.activation == 'relu':
            self.activation = nn.ReLU()
        elif network_params.activation == 'tanh':
            self.activation = nn.Tanh()
        elif network_params.activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif network_params.activation == 'elu':
            self.activation = nn.ELU()
        elif network_params.activation == 'selu':
            self.activation = nn.SELU()
        elif network_params.activation == 'swish':
            self.activation = nn.SiLU()
        elif network_params.activation == 'gelu':
            self.activation = nn.GELU()
        elif network_params.activation == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.Identity()

        in_size = np.array(envs.single_observation_space.shape).prod()
        layers = []
        for unit in self.units:
            layers.append(layer_init(nn.Linear(in_size, unit)))
            layers.append(self.activation)
            in_size = unit
        layers.append(layer_init(nn.Linear(in_size, 1), std=1.0))
        self.critic = nn.Sequential(*layers)

        in_size = np.array(envs.single_observation_space.shape).prod()
        layers = []
        for unit in self.units:
            layers.append(layer_init(nn.Linear(in_size, unit)))
            layers.append(self.activation)
            in_size = unit
        layers.append(layer_init(nn.Linear(in_size, np.prod(envs.single_action_space.shape)), std=0.01))
        self.actor_mean = nn.Sequential(*layers)

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        self.norm_input = norm_input

        if norm_input:
            self.running_mean_std = RunningMeanStd(envs.single_observation_space.shape)

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation) if self.norm_input else observation

    def get_value(self, x):
        # normalize observation x
        if self.norm_input:
            x = self.norm_obs(x)
        return self.critic(x)

    def reset_final_layer(self, device):
        self.actor_mean[-1] = layer_init(nn.Linear(self.units[-1], np.prod(self.action_space.shape)), std=0.01).to(device)
        self.critic[-1] = layer_init(nn.Linear(self.units[-1], 1), std=1.0).to(device)

    def get_action_and_value(self, x, action=None):
        # normalize observation x
        if self.norm_input:
            x = self.norm_obs(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # Failsafe mechanism (hopefully, we will never need it)
        try:
            probs = Normal(action_mean, action_std)
        except:
            try:
                probs = Normal(action_mean, torch.ones_like(action_std) * 1e-5)
            except:
                action_low = torch.tensor(self.action_space.low).to(action_mean.device)
                action_high = torch.tensor(self.action_space.high).to(action_mean.device)
                probs = Normal(torch.rand_like(action_mean) * (action_low - action_high) + action_high, torch.ones_like(action_std) * 1e-5)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
