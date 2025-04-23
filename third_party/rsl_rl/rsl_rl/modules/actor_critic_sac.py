
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class SAC_Actor(nn.Module):
    def __init__(
        self, 
        obs_dim, 
        action_dim,
        actor_hidden_dims=None,
        activation="relu",
        log_std_init=-3,
        clip_mean=2.0,
        **kwargs
    ):
        super(SAC_Actor, self).__init__()
        

        if kwargs:
            unexpected_keys = [key for key in kwargs.keys()]
            if unexpected_keys:
                print(f"SAC_Actor.__init__ got unexpected arguments, which will be ignored: {unexpected_keys}")
        
        self.action_dim = action_dim
        self.clip_mean = clip_mean


        net_arch = actor_hidden_dims if actor_hidden_dims is not None else [256, 256]
        

        layers = []
        input_dim = obs_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_activation(activation))
            input_dim = hidden_dim
        self.net = nn.Sequential(*layers)


        self.mean_layer = nn.Linear(net_arch[-1], action_dim)
        self.log_std_layer = nn.Linear(net_arch[-1], action_dim)


        self._init_weights(log_std_init)

    def _init_weights(self, log_std_init):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.constant_(layer.bias, 0)

        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.constant_(self.mean_layer.bias, 0)
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        nn.init.constant_(self.log_std_layer.bias, log_std_init)

    def forward(self, obs, deterministic=False):
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        std = log_std.exp()

        if deterministic:
            return torch.tanh(mean)


        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)


        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, std
    
    def sample(self, obs):
        return self.forward(obs, deterministic=False)
    
    def act(self, obs, deterministic=False):

        with torch.no_grad():
            if deterministic:
                x = self.net(obs)
                mean = self.mean_layer(x)
                return torch.tanh(mean)
            else:
                actions, _, _ = self.forward(obs, deterministic=False)
                return actions
    
    def act_inference(self, obs):

        return self.act(obs, deterministic=True)
    
    @torch.jit.export
    def act_jit(self, obs):

        x = self.net(obs)
        mean = self.mean_layer(x)
        return torch.tanh(mean)

    def reset_noise(self, batch_size=1):
        pass  


class SAC_Actor_JIT(nn.Module):

    def __init__(self, sac_actor):
        super().__init__()

        self.net = sac_actor.net
        self.mean_layer = sac_actor.mean_layer
    
    def forward(self, observations):

        x = self.net(observations)
        mean = self.mean_layer(x)
        return torch.tanh(mean)
    
    def act_inference(self, observations):

        return self.forward(observations)


class SAC_Critic(nn.Module):
    def __init__(
        self, 
        obs_dim, 
        action_dim,
        critic_hidden_dims=None,
        activation="relu", 
        n_critics=2,
        **kwargs
    ):
        super(SAC_Critic, self).__init__()
        

        if kwargs:
            unexpected_keys = [key for key in kwargs.keys()]
            if unexpected_keys:
                print(f"SAC_Critic.__init__ got unexpected arguments, which will be ignored: {unexpected_keys}")

        layers_dims = critic_hidden_dims if critic_hidden_dims is not None else [256, 256]
        
        self.activation_fn = get_activation(activation)
        self.n_critics = n_critics
        

        self.critics = nn.ModuleList(
            [self.build_critic(obs_dim, action_dim, layers_dims) for _ in range(n_critics)]
        )
        
        self._init_weights()

    def build_critic(self, obs_dim, action_dim, net_arch):
        layers = []
        input_dim = obs_dim + action_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation_fn)
            input_dim = hidden_dim
        layers.append(nn.Linear(net_arch[-1], 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for critic in self.critics:
            for layer in critic:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                    nn.init.constant_(layer.bias, 0)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return tuple(critic(x) for critic in self.critics)
    

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None