import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys
import os
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rsl_rl.modules.actor_critic_sac import SAC_Actor, SAC_Critic
from rsl_rl.buffer.replay_buffer import ReplayBuffer

class SAC:
    def __init__(self, obs_dim, action_dim, n_envs=1, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4, device='cpu', **kwargs):
        # Initialize actor and critic networks
        self.actor = SAC_Actor(obs_dim, action_dim, hidden_dims=[256, 256], activation="relu").to(device)
        self.critic = SAC_Critic(obs_dim, action_dim, hidden_dims=[256, 256], activation="relu").to(device)

        # Create the replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=100000, obs_shape=(obs_dim,), action_shape=(action_dim,), device='cpu')

        self.gamma = gamma
        self.tau = tau
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float32))  # Temperature coefficient
        self.device = device
        
        # Initialize target Q networks
        # self.target_q1 = SAC_Critic(obs_dim, action_dim, hidden_dims=[256, 256], activation=nn.ReLU).to(device)
        # self.target_q2 = SAC_Critic(obs_dim, action_dim, hidden_dims=[256, 256], activation=nn.ReLU).to(device)
        self.target_q1 = copy.deepcopy(self.critic.critics[0])
        self.target_q2 = copy.deepcopy(self.critic.critics[1])

        # Copy parameters from critic to target networks
        self.target_q1.load_state_dict(self.critic.critics[0].state_dict())
        self.target_q2.load_state_dict(self.critic.critics[1].state_dict())
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=lr)  # Optimizer for the temperature coefficient alpha

        # Target entropy for temperature adjustment (default as -dim of action space)
        self.target_entropy = -np.prod(action_dim).item()

    def soft_update(self, target, source):
        """
        Soft update the target Q networks using the specified tau
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update_alpha(self, log_probs):
        """
        Update the temperature coefficient alpha based on log probability
        """
        alpha_loss = -(self.alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return self.alpha

    # def update(self, batch_size):
    def update(self, batch_size, obs, actions, rewards, next_obs, dones):
        """
        Perform a single update step on both the actor and critic networks
        """
        # Sample a batch of experiences from the replay buffer
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)

        # Get current Q values from the critic networks
        q1, q2 = self.critic(obs, actions)

        # Calculate target Q values using Double Q-learning and target networks
        with torch.no_grad():
            # Get the next actions and log probabilities
            next_actions, next_log_probs, _ = self.actor(next_obs)
            next_obs_action = torch.cat([next_obs, next_actions], dim=-1)
            target_q1, target_q2 = self.target_q1(next_obs_action), self.target_q2(next_obs_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Critic loss (Mean Squared Error)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss (maximize Q value and entropy)
        actions, log_probs, _ = self.actor(obs)
        q1, q2 = self.critic(obs, actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (temperature coefficient)
        self.update_alpha(log_probs)

        # Soft update the target Q networks
        self.soft_update(self.target_q1, self.critic.critics[0])
        self.soft_update(self.target_q2, self.critic.critics[1])

        return actor_loss.item(), critic_loss.item(), self.alpha.item()

    def train(self, num_epochs, batch_size, save_interval=1000):
        """
        Train the SAC algorithm over multiple epochs
        """
        total_steps = 0
        for epoch in range(num_epochs):
            actor_loss_total, critic_loss_total = 0, 0
            for step in range(1000):  # Each epoch trains for 1000 steps
                if self.replay_buffer.is_ready(batch_size):
                    actor_loss, critic_loss, alpha = self.update(batch_size)
                    actor_loss_total += actor_loss
                    critic_loss_total += critic_loss
                    total_steps += 1

                if total_steps % save_interval == 0:
                    print(f"Epoch {epoch}, Step {total_steps}: actor_loss = {actor_loss_total / max(1, total_steps):.4f}, "
                          f"critic_loss = {critic_loss_total / max(1, total_steps):.4f}, alpha = {alpha:.4f}")
                    actor_loss_total, critic_loss_total = 0, 0

            # Print the alpha value at intervals
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Alpha = {alpha:.4f}")


