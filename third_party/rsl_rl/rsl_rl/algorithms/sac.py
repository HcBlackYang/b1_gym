
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
        self.device = device
        
        # 从kwargs中提取相关参数，避免重复
        actor_hidden_dims = kwargs.get("actor_hidden_dims", kwargs.get("hidden_dims", [256, 256]))
        critic_hidden_dims = kwargs.get("critic_hidden_dims", kwargs.get("hidden_dims", [256, 256]))
        activation = kwargs.get("activation", "relu")
        log_std_init = kwargs.get("log_std_init", -3)
        
        # 创建Actor网络，只传递必要的参数
        self.actor = SAC_Actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            log_std_init=log_std_init
        ).to(device)
        
        # 创建Critic网络，只传递必要的参数
        self.critic = SAC_Critic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            n_critics=2  # SAC需要两个critic网络
        ).to(device)

        # 创建回放缓冲区
        self.replay_buffer = ReplayBuffer(
            buffer_size=kwargs.get("buffer_size", 100000), 
            obs_shape=(obs_dim,), 
            action_shape=(action_dim,), 
            device=self.device
        )

        self.gamma = gamma
        self.tau = tau
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device, requires_grad=True)
        
        # 初始化目标Q网络
        self.target_q1 = copy.deepcopy(self.critic.critics[0])
        self.target_q2 = copy.deepcopy(self.critic.critics[1] if len(self.critic.critics) > 1 else self.critic.critics[0])
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=lr)

        # 目标熵
        self.target_entropy = -float(action_dim)

    def soft_update(self, target, source):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update_alpha(self, log_probs):
        """更新温度系数alpha"""
        alpha_loss = -(self.alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return self.alpha.item()

    def update(self, batch_size, obs, actions, rewards, next_obs, dones):
        """执行一次SAC更新"""
        # 确保所有数据都在正确的设备上
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        
        # 获取当前Q值
        q1, q2 = self.critic(obs, actions)

        # 计算目标Q值
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor(next_obs)
            target_q1_next = self.target_q1(torch.cat([next_obs, next_actions], dim=-1))
            target_q2_next = self.target_q2(torch.cat([next_obs, next_actions], dim=-1))
            target_q_next = torch.min(target_q1_next, target_q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q_next

        # 计算critic损失
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # 更新critic网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算actor损失
        actions_pred, log_probs, _ = self.actor(obs)
        q1_pred, q2_pred = self.critic(obs, actions_pred)
        q_pred = torch.min(q1_pred, q2_pred)
        actor_loss = (self.alpha * log_probs - q_pred).mean()

        # 更新actor网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新温度系数
        alpha_val = self.update_alpha(log_probs)

        # 软更新目标网络
        self.soft_update(self.target_q1, self.critic.critics[0])
        if len(self.critic.critics) > 1:
            self.soft_update(self.target_q2, self.critic.critics[1])
        else:
            self.soft_update(self.target_q2, self.critic.critics[0])

        return actor_loss.item(), critic_loss.item(), alpha_val

    def train(self, num_epochs, batch_size, save_interval=1000):
        """训练SAC算法"""
        total_steps = 0
        for epoch in range(num_epochs):
            actor_loss_total, critic_loss_total = 0, 0
            for step in range(1000):  # 每个周期训练1000步
                if self.replay_buffer.is_ready(batch_size):
                    # 从回放缓冲区采样
                    batch = self.replay_buffer.sample(batch_size)
                    actor_loss, critic_loss, alpha = self.update(batch_size, *batch)
                    actor_loss_total += actor_loss
                    critic_loss_total += critic_loss
                    total_steps += 1

                if total_steps % save_interval == 0:
                    print(f"Epoch {epoch}, Step {total_steps}: actor_loss = {actor_loss_total / max(1, total_steps):.4f}, "
                          f"critic_loss = {critic_loss_total / max(1, total_steps):.4f}, alpha = {alpha:.4f}")
                    actor_loss_total, critic_loss_total = 0, 0

            # 每隔10个周期打印alpha值
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Alpha = {self.alpha.item():.4f}")