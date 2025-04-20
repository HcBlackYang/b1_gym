
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, obs_shape, action_shape, device="cpu", optimize_memory_usage=False):
        self.buffer_size = buffer_size
        self.device = torch.device(device)  # 使用传入的设备
        self.optimize_memory_usage = optimize_memory_usage

        # 直接在指定设备上创建缓冲区
        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        if not optimize_memory_usage:
            self.next_observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)

        self.ptr = 0
        self.full = False

    def __len__(self):
        return self.buffer_size if self.full else self.ptr

    def add_transition(self, obs, action, reward, next_obs, done):
        """优化后的添加转换函数"""
        n = obs.shape[0]  # n_envs
        
        # 确保输入已经在正确的设备上并转换为正确的类型
        if obs.device != self.device:
            obs = obs.to(self.device)
        if action.device != self.device:
            action = action.to(self.device)
        if reward.device != self.device:
            reward = reward.to(self.device)
        if isinstance(reward, torch.Tensor) and reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        reward = reward.float()  # 确保是浮点类型
        
        if next_obs.device != self.device:
            next_obs = next_obs.to(self.device)
            
        # 处理done张量，确保它是float类型且具有正确的形状
        if done.device != self.device:
            done = done.to(self.device)
        if isinstance(done, torch.Tensor) and done.dim() == 1:
            done = done.unsqueeze(-1)
        done = done.float()  # 确保是浮点类型
        
        # 批量处理，避免循环
        start_idx = self.ptr
        end_idx = (self.ptr + n) % self.buffer_size
        
        if end_idx > start_idx:  # 没有回绕
            indices = range(start_idx, end_idx)
            self.observations[indices] = obs
            self.actions[indices] = action
            self.rewards[indices, 0] = reward.squeeze(-1)
            self.dones[indices, 0] = done.squeeze(-1)
            
            if not self.optimize_memory_usage:
                self.next_observations[indices] = next_obs
            else:
                next_indices = [(idx + 1) % self.buffer_size for idx in indices]
                self.observations[next_indices] = next_obs
                
        else:  # 回绕的情况
            indices1 = range(start_idx, self.buffer_size)
            indices2 = range(0, end_idx)
            
            # 分段赋值
            self.observations[indices1] = obs[:len(indices1)]
            self.observations[indices2] = obs[len(indices1):]
            
            self.actions[indices1] = action[:len(indices1)]
            self.actions[indices2] = action[len(indices1):]
            
            self.rewards[indices1, 0] = reward[:len(indices1)].squeeze(-1)
            self.rewards[indices2, 0] = reward[len(indices1):].squeeze(-1)
            
            self.dones[indices1, 0] = done[:len(indices1)].squeeze(-1)
            self.dones[indices2, 0] = done[len(indices1):].squeeze(-1)
            
            if not self.optimize_memory_usage:
                self.next_observations[indices1] = next_obs[:len(indices1)]
                self.next_observations[indices2] = next_obs[len(indices1):]
            else:
                next_indices1 = [(idx + 1) % self.buffer_size for idx in indices1]
                next_indices2 = [(idx + 1) % self.buffer_size for idx in indices2]
                self.observations[next_indices1] = next_obs[:len(indices1)]
                self.observations[next_indices2] = next_obs[len(indices1):]
        
        # 更新指针
        self.ptr = (self.ptr + n) % self.buffer_size
        if self.ptr < start_idx:  # 发生了回绕
            self.full = True

    def sample(self, batch_size):
        max_size = len(self)
        indices = np.random.randint(0, max_size, size=batch_size)

        # 直接返回已在设备上的数据
        obs = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        if self.optimize_memory_usage:
            next_obs = self.observations[(indices + 1) % self.buffer_size]
        else:
            next_obs = self.next_observations[indices]

        return obs, actions, rewards, next_obs, dones

    def is_ready(self, batch_size):
        return len(self) >= batch_size
