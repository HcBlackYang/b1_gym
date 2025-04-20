# # import torch
# # import numpy as np

# # class ReplayBuffer:
# #     def __init__(
# #         self, 
# #         buffer_size, 
# #         obs_shape, 
# #         action_shape, 
# #         device="cpu", 
# #         n_envs=1, 
# #         optimize_memory_usage=False, 
# #         handle_timeout_termination=True
# #     ):
# #         """
# #         经验回放缓冲区 (Replay Buffer)，用于存储强化学习经验 (s, a, r, s', done)
        
# #         :param buffer_size: 缓冲区最大容量
# #         :param obs_shape: 观察空间形状
# #         :param action_shape: 动作空间形状
# #         :param device: 存储设备 (cpu 或 gpu)
# #         :param n_envs: 并行环境数
# #         :param optimize_memory_usage: 是否优化内存使用（节省约 50% 的存储）
# #         :param handle_timeout_termination: 处理 Gym 中的 `TimeLimit.truncated` 终止条件
# #         """
# #         self.buffer_size = buffer_size
# #         self.device = torch.device(device)
# #         self.n_envs = n_envs
# #         self.optimize_memory_usage = optimize_memory_usage
# #         self.handle_timeout_termination = handle_timeout_termination
# #         self.ptr = 0  # 记录当前存储位置
# #         self.full = False  # 缓冲区是否已满

# #         # 经验存储
# #         self.observations = torch.zeros((buffer_size, n_envs, *obs_shape), dtype=torch.float32, device=self.device)
# #         self.actions = torch.zeros((buffer_size, n_envs, *action_shape), dtype=torch.float32, device=self.device)
# #         self.rewards = torch.zeros((buffer_size, n_envs, 1), dtype=torch.float32, device=self.device)
# #         self.dones = torch.zeros((buffer_size, n_envs, 1), dtype=torch.float32, device=self.device)
# #         self.timeouts = torch.zeros((buffer_size, n_envs, 1), dtype=torch.float32, device=self.device)

# #         if not optimize_memory_usage:
# #             self.next_observations = torch.zeros((buffer_size, n_envs, *obs_shape), dtype=torch.float32, device=self.device)
    
# #     def __len__(self):
# #             """
# #             返回 ReplayBuffer 中存储的样本数量
# #             """
# #             return self.buffer_size if self.full else self.ptr

# #     def add_transition(self, obs, action, reward, next_obs, done, info=None):
# #         """
# #         添加一个 (s, a, r, s', done) 经验到缓冲区
# #         :param obs: 当前状态 (s)
# #         :param action: 采取的动作 (a)
# #         :param reward: 观察到的奖励 (r)
# #         :param next_obs: 下一个状态 (s')
# #         :param done: 终止标志 (done)
# #         :param info: 额外环境信息（用于处理超时终止）
# #         """
# #         idx = self.ptr  # 获取当前存储索引
# #         reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
# #         if reward.ndimension() == 1:
# #             reward = reward.unsqueeze(-1)  # 将 reward 转换为形状 [n_envs, 1]

# #         self.observations[idx] = torch.tensor(obs, dtype=torch.float32, device=self.device)
# #         self.actions[idx] = torch.tensor(action, dtype=torch.float32, device=self.device)
# #         self.rewards[idx] = reward

# #         done = torch.tensor(done, dtype=torch.float32, device=self.device)
# #         if done.ndimension() == 1:
# #             done = done.unsqueeze(-1)
# #         self.dones[idx] = done

# #         if self.handle_timeout_termination and info is not None:
# #             self.timeouts[idx] = torch.tensor([i.get("TimeLimit.truncated", False) for i in info], dtype=torch.float32, device=self.device)

# #         if self.optimize_memory_usage:
# #             # 内存优化模式下，`observations` 存储了 `next_observations`
# #             next_idx = (idx + 1) % self.buffer_size
# #             self.observations[next_idx] = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
# #         else:
# #             self.next_observations[idx] = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

# #         # 更新指针和缓冲区状态
# #         self.ptr = (self.ptr + 1) % self.buffer_size
# #         if self.ptr == 0:
# #             self.full = True

# #     def sample(self, batch_size):
# #         """
# #         从 Replay Buffer 随机采样 batch_size 个样本
# #         :param batch_size: 采样的 batch 大小
# #         :return: (obs, actions, rewards, next_obs, dones)
# #         """
# #         max_size = self.buffer_size if self.full else self.ptr
# #         batch_indices = np.random.randint(0, batch_size, max_size)

# #         if self.optimize_memory_usage:
# #             next_obs = self.observations[(batch_indices + 1) % self.buffer_size]
# #         else:
# #             next_obs = self.next_observations[batch_indices]

# #         # 处理 done 和超时终止的情况
# #         dones = self.dones[batch_indices] * (1 - self.timeouts[batch_indices])

# #         return (
# #             self.observations[batch_indices].to(self.device),
# #             self.actions[batch_indices].to(self.device),
# #             self.rewards[batch_indices].to(self.device),
# #             next_obs.to(self.device),
# #             dones.to(self.device),
# #         )

# #     def is_ready(self, batch_size):
# #         """
# #         检查是否有足够的数据进行采样
# #         """
# #         return (self.buffer_size if self.full else self.ptr) >= batch_size
# import torch
# import numpy as np

# class ReplayBuffer:
#     def __init__(self, buffer_size, obs_shape, action_shape, device="cpu", optimize_memory_usage=False):
#         self.buffer_size = buffer_size
#         self.device = torch.device(device)
#         self.optimize_memory_usage = optimize_memory_usage

#         self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32)
#         self.actions = torch.zeros((buffer_size, *action_shape), dtype=torch.float32)
#         self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32)
#         self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32)

#         if not optimize_memory_usage:
#             self.next_observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32)

#         self.ptr = 0
#         self.full = False

#     def __len__(self):
#         return self.buffer_size if self.full else self.ptr

#     def add_transition(self, obs, action, reward, next_obs, done):
#         n = obs.shape[0]  # n_envs
#         for i in range(n):
#             idx = self.ptr
#             self.observations[idx] = obs[i].clone().detach().to(torch.float32)
#             self.actions[idx] = action[i].clone().detach().to(torch.float32)
#             self.rewards[idx] = reward[i].clone().detach().unsqueeze(0).to(torch.float32)
#             self.dones[idx] = done[i].clone().detach().unsqueeze(0).to(torch.float32)
#             if self.optimize_memory_usage:
#                 next_idx = (idx + 1) % self.buffer_size
#                 self.observations[next_idx] = next_obs[i].clone().detach().to(torch.float32)
#             else:
#                 self.next_observations[idx] = next_obs[i].clone().detach().to(torch.float32)

#             self.ptr = (self.ptr + 1) % self.buffer_size
#             if self.ptr == 0:
#                 self.full = True

#     def sample(self, batch_size):
#         max_size = len(self)
#         indices = np.random.randint(0, max_size, size=batch_size)

#         obs = self.observations[indices].to(self.device)
#         actions = self.actions[indices].to(self.device)
#         rewards = self.rewards[indices].to(self.device)
#         dones = self.dones[indices].to(self.device)

#         if self.optimize_memory_usage:
#             next_obs = self.observations[(indices + 1) % self.buffer_size].to(self.device)
#         else:
#             next_obs = self.next_observations[indices].to(self.device)

#         return obs, actions, rewards, next_obs, dones

#     def is_ready(self, batch_size):
#         return len(self) >= batch_size

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
