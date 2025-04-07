# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)


# import wandb
#
# def train(args):
#     # 初始化环境和算法
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
#
#     # 初始化 wandb
#     wandb.init(project="anymal_c_rough", config=train_cfg)
#
#     # 监控模型
#     wandb.watch(ppo_runner.alg.actor_critic, log="all")
#
#     # 开始训练
#     ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

#
# if __name__ == '__main__':
#     args = get_args()
#     train(args)

# import isaacgym
# import wandb
# from legged_gym.envs import *
# from legged_gym.utils import get_args, task_registry
# import torch
# import statistics
#
# def train(args):
#     # 初始化环境和训练算法
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
#
#     # ✅ 初始化 `wandb`
#     wandb.init(project="legged_gym", name=args.task, config=train_cfg)
#
#     # ✅ 监控神经网络
#     wandb.watch(ppo_runner.alg.actor_critic, log="all")
#
#     # ✅ 运行训练循环
#     for i in range(train_cfg.runner.max_iterations):
#         ppo_runner.learn(1)  # 运行一次训练
#
#         # ✅ 直接访问 `rewbuffer`
#         rewbuffer = getattr(ppo_runner, "rewbuffer", [])
#         lenbuffer = getattr(ppo_runner, "lenbuffer", [])
#
#         mean_reward = statistics.mean(rewbuffer) if len(rewbuffer) > 0 else 0
#         mean_episode_length = statistics.mean(lenbuffer) if len(lenbuffer) > 0 else 0
#
#         # ✅ 记录 `wandb`
#         wandb.log({
#             "iteration": i,
#             "mean_reward": mean_reward,
#             "mean_episode_length": mean_episode_length,
#             "policy_loss": getattr(ppo_runner, "mean_surrogate_loss", 0),
#             "value_loss": getattr(ppo_runner, "mean_value_loss", 0),
#             "fps": getattr(ppo_runner, "fps", 0),
#             "collection_time": getattr(ppo_runner, "collection_time", 0),
#             "learning_time": getattr(ppo_runner, "learn_time", 0),
#         })
#
#         print(f"Iteration {i}: Reward={mean_reward}")
#
# if __name__ == '__main__':
#     args = get_args()
#     train(args)
#
#
#
# print("ppo_runner attributes:", dir(ppo_runner))

# import numpy as np
# import os
# from datetime import datetime
#
# import isaacgym
# from legged_gym.envs import *
# from legged_gym.utils import get_args, task_registry
# import torch
#
# def train(args):
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
#     log_data = ppo_runner.log(locals())
#
#     print("Log Data Keys:", log_data.keys())  # 查看 `log_data` 里有什么
#     # ✅ 打印 `ppo_runner` 里的所有属性，看看 `rewbuffer` 是否存在
#     print("ppo_runner attributes:", dir(ppo_runner))
#     print("ppo_runner.env attributes:", dir(ppo_runner.env))
#
#     ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
#
# if __name__ == '__main__':
#     args = get_args()
#     train(args)

# import isaacgym
# import wandb
# from legged_gym.envs import *
# from legged_gym.utils import get_args, task_registry
# import torch
# import statistics
#
# def train(args):
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
#
#     wandb.init(project="legged_gym", name=args.task, config=train_cfg)
#     wandb.watch(ppo_runner.alg.actor_critic, log="all")
#
#     for i in range(train_cfg.runner.max_iterations):
#         ppo_runner.learn(1)  # 运行 1 次训练
#
#         # ✅ 直接从 `ppo_runner.env.rew_buf` 获取 `reward`
#         rewbuffer = getattr(ppo_runner.env, "rew_buf", []).tolist()  # 确保转换成 Python 列表
#         mean_reward = statistics.mean(rewbuffer) if rewbuffer else 0
#
#         wandb.log({
#             "iteration": i,
#             "mean_reward": mean_reward,
#             "policy_loss": getattr(ppo_runner, "mean_surrogate_loss", 0),
#             "value_loss": getattr(ppo_runner, "mean_value_loss", 0),
#             "fps": getattr(ppo_runner, "fps", 0),
#             "collection_time": getattr(ppo_runner, "collection_time", 0),
#             "learning_time": getattr(ppo_runner, "learn_time", 0),
#         })
#
#         print(f"Iteration {i}: Reward={mean_reward}")
#
# if __name__ == '__main__':
#     args = get_args()
#     train(args)

# import isaacgym
# import wandb
# from legged_gym.envs import *
# from legged_gym.utils import get_args, task_registry
# import torch
# import statistics
#
# def train(args):
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
#
#     wandb.init(project="legged_gym", name=args.task, config=train_cfg)
#     wandb.watch(ppo_runner.alg.actor_critic, log="all")
#
#     for i in range(train_cfg.runner.max_iterations):
#         ppo_runner.learn(1)  # 运行 1 次训练
#
#         # ✅ 打印 `extras` 内容
#         print("ppo_runner.env attributes:", dir(ppo_runner.env))
#
#         # ✅ 获取 reward
#         rewbuffer = getattr(ppo_runner.env, "rew_buf", []).tolist()
#         mean_reward = statistics.mean(rewbuffer) if rewbuffer else 0
#
#         # ✅ 计算 success_rate（基于 `time_outs`）
#         success_rate = 0
#         if 'time_outs' in ppo_runner.env.extras:
#             time_outs = ppo_runner.env.extras['time_outs']
#             success_rate = torch.sum(time_outs).item() / ppo_runner.env.num_envs
#
#         wandb.log({
#             "iteration": i,
#             "mean_reward": mean_reward,
#             "success_rate": success_rate,  # ✅ 记录成功率
#         })
#
#         print(f"Iteration {i}: Reward={mean_reward}, Success Rate={success_rate}")
#
# if __name__ == '__main__':
#     args = get_args()
#     train(args)

# import numpy as np
# import os
# from datetime import datetime
# import isaacgym
# import wandb
# import torch
# import statistics
# import matplotlib.pyplot as plt
# from legged_gym.envs import *
# from legged_gym.utils import get_args, task_registry
#
# def train(args):
#     # ✅ 初始化环境
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
#
#     # ✅ 读取 `max_iterations` 作为训练次数
#     num_iterations = train_cfg.runner.max_iterations
#     print(f"Training for {num_iterations} iterations...")
#
#     # ✅ 初始化 wandb 进行可视化
#     wandb.init(project="legged_gym", name=args.task, config=train_cfg)
#     wandb.watch(ppo_runner.alg.actor_critic, log="all")
#
#     step_heights = []
#     success_rates = []
#
#     # ✅ 进行完整训练
#     ppo_runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=True)
#
#     # ✅ 训练完成后获取数据
#     for i in range(num_iterations):
#         # ✅ 获取 `terrain_level`
#         step_height = torch.mean(ppo_runner.env.extras["episode"]["terrain_level"]).item() if "terrain_level" in ppo_runner.env.extras["episode"] else 0
#
#         # ✅ 计算 `success_rate`
#         success_rate = torch.sum(ppo_runner.env.time_out_buf).item() / ppo_runner.env.num_envs if hasattr(ppo_runner.env, "time_out_buf") else 0
#
#         # ✅ 记录数据
#         step_heights.append(step_height)
#         success_rates.append(success_rate)
#
#         wandb.log({
#             "iteration": i,
#             "step_height": step_height,
#             "success_rate": success_rate,
#         })
#
#         print(f"Iteration {i}/{num_iterations}: Step Height={step_height}, Success Rate={success_rate}")
#
#     # ✅ 训练完成后绘制成功率曲线
#     plot_success_rate(step_heights, success_rates)
#
# def plot_success_rate(step_heights, success_rates):
#     step_heights, success_rates = np.array(step_heights), np.array(success_rates)
#
#     # ✅ 计算 `step height` 下的平均成功率
#     unique_steps = np.unique(step_heights)
#     avg_success_rate_step = [np.mean(success_rates[step_heights == s]) for s in unique_steps]
#
#     plt.figure(figsize=(6, 4))
#     plt.plot(unique_steps, avg_success_rate_step, '-o', label="Success Rate")
#     plt.xlabel("Step height [m]")
#     plt.ylabel("Success rate [%]")
#     plt.legend()
#     plt.title("Success Rate vs. Step Height")
#     plt.savefig("success_rate_plot.png")
#     plt.show()
#
# if __name__ == '__main__':
#     args = get_args()
#     train(args)

# import numpy as np
# import os
# from datetime import datetime
# import isaacgym
# import wandb
# import torch
# import statistics
# import matplotlib.pyplot as plt
# from legged_gym.envs import *
# from legged_gym.utils import get_args, task_registry
#
# def train(args):
#     # ✅ 初始化环境
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
#
#     # ✅ 读取 `max_iterations` 作为训练次数
#     num_iterations = train_cfg.runner.max_iterations
#     print(f"Training for {num_iterations} iterations...")
#
#     # ✅ 初始化 wandb 进行可视化
#     wandb.init(project="legged_gym", name=args.task, config=train_cfg)
#     wandb.watch(ppo_runner.alg.actor_critic, log="all")
#
#     step_heights = []
#     success_rates = []
#     rewards = []
#
#
#     for i in range(num_iterations):
#         ppo_runner.learn(1)  # ✅ 只训练 1 次 iteration，然后打印信息
#
#         # ✅ 获取 `terrain_level`
#         step_height = torch.mean(ppo_runner.env.extras["episode"]["terrain_level"]).item() if "terrain_level" in ppo_runner.env.extras["episode"] else 0
#
#         # ✅ 计算 `success_rate`
#         success_rate = torch.sum(ppo_runner.env.time_out_buf).item() / ppo_runner.env.num_envs if hasattr(ppo_runner.env, "time_out_buf") else 0
#
#         # ✅ 获取 `mean_reward`
#         rewbuffer = getattr(ppo_runner.env, "rew_buf", []).tolist()
#         mean_reward = statistics.mean(rewbuffer) if rewbuffer else 0
#
#         # ✅ 记录数据
#         step_heights.append(step_height)
#         success_rates.append(success_rate)
#         rewards.append(mean_reward)
#
#         # ✅ 记录到 wandb
#         wandb.log({
#             "iteration": i,
#             "step_height": step_height,
#             "success_rate": success_rate,
#             "mean_reward": mean_reward,
#         })
#
#         # ✅ 在终端输出结果
#         print(f"Iteration {i}/{num_iterations}: Step Height={step_height:.2f}, Success Rate={success_rate:.2f}, Reward={mean_reward:.4f}")
#
#         print(f"time_out_buf: {ppo_runner.env.time_out_buf}")
#         print(f"Number of successful episodes: {torch.sum(ppo_runner.env.time_out_buf).item()}")
#     # ✅ 训练完成后绘制成功率曲线
#     plot_success_rate(step_heights, success_rates)
#
# def plot_success_rate(step_heights, success_rates):
#     step_heights, success_rates = np.array(step_heights), np.array(success_rates)
#
#     # ✅ 计算 `step height` 下的平均成功率
#     unique_steps = np.unique(step_heights)
#     avg_success_rate_step = [np.mean(success_rates[step_heights == s]) for s in unique_steps]
#
#     plt.figure(figsize=(6, 4))
#     plt.plot(unique_steps, avg_success_rate_step, '-o', label="Success Rate")
#     plt.xlabel("Step height [m]")
#     plt.ylabel("Success rate [%]")
#     plt.legend()
#     plt.title("Success Rate vs. Step Height")
#     plt.savefig("success_rate_plot.png")
#     plt.show()
#
# if __name__ == '__main__':
#     args = get_args()
#     train(args)


        # import numpy as np
        # import os
        # import matplotlib.pyplot as plt
        # from datetime import datetime
        #
        # import isaacgym
        # from legged_gym.envs import *
        # from legged_gym.utils import get_args, task_registry
        # import torch
        #
        #
        # def train(args):
        #     # 创建环境
        #     env, env_cfg = task_registry.make_env(name=args.task, args=args)
        #     # 创建 PPO 训练器
        #     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
        #
        #     # 初始化数据存储
        #     reward_list = []  # 记录奖励
        #     iteration_list = []  # 记录迭代次数
        #
        #     # 训练过程
        #     max_iterations = train_cfg.runner.max_iterations
        #     for iteration in range(max_iterations):
        #         # 运行训练，获取当前迭代的平均奖励
        #         ppo_runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
        #
        #         # 获取最近的 100 个 episode 的平均 reward
        #         if hasattr(ppo_runner, 'rewbuffer') and len(ppo_runner.rewbuffer) > 0:
        #             avg_reward = np.mean(ppo_runner.rewbuffer)
        #         else:
        #             avg_reward = 0
        #
        #         # 记录数据
        #         reward_list.append(avg_reward)
        #         iteration_list.append(iteration)
        #         # 每 1000 轮打印一次信息
        #         if iteration % 20 == 0:
        #             print(f"Iteration {iteration}: Average Reward = {avg_reward}")
        #
        #         # 训练结束后绘制 Reward 变化曲线
        #         plt.figure(figsize=(10, 5))
        #         plt.plot(iteration_list, reward_list, label="Average Reward", color='b')
        #         plt.xlabel("Iterations")
        #         plt.ylabel("Average Reward")
        #         plt.title(f"Training Reward Curve ({args.task})")
        #         plt.legend()
        #         plt.grid()
        #         plt.savefig("reward_curve.png")  # 保存图像
        #         plt.show()
        #
        #         if __name__ == '__main__':
        #             args = get_args()
        #         train(args)





