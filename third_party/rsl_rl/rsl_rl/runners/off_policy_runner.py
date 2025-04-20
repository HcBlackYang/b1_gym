

import time
import sys
import os
import statistics
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rsl_rl.algorithms.sac import SAC
from rsl_rl.env import VecEnv
from rsl_rl.modules.actor_critic_sac import SAC_Actor, SAC_Critic
from rsl_rl.buffer.replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt
import math  

class OffPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # 创建Actor和Critic网络
        self.actor = SAC_Actor(
            obs_dim=self.env.num_obs,
            action_dim=self.env.num_actions,
            actor_hidden_dims=self.policy_cfg.get("actor_hidden_dims", [256, 256]),
            activation=self.policy_cfg.get("activation", "relu")
        ).to(self.device)
        
        self.critic1 = SAC_Critic(
            obs_dim=self.env.num_obs,
            action_dim=self.env.num_actions,
            critic_hidden_dims=self.policy_cfg.get("critic_hidden_dims", [256, 256]),
            activation=self.policy_cfg.get("activation", "relu"),
            n_critics=2
        ).to(self.device)
        
        self.critic2 = SAC_Critic(
            obs_dim=self.env.num_obs,
            action_dim=self.env.num_actions,
            critic_hidden_dims=self.policy_cfg.get("critic_hidden_dims", [256, 256]),
            activation=self.policy_cfg.get("activation", "relu"),
            n_critics=2
        ).to(self.device)
        
        self.target_critic1 = SAC_Critic(
            obs_dim=self.env.num_obs,
            action_dim=self.env.num_actions,
            critic_hidden_dims=self.policy_cfg.get("critic_hidden_dims", [256, 256]),
            activation=self.policy_cfg.get("activation", "relu"),
            n_critics=2
        ).to(self.device)
        
        self.target_critic2 = SAC_Critic(
            obs_dim=self.env.num_obs,
            action_dim=self.env.num_actions,
            critic_hidden_dims=self.policy_cfg.get("critic_hidden_dims", [256, 256]),
            activation=self.policy_cfg.get("activation", "relu"),
            n_critics=2
        ).to(self.device)
        
        # 复制参数到目标网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # SAC算法
        # 移除可能与SAC构造函数参数冲突的配置项
        alg_cfg_filtered = {k: v for k, v in self.alg_cfg.items() 
                          if k not in ['obs_dim', 'action_dim', 'n_envs', 'device']}
        self.alg = SAC(
            obs_dim=self.env.num_obs, 
            action_dim=self.env.num_actions, 
            n_envs=self.env.num_envs,
            device=self.device, 
            **alg_cfg_filtered
        )

        # 初始化经验回放缓冲区，使用与算法相同的设备
        buffer_size = self.cfg.get("replay_buffer_size", 100000)
        self.replay_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            obs_shape=(self.env.num_obs,),
            action_shape=(self.env.num_actions,),
            device=self.device  # 使用与算法相同的设备
        )

        # TensorBoard日志
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # 初始化环境
        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # 初始化TensorBoard writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # 初始化变量
        obs = self.env.get_observations().to(self.device)
        
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # 回合：与环境交互并存储到回放缓冲区
            for _ in range(self.cfg["num_steps_per_env"]):
                with torch.no_grad():
                    actions = self.actor.sample(obs)[0]  # 从actor采样动作
                    next_obs, _, rewards, dones, infos = self.env.step(actions)
                    
                    # 确保数据在正确的设备上
                    if not isinstance(next_obs, torch.Tensor):
                        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                    elif next_obs.device != self.device:
                        next_obs = next_obs.to(self.device)

                    # 同样处理rewards和dones
                    if not isinstance(rewards, torch.Tensor):
                        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                    elif rewards.device != self.device:
                        rewards = rewards.to(self.device)
                        
                    if not isinstance(dones, torch.Tensor):
                        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
                    else:
                        # 确保dones是float类型
                        dones = dones.float()
                        if dones.device != self.device:
                            dones = dones.to(self.device)
                    
                    # 更新奖励和episode长度统计
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    
                    # 检查是否有episode完成
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    for j in new_ids:
                        rewbuffer.append(cur_reward_sum[j].item())
                        lenbuffer.append(cur_episode_length[j].item())
                        cur_reward_sum[j] = 0
                        cur_episode_length[j] = 0

                    # 存储到回放缓冲区
                    # self.replay_buffer.add_transition(obs, actions, rewards, next_obs, dones)
                    if dones.dtype == torch.bool:
                        dones = dones.float()
                    self.replay_buffer.add_transition(obs, actions, rewards, next_obs, dones)

                    # 更新当前观察
                    obs = next_obs

            collection_time = time.time() - start



            # 执行SAC更新
            batch_size = self.cfg.get("batch_size", 256)
            mean_value_loss = 0
            mean_surrogate_loss = 0
            learn_start = time.time()
            
            if self.replay_buffer.is_ready(batch_size):
                # 从回放缓冲区采样批次数据
                batch = self.replay_buffer.sample(batch_size)
                
                # 进行多次更新
                num_updates = self.cfg.get("num_updates_per_env", 1)
                for _ in range(num_updates):
                    actor_loss, critic_loss, alpha = self.alg.update(batch_size, *batch)
                    mean_value_loss += critic_loss
                    mean_surrogate_loss += actor_loss
                
                mean_value_loss /= num_updates
                mean_surrogate_loss /= num_updates
            
            learn_time = time.time() - learn_start

            # 记录学习步骤
            if self.log_dir is not None:
                locals_with_losses = locals()
                locals_with_losses['mean_value_loss'] = mean_value_loss
                locals_with_losses['mean_surrogate_loss'] = mean_surrogate_loss
                locals_with_losses['learn_time'] = learn_time
                locals_with_losses['ep_infos'] = ep_infos  # 确保传递episode信息
                self.log(locals_with_losses, collection_time)

            # 在间隔保存模型
            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))

            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))




    # def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    #     # 初始化TensorBoard writer
    #     if self.log_dir is not None and self.writer is None:
    #         self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    #     if init_at_random_ep_len:
    #         self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

    #     # 初始化变量
    #     obs = self.env.get_observations().to(self.device)
        
    #     self.actor.train()
    #     self.critic1.train()
    #     self.critic2.train()

    #     ep_infos = []  # 这个很重要，要在每个iteration开始时清空
    #     rewbuffer = deque(maxlen=100)
    #     lenbuffer = deque(maxlen=100)
    #     cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    #     cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    #     tot_iter = self.current_learning_iteration + num_learning_iterations
    #     for it in range(self.current_learning_iteration, tot_iter):
    #         start = time.time()
            
    #         # 清空episode信息
    #         ep_infos.clear()

    #         # 回合：与环境交互并存储到回放缓冲区
    #         for _ in range(self.cfg["num_steps_per_env"]):
    #             with torch.no_grad():
    #                 actions = self.actor.sample(obs)[0]  # 从actor采样动作
    #                 next_obs, _, rewards, dones, infos = self.env.step(actions)
                    
    #                 # 确保数据在正确的设备上
    #                 if not isinstance(next_obs, torch.Tensor):
    #                     next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
    #                 elif next_obs.device != self.device:
    #                     next_obs = next_obs.to(self.device)

    #                 # 同样处理rewards和dones
    #                 if not isinstance(rewards, torch.Tensor):
    #                     rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
    #                 elif rewards.device != self.device:
    #                     rewards = rewards.to(self.device)
                        
    #                 if not isinstance(dones, torch.Tensor):
    #                     dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
    #                 else:
    #                     # 确保dones是float类型
    #                     if dones.dtype == torch.bool:
    #                         dones = dones.float()
    #                     if dones.device != self.device:
    #                         dones = dones.to(self.device)
                    
    #                 # 更新奖励和episode长度统计
    #                 cur_reward_sum += rewards
    #                 cur_episode_length += 1
                    
    #                 # 检查是否有episode完成
    #                 new_ids = (dones > 0).nonzero(as_tuple=False)
    #                 for j in new_ids:
    #                     env_id = j.item()
    #                     # 收集episode信息
    #                     if "episode" in infos and env_id in infos["episode"]:
    #                         ep_infos.append(infos["episode"][env_id])
    #                     elif isinstance(infos, dict) and env_id < len(infos):
    #                         # 尝试直接从infos中获取
    #                         if "episode" in infos[env_id]:
    #                             ep_infos.append(infos[env_id]["episode"])
    #                         else:
    #                             # 如果环境没有提供详细信息，手动构建
    #                             episode_info = {
    #                                 "episode": {
    #                                     "r": cur_reward_sum[env_id].item(),
    #                                     "l": cur_episode_length[env_id].item()
    #                                 }
    #                             }
    #                             # 尝试直接从环境获取reward组成部分
    #                             if hasattr(self.env, "reward_scales"):
    #                                 for key in self.env.reward_scales.keys():
    #                                     if hasattr(self.env, key):
    #                                         value = getattr(self.env, key)[env_id]
    #                                         if isinstance(value, torch.Tensor):
    #                                             value = value.item()
    #                                         episode_info["episode"][key] = value
    #                             ep_infos.append(episode_info["episode"])
                        
    #                     rewbuffer.append(cur_reward_sum[j].item())
    #                     lenbuffer.append(cur_episode_length[j].item())
    #                     cur_reward_sum[j] = 0
    #                     cur_episode_length[j] = 0

    #                 # 存储到回放缓冲区
    #                 self.replay_buffer.add_transition(obs, actions, rewards, next_obs, dones)

    #                 # 更新当前观察
    #                 obs = next_obs

    #         collection_time = time.time() - start

    #         # 执行SAC更新
    #         batch_size = self.cfg.get("batch_size", 256)
    #         mean_value_loss = 0
    #         mean_surrogate_loss = 0
    #         learn_start = time.time()
            
    #         if self.replay_buffer.is_ready(batch_size):
    #             # 从回放缓冲区采样批次数据
    #             batch = self.replay_buffer.sample(batch_size)
                
    #             # 进行多次更新
    #             num_updates = self.cfg.get("num_updates_per_env", 1)
    #             for _ in range(num_updates):
    #                 actor_loss, critic_loss, alpha = self.alg.update(batch_size, *batch)
    #                 mean_value_loss += critic_loss
    #                 mean_surrogate_loss += actor_loss
                
    #             mean_value_loss /= num_updates
    #             mean_surrogate_loss /= num_updates
            
    #         learn_time = time.time() - learn_start

    #         # 记录学习步骤
    #         if self.log_dir is not None:
    #             locals_with_losses = locals()
    #             locals_with_losses['mean_value_loss'] = mean_value_loss
    #             locals_with_losses['mean_surrogate_loss'] = mean_surrogate_loss
    #             locals_with_losses['learn_time'] = learn_time  # 添加学习时间
    #             locals_with_losses['ep_infos'] = ep_infos  # 确保ep_infos被传递
    #             self.log(locals_with_losses, collection_time)

    #         # 在间隔保存模型
    #         if it % self.cfg["save_interval"] == 0:
    #             self.save(os.path.join(self.log_dir, f'model_{it}.pt'))

    #     self.current_learning_iteration += num_learning_iterations
    #     self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))




    def log(self, locs, collection_time, width=80, pad=35):
        """修复后的SAC日志记录方法 - 调试版本"""
        
        # 初始化 reward 历史记录（如果没有）
        if not hasattr(self, "reward_history"):
            self.reward_history = {
                "mean_reward": [],
                "rew_action_rate": [],
                "rew_ang_vel_xy": [],
                "rew_collision": [],
                "rew_dof_acc": [],
                "rew_feet_air_time": [],
                "rew_lin_vel_z": [],
                "rew_torques": [],
                "rew_tracking_ang_vel": [],
                "rew_tracking_lin_vel": []
            }
            print("Initialized reward_history with keys:", list(self.reward_history.keys()))

        def safe_mean(value):
            """确保 `value` 是 float 或 list，否则转换"""
            if isinstance(value, torch.Tensor):
                return float(value.item()) if value.numel() == 1 else float(value.mean().item())
            elif isinstance(value, deque):
                return statistics.mean(list(value)) if len(value) > 0 else 0.0
            elif isinstance(value, list) and len(value) > 0:
                return statistics.mean(value)
            elif isinstance(value, (int, float)):
                return float(value)
            return 0.0

        # 原有的时间统计
        self.tot_timesteps += self.cfg["num_steps_per_env"] * self.env.num_envs
        self.tot_time += collection_time + locs.get('learn_time', 0)
        iteration_time = collection_time + locs.get('learn_time', 0)
        obs = self.env.get_observations().to(self.device)

        # 处理episode信息并记录到TensorBoard
        ep_string = ""
        if locs.get('ep_infos'):
            print(f"Found ep_infos with {len(locs['ep_infos'])} entries")
            
            # 收集每个reward组成部分的数据
            reward_components = {
                "rew_action_rate": [],
                "rew_ang_vel_xy": [],
                "rew_collision": [],
                "rew_dof_acc": [],
                "rew_feet_air_time": [],
                "rew_lin_vel_z": [],
                "rew_torques": [],
                "rew_tracking_ang_vel": [],
                "rew_tracking_lin_vel": []
            }
            
            for ep_info in locs['ep_infos']:
                if isinstance(ep_info, dict):
                    print("Processing ep_info keys:", list(ep_info.keys()))
                    for key in self.reward_history.keys():
                        if key != "mean_reward" and key in ep_info:
                            value = safe_mean(ep_info[key])
                            reward_components[key].append(value)
                            print(f"Found {key}: {value}")
            
            # 将收集到的数据添加到历史记录
            for key, values in reward_components.items():
                if values:
                    avg_value = statistics.mean(values)
                    self.reward_history[key].append(avg_value)
                    self.writer.add_scalar(f'Episode/{key}', avg_value, locs['it'])
                    ep_string += f"""{'Mean episode ' + key:>{pad}} {avg_value:.4f}\n"""
                    print(f"Added {key} to history: {avg_value}")
            
            # 如果某些reward组件没有数据，尝试从环境状态获取
            if not any(reward_components.values()):
                print("No reward components found in ep_infos, trying to get from other sources...")
                # 你可能需要在这里根据你的环境结构添加获取reward组件的代码
        
        # 记录总体mean_reward
        if len(locs.get("rewbuffer", [])) > 0:
            mean_value = safe_mean(locs["rewbuffer"])
            self.reward_history["mean_reward"].append(mean_value)
            print(f"Added mean_reward to history: {mean_value}")

        # 获取统计信息
        _, _, std = self.actor(obs)
        mean_std = std.mean()
        fps = int(self.cfg["num_steps_per_env"] * self.env.num_envs / iteration_time)

        # 写入TensorBoard
        self.writer.add_scalar('Loss/value_function', locs.get('mean_value_loss', 0), locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs.get('mean_surrogate_loss', 0), locs['it'])
        self.writer.add_scalar('Learning_rate/actor', self.alg.actor_optimizer.param_groups[0]['lr'], locs['it'])
        self.writer.add_scalar('Learning_rate/critic', self.alg.critic_optimizer.param_groups[0]['lr'], locs['it'])
        self.writer.add_scalar('Learning_rate/alpha', self.alg.alpha_optimizer.param_groups[0]['lr'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', collection_time, locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs.get('learn_time', 0), locs['it'])

        mean_rew = 0.0
        mean_len = 0.0
        if len(locs.get('rewbuffer', [])) > 0:
            mean_rew = statistics.mean(locs['rewbuffer'])
            mean_len = statistics.mean(locs['lenbuffer'])
            self.writer.add_scalar('Train/mean_reward', mean_rew, locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', mean_len, locs['it'])

        # 记录信息
        log_string = (f"""{'#' * width}\n"""
                    f"""{"Learning iteration " + str(locs['it']) + "/" + str(self.current_learning_iteration + locs['num_learning_iterations']).center(width, ' ')}\n\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {locs.get('learn_time', 0):.3f}s)\n"""
                    f"""{'Value function loss:':>{pad}} {locs.get('mean_value_loss', 0):.4f}\n"""
                    f"""{'Surrogate loss:':>{pad}} {locs.get('mean_surrogate_loss', 0):.4f}\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                    f"""{'Mean reward:':>{pad}} {mean_rew:.2f}\n"""
                    f"""{'Mean episode length:':>{pad}} {mean_len:.2f}\n""")
        
        log_string += ep_string  # 添加reward组件信息
        
        log_string += (f"""{'-' * width}\n"""
                    f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                    f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                    f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                    f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

        # 训练过程中的奖励可视化
        if locs["it"] % 149 == 0 or locs["it"] == locs["num_learning_iterations"] - 1:
            num_rewards = len(self.reward_history)
            num_cols = 3
            num_rows = 3
            rewards_per_figure = num_cols * num_rows
            
            reward_items = list(self.reward_history.items())
            num_figures = math.ceil(num_rewards / rewards_per_figure)
            
            print(f"Creating {num_figures} figures for {num_rewards} rewards")
            for key, values in self.reward_history.items():
                print(f"{key}: {len(values)} values")
            
            for fig_idx in range(num_figures):
                fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))
                axes = axes.flatten()
                
                start_idx = fig_idx * rewards_per_figure
                end_idx = min(start_idx + rewards_per_figure, num_rewards)
                
                plot_idx = 0
                for i, (key, values) in enumerate(reward_items[start_idx:end_idx]):
                    if len(values) > 0:  # 只绘制有数据的图表
                        axes[plot_idx].plot(values, label=key, color="b")
                        axes[plot_idx].set_title(key)
                        axes[plot_idx].set_xlabel("Iteration")
                        axes[plot_idx].set_ylabel("Reward Value")
                        axes[plot_idx].grid(True)
                        axes[plot_idx].legend()
                        plot_idx += 1
                        print(f"Plotted {key} with {len(values)} values")
                    else:
                        print(f"Skipping {key} - no data")
                
                # 隐藏未使用的子图
                for i in range(plot_idx, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                save_dir = os.path.join(self.log_dir, "reward_plots")
                os.makedirs(save_dir, exist_ok=True)
                fig_path = os.path.join(save_dir, f"rewards_iter_{locs['it']:06d}_fig{fig_idx}.png")
                plt.savefig(fig_path)
                print(f"✅ Reward plot saved to: {fig_path}")
                plt.close(fig)


    def save(self, path, infos=None):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.alg.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.alg.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alg.alpha_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.critic1.load_state_dict(loaded_dict['critic1_state_dict'])
        self.critic2.load_state_dict(loaded_dict['critic2_state_dict'])
        self.target_critic1.load_state_dict(loaded_dict['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(loaded_dict['target_critic2_state_dict'])

        if load_optimizer:
            self.alg.actor_optimizer.load_state_dict(loaded_dict['actor_optimizer_state_dict'])
            self.alg.critic_optimizer.load_state_dict(loaded_dict['critic_optimizer_state_dict'])
            self.alg.alpha_optimizer.load_state_dict(loaded_dict['alpha_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        """
        获取用于推理的策略函数
        
        Args:
            device: 指定设备，如果为None则使用当前设备
            
        Returns:
            inference_policy: 推理策略函数
        """
        self.actor.eval()
        if device is not None:
            self.actor.to(device)
        return self.actor.act_inference