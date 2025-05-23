# # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: BSD-3-Clause
# # 
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# # list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# # this list of conditions and the following disclaimer in the documentation
# # and/or other materials provided with the distribution.
# #
# # 3. Neither the name of the copyright holder nor the names of its
# # contributors may be used to endorse or promote products derived from
# # this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #
# # Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

import matplotlib.pyplot as plt
import math  # ✅ 解决 NameError



class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]


        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.mean_reward = 0.0
        self.mean_episode_length = 0.0

        self.success_rate = 0.0  
        self.current_statistics = {}  

        self.finished_episodes_info_list = []

        _, _ = self.env.reset()
    


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)


        self.finished_episodes_info_list = []


        ep_infos_for_log = []
        rewbuffer = deque(maxlen=100)  
        lenbuffer = deque(maxlen=100)  
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(
                        self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)  

                    if self.log_dir is not None:
                        # Book keeping for logging and statistics
                        if 'episode' in infos:
                            
                            if isinstance(infos['episode'], dict):
                                 
                                 self.finished_episodes_info_list.append(infos['episode'].copy()) 
                                 
                                 ep_infos_for_log.append(infos['episode'])
                            else:
                                 print(f"⚠️ 警告: infos['episode'] 不是字典: {infos['episode']}")

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)  

            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            
            current_locs = locals()

            current_locs['ep_infos'] = ep_infos_for_log



            if len(rewbuffer) > 0:
                self.mean_reward = statistics.mean(rewbuffer)
                self.mean_episode_length = statistics.mean(lenbuffer)


            temp_success_rate = []
            if self.finished_episodes_info_list: 
                 for ep_info in self.finished_episodes_info_list:
                     if isinstance(ep_info, dict): 
                         if 'success_rate' in ep_info:
                              try: temp_success_rate.append(float(ep_info['success_rate']))
                              except (TypeError, ValueError): pass
                         elif 'success' in ep_info:
                              try: temp_success_rate.append(float(ep_info['success']))
                              except (TypeError, ValueError): pass
            if temp_success_rate:
                self.success_rate = statistics.mean(temp_success_rate)
            # else: self.success_rate = 0.0 # or keep old value


            self.current_statistics = {
                'Mean/reward': self.mean_reward,
                'Mean/episode_length': self.mean_episode_length,
                'success_rate': self.success_rate,  
                'Loss/value_function': mean_value_loss,
                'Loss/surrogate': mean_surrogate_loss,

            }

            if self.log_dir is not None:

                self.log(current_locs)  


            if it % self.save_interval == 0:
                save_path = os.path.join(self.log_dir, f'model_{it}.pt')
                self.save(save_path)


            ep_infos_for_log.clear()


        self.current_learning_iteration += num_learning_iterations


    def log(self, locs, width=80, pad=35):



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

        def safe_mean(value):
            """ 确保 `value` 是 float 或 list，否则转换 """
            if isinstance(value, torch.Tensor):
                return float(value.item())  
            elif isinstance(value, deque):
                return statistics.mean(list(value)) if len(value) > 0 else 0.0  
            elif isinstance(value, list) and len(value) > 0:
                return statistics.mean(value)  
            elif isinstance(value, (int, float)):
                return float(value)  
            return 0.0  

        if len(locs["rewbuffer"]) > 0:
            self.reward_history["mean_reward"].append(safe_mean(locs["rewbuffer"]))
            self.reward_history["rew_action_rate"].append(safe_mean(locs["ep_infos"][0].get("rew_action_rate", 0)))
            self.reward_history["rew_ang_vel_xy"].append(safe_mean(locs["ep_infos"][0].get("rew_ang_vel_xy", 0)))
            self.reward_history["rew_collision"].append(safe_mean(locs["ep_infos"][0].get("rew_collision", 0)))
            self.reward_history["rew_dof_acc"].append(safe_mean(locs["ep_infos"][0].get("rew_dof_acc", 0)))
            self.reward_history["rew_feet_air_time"].append(safe_mean(locs["ep_infos"][0].get("rew_feet_air_time", 0)))
            self.reward_history["rew_lin_vel_z"].append(safe_mean(locs["ep_infos"][0].get("rew_lin_vel_z", 0)))
            self.reward_history["rew_torques"].append(safe_mean(locs["ep_infos"][0].get("rew_torques", 0)))
            self.reward_history["rew_tracking_ang_vel"].append(
                safe_mean(locs["ep_infos"][0].get("rew_tracking_ang_vel", 0)))
            self.reward_history["rew_tracking_lin_vel"].append(
                safe_mean(locs["ep_infos"][0].get("rew_tracking_lin_vel", 0)))

        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

        # if locs["it"] % 1499 == 0 or locs["it"] == locs["max_iterations"] - 1:
        if locs["it"] % 1499 == 0:
            num_rewards = len(self.reward_history)  
            num_cols = 3  
            num_rows = 3  
            rewards_per_figure = num_cols * num_rows  

            reward_items = list(self.reward_history.items())  
            num_figures = math.ceil(num_rewards / rewards_per_figure)  

            for fig_idx in range(num_figures):  
                fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))
                axes = axes.flatten()  

                start_idx = fig_idx * rewards_per_figure  
                end_idx = min(start_idx + rewards_per_figure, num_rewards)  
                for i, (key, values) in enumerate(reward_items[start_idx:end_idx]):
                    axes[i].plot(values, label=key, color="b")
                    axes[i].set_title(key)
                    axes[i].set_xlabel("Iteration")
                    axes[i].set_ylabel("Reward Value")
                    axes[i].grid(True)
                    axes[i].legend()

                plt.tight_layout()
                save_dir = os.path.join(self.log_dir, "reward_plots")
                os.makedirs(save_dir, exist_ok=True)
                fig_path = os.path.join(save_dir, f"rewards_iter_{locs['it']:06d}_fig{fig_idx}.png")
                plt.savefig(fig_path)
                print(f"Reward plot saved to: {fig_path}")
                plt.close(fig)  
    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

