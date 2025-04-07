
from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .b1_config import B1RobotCfg


class B1Env(LeggedRobot):
    cfg: B1RobotCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)


    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs * self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs * self.num_actions, 8, device=self.device,
                                            requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs * self.num_actions, 8, device=self.device,
                                          requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def _compute_torques(self, actions):
        # print(f"Actions: {actions}")  # 打印输入动作
        if self.cfg.control.use_actuator_network:
            with torch.inference_mode():
                self.sea_input[:, 0, 0] = (
                        actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos
                ).flatten()
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(
                    self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
                )
            # print(f"Torques from actuator network: {torques}")  # 打印网络计算出的力矩
            return torques
        else:
            torques = super()._compute_torques(actions)
            # print(f"Torques from PD Controller: {torques}")  # 打印 PD 控制的力矩
            return torques

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt


        # 计算基础奖励：空中时间超过0.2秒开始给奖励
        base_reward = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1)

        # 对超过0.8秒的空中时间添加惩罚
        penalty = torch.sum(torch.where(self.feet_air_time > 0.8,
                                        (0.8 - self.feet_air_time) * first_contact,
                                        torch.zeros_like(self.feet_air_time)),
                            dim=1)


        # 合并奖励和惩罚
        rew_airTime = base_reward + penalty


        # 只有当有移动命令时才给予奖励
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1

        # 重置已接触地面的足部空中时间
        self.feet_air_time *= ~contact_filt

        print("rew_airtime:",rew_airTime)
        print("feet_air_time:",self.feet_air_time)
        print("base_reward:",base_reward)
        print("penalty:",penalty)
        # print("balance_reward:",balance_reward)

        return rew_airTime






