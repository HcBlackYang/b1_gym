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

class OffPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Initialize Actor and Critic Networks for SAC
        self.actor = SAC_Actor(self.env.num_obs, self.env.num_actions, **self.policy_cfg).to(self.device)
        self.critic1 = SAC_Critic(self.env.num_obs, self.env.num_actions, **self.policy_cfg).to(self.device)
        self.critic2 = SAC_Critic(self.env.num_obs, self.env.num_actions, **self.policy_cfg).to(self.device)
        self.target_critic1 = SAC_Critic(self.env.num_obs, self.env.num_actions, **self.policy_cfg).to(self.device)
        self.target_critic2 = SAC_Critic(self.env.num_obs, self.env.num_actions, **self.policy_cfg).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # SAC Algorithm
        self.alg = SAC(obs_dim=self.env.num_obs, 
                       action_dim=self.env.num_actions, 
                       n_envs=self.env.num_envs,
                       device=self.device, 
                       **self.alg_cfg)

        # Initialize experience replay buffer
        buffer_size = self.cfg.get("replay_buffer_size", 100)
        obs_shape = (env.num_obs,)
        action_shape = (env.num_actions,)
        n_envs = env.num_envs

        self.replay_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            n_envs=n_envs,
            )   


        # TensorBoard Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Initialize environment
        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # Initialize TensorBoard writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # Optionally initialize episode length buffer
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # Initialize variables
        obs = self.env.get_observations().to(self.device)
        actions = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.device)
        rewards = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        dones = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        infos = []

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

            # Rollout: Interact with environment and store in replay buffer
            for _ in range(self.cfg["num_steps_per_env"]):
                with torch.no_grad():
                    actions = self.actor.sample(obs)[0]  # Sample action from actor
                    next_obs, _, rewards, dones, infos = self.env.step(actions)
                    next_obs = torch.tensor(next_obs, device=self.device, dtype=torch.float32)

                    # Store in replay buffer
                    self.replay_buffer.add_transition(obs, actions, rewards, next_obs, dones)

                    # Update current observations
                    obs = next_obs

            collection_time = time.time() - start

            # Perform SAC updates
            if "batch_size" not in self.cfg:
                self.cfg["batch_size"] = 256
            if len(self.replay_buffer) > self.cfg["batch_size"]:
                # Sample a batch from replay buffer
                # batch = self.replay_buffer.sample(self.cfg["batch_size"])

                # Update SAC agent (actor, critic networks, target critic)
                self.alg.update(self.cfg["batch_size"])

            # Log step
            if self.log_dir is not None:
                self.log(locals(), collection_time)

            # Save model at interval
            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))

            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))

    def log(self, locs, collection_time, width=80, pad=35):
        self.tot_timesteps += self.cfg["num_steps_per_env"] * self.env.num_envs
        self.tot_time += collection_time + locs.get('learn_time', 0)
        iteration_time = collection_time + locs.get('learn_time', 0)
        obs = self.env.get_observations().to(self.device)

        ep_string = ""
        if locs.get('ep_infos'):
            for key in locs['ep_infos'][0]:
                infotensor = torch.cat([ep_info[key].unsqueeze(0) if isinstance(ep_info[key], torch.Tensor) and ep_info[key].dim() == 0 else torch.tensor([ep_info[key]]) for ep_info in locs['ep_infos']], dim=0).to(self.device)
                value = torch.mean(infotensor)
                self.writer.add_scalar(f'Episode/{key}', value, locs['it'])
                ep_string += f"""{'Mean episode ' + key:>{pad}} {value:.4f}\n"""

        _, _, std = self.actor(obs)
        mean_std = std.mean()
        fps = int(self.cfg["num_steps_per_env"] * self.env.num_envs / iteration_time)

        self.writer.add_scalar('Loss/value_function', locs.get('mean_value_loss', 0), locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs.get('mean_surrogate_loss', 0), locs['it'])

        # self.writer.add_scalar('Loss/learning_rate', learning_rate, locs['it'])
        self.writer.add_scalar('actor_lr', self.alg.actor_optimizer.param_groups[0]['lr'], locs['it'])
        self.writer.add_scalar('critic_lr', self.alg.critic_optimizer.param_groups[0]['lr'], locs['it'])
        self.writer.add_scalar('alpha_lr', self.alg.alpha_optimizer.param_groups[0]['lr'], locs['it'])

        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', collection_time, locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs.get('learn_time', 0), locs['it'])

        # if len(locs['rewbuffer']) > 0:
        #     self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
        #     self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
        if len(locs['rewbuffer']) > 0:
            mean_rew = statistics.mean(locs['rewbuffer'])
            mean_len = statistics.mean(locs['lenbuffer'])
            self.writer.add_scalar('Train/mean_reward', mean_rew, locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', mean_len, locs['it'])
        else:
            mean_rew = 0.0
            mean_len = 0.0


        # Log info
        log_string = (f"""{'#' * width}\n"""
                      f"""{"Learning iteration " + str(locs['it']) + "/" + str(self.current_learning_iteration + locs['num_learning_iterations']).center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {locs.get('learn_time', 0):.3f}s)\n"""
                      f"""{'Value function loss:':>{pad}} {locs.get('mean_value_loss', 0):.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {locs.get('mean_surrogate_loss', 0):.4f}\n"""
                      f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                      f"""{'Mean reward:':>{pad}} {mean_rew:.2f}\n"""
                      f"""{'Mean episode length:':>{pad}} {mean_len:.2f}\n"""
                      f"""{'-' * width}\n"""
                      f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                      f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                      f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                      f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

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
        self.actor.eval()
        if device is not None:
            self.actor.to(device)
        return self.actor.act_inference
