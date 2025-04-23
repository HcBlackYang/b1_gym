
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class B1RobotCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]  
        default_joint_angles = {
            'FL_hip_joint': 0.2,  # [rad]
            'RL_hip_joint': 0.2,  # [rad]
            'FR_hip_joint': -0.2,  # [rad]
            'RR_hip_joint': -0.2,  # [rad]

            'FL_thigh_joint': 0.6,  # [rad]
            'RL_thigh_joint': 1.0,  # [rad]
            'FR_thigh_joint': 0.6,  # [rad]
            'RR_thigh_joint': 1.0,  # [rad]

            'FL_calf_joint': -1.3,  # [rad]
            'RL_calf_joint': -1.3,  # [rad]
            'FR_calf_joint': -1.3,  # [rad]
            'RR_calf_joint': -1.3  # [rad]
        }

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.5, 1.5] # min max [m/s]
            lin_vel_y = [-1.5, 1.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class control(LeggedRobotCfg.control):
        control_type = 'P'  
        stiffness = {'joint': 100.}  
        damping = {'joint': 2.5}  
        action_scale = 0.25  
        decimation = 4  
        use_actuator_network = False  
        # actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/B1_actuator_network.pt"

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 235
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/b1_description/xacro/b1.urdf"
        name = "b1"
        foot_name = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]  
        # foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        # terminate_after_contacts_on = ["base", "imu_link"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

    class sensors:
        contact_sensors = ["FR_foot_contact", "FL_foot_contact", "RR_foot_contact", "RL_foot_contact"]

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.5  
        max_contact_force = 500.
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 6.0
            tracking_ang_vel = 3
            torques = -0.0001/8
            action_rate = -0.01 
            # dof_pos_limits = -0.3
            dof_pos_limits = -10 
            collision = -1.2  
            lin_vel_z = -3
            feet_air_time = 1  
            dof_acc = -5e-7  
            orientation = -5 
            base_height = -30.  

        soft_dof_pos_limit = 0.9

class B1RobotCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.015  # 增加探索，避免局部最优


    # class policy:
    #     init_noise_std = 0.8
    #     actor_hidden_dims = [32]
    #     critic_hidden_dims = [32]
    #     activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    #     # only for 'ActorCriticRecurrent':
    #     rnn_type = 'lstm'
    #     rnn_hidden_size = 64
    #     rnn_num_layers = 1

    # class runner(LeggedRobotCfgPPO.runner):
    #     policy_class_name = "ActorCriticRecurrent"
    #     experiment_name = 'b1'
    #     load_run = -1

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'b1'
        load_run = -1



