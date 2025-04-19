
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class B1RobotCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # 降低初始高度，减少摔倒
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
        curriculum = False
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
        control_type = 'P'  # 使用 P 控制
        stiffness = {'joint': 100.}  # 增加刚度，减少腿部震荡
        damping = {'joint': 2.5}  # 增加阻尼，减少摔倒后的乱踢
        action_scale = 0.25  # #数值减小，小步行走，数值变大，大步行走
        decimation = 4  # 降低时间步长，提高控制频率
        use_actuator_network = False  # 适用于 B1
        # actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/B1_actuator_network.pt"

    class env(LeggedRobotCfg.env):
        num_envs = 32768
        num_observations = 235
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/b1_description/xacro/b1.urdf"
        name = "b1"
        foot_name = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]  # 用 calf 代替 foot
        # foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        # terminate_after_contacts_on = ["base", "imu_link"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

    class sensors:
        contact_sensors = ["FR_foot_contact", "FL_foot_contact", "RR_foot_contact", "RL_foot_contact"]

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.5  # 适当降低目标高度，减少摔倒风险
        max_contact_force = 500.
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 6.0
            tracking_ang_vel = 3
            torques = -0.0001/8
            action_rate = -0.01 #数值减小，灵活行走，数值变大，平稳行走
            # dof_pos_limits = -0.3
            dof_pos_limits = -10 #数值变大限制关节自由
            collision = -1.2  # 减少碰撞惩罚，鼓励机器人尝试站起
            lin_vel_z = -3
            feet_air_time = 1  # 适当增加奖励，鼓励合理交替步态
            dof_acc = -5e-7  # 增强关节加速度约束，防止异常腿部运动
            orientation = -5 # 姿态稳定性惩罚
            base_height = -30.  # 底座高度惩罚

        soft_dof_pos_limit = 0.9

class B1RobotCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.015  # 增加探索，避免局部最优

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'b1'
        load_run = -1




