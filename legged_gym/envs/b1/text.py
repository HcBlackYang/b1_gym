# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 读取仿真数据
# try:
#     with open("tracking_data.pkl", "rb") as f:
#         data = pickle.load(f)
#     print("✅ Tracking data loaded!")
# except FileNotFoundError:
#     print("❌ No tracking data found! Run `b1_env.py` first.")
#     exit()
#
# # 转换数据
# time_log = np.array(data["time_log"])
# q_target_log = np.array(data["q_target_log"])
# q_target_log = np.expand_dims(q_target_log, axis=(1, 2))  # 变成 (1000, 1, 1)
# q_actual_log = np.array(data["q_actual_log"])
#
# # 确保数据不是空的
# print("time_log shape:", time_log.shape, "First 10:", time_log[:10])
# print("q_target_log shape:", q_target_log.shape, "First 10:", q_target_log[:10])
# print("q_actual_log shape:", q_actual_log.shape, "First 10:", q_actual_log[:10])
#
# plt.figure(figsize=(10, 5))
#
# # 🚀 只绘制 Joint 0
# joint_idx = 0
# q_actual_selected = q_actual_log[:, 0, joint_idx]  # 选择环境 0 的 Joint 0
# q_target_selected = q_target_log[:, 0, 0]  # 目标轨迹 Joint 0
#
# plt.plot(time_log, q_target_selected, linestyle="--", label="Target Joint 0", color="blue")
# plt.plot(time_log, q_actual_selected, label="Actual Joint 0", color="red")
#
# plt.xlabel("Time (s)")
# plt.ylabel("Joint Angle (rad)")
# plt.legend()
# plt.title("Leg Tracking Performance (Only Joint 0)")
#
# # ✅ 让 Y 轴刻度自动调整
# plt.gca().yaxis.set_major_locator(plt.MaxNLocator(prune='both', nbins=10))
#
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
import pandas as pd
from tabulate import tabulate


# === 1. 读取 B1 机器人配置 ===
class B1RobotCfg:
    class control:
        # stiffness = {'joint': 60.}  # 真实数据
        # damping = {'joint': 1.5}  # 真实数据

        stiffness = {'joint': 80.}  # 真实数据
        damping = {'joint': 1}  # 真实数据

    class env:
        num_actions = 12  # 12 个关节


# 读取 LEGGED_GYM_ROOT_DIR（你需要修改为你的真实路径）
LEGGED_GYM_ROOT_DIR = "/home/blake/legged_gym"

# 解析 B1 URDF 的实际路径
urdf_path = os.path.join(LEGGED_GYM_ROOT_DIR, "resources/robots/b1_description/xacro/b1.urdf")


# # === 解析 `b1.urdf` 获取转动惯量 ===
# def get_b1_inertia(urdf_path):
#     if not os.path.exists(urdf_path):
#         raise FileNotFoundError(f"❌  URDF 文件未找到: {urdf_path}")
#
#     tree = ET.parse(urdf_path)
#     root = tree.getroot()
#     inertia_values = []
#     for inertia in root.findall(".//inertial/inertia"):
#         izz = float(inertia.get("izz", 1.0))  # 获取 `izz` 作为惯性
#         inertia_values.append(izz)
#
#     if not inertia_values:
#         raise ValueError("❌ 没有找到 inertia 数据，请检查 URDF 文件格式！")
#
#     return np.mean(inertia_values)  # 计算所有关节惯性的平均值


def get_b1_inertia(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    inertia_dict = {}

    for link in root.findall(".//link"):
        link_name = link.get("name")
        inertia = link.find(".//inertial/inertia")

        if inertia is not None:
            izz = float(inertia.get("izz", 1.0))  # 获取 `izz`
            inertia_dict[link_name] = izz  # 存储每个关节的惯性
        else:
            inertia_dict[link_name] = 1.0  # 默认值，防止出错

    return inertia_dict  # 返回一个字典，每个关节一个惯性值


# 解析 URDF 文件
mass = get_b1_inertia(urdf_path)  # 使用真实惯性数据
print(f"✅  B1 机器人惯性 (mass) = {mass}")

# === 3. 设定仿真参数 ===
dt = 0.0001  # 时间步长 (s)
T = 2.0  # 总仿真时间 (s)
num_steps = int(T / dt)
num_joints = B1RobotCfg.env.num_actions  # 12 个关节

# 读取真实的 `stiffness` 和 `damping`
stiffness = B1RobotCfg.control.stiffness['joint']
damping = B1RobotCfg.control.damping['joint']

# === 4. 生成目标正弦轨迹 ===
A = 0.2  # 振幅 (rad)
omega = 2 * np.pi  # 1Hz 角频率
time_log = np.linspace(0, T, num_steps)
q_target_log = A * np.sin(omega * time_log)  # 目标角度



# === 5. 初始化关节状态 ===
q_actual_log = np.zeros((num_steps, num_joints))  # 每个关节的角度
q_velocity = np.zeros(num_joints)  # 角速度初始化为 0


joint_masses = get_b1_inertia(urdf_path)  # 获取每个关节的惯性
joint_names = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
               "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]

# 确保我们得到的惯性按照 joint_names 的顺序排列
mass_values = np.array([joint_masses.get(joint, 1.0) for joint in joint_names])

# 力矩记录
torque_log = np.zeros((num_steps, num_joints))

# === 6. 运行仿真 ===
for step in range(num_steps):
    t = time_log[step]
    q_target = A * np.sin(omega * t)  # 目标轨迹

    for i in range(num_joints):
        # 计算 PD 控制力矩
        torque = stiffness * (q_target - q_actual_log[step - 1, i]) - damping * q_velocity[i]

        # 牛顿欧拉积分计算角速度和角度
        # q_acceleration = torque / mass

        q_acceleration = torque / mass_values[i]  # 让每个关节使用不同惯性

        q_velocity[i] += q_acceleration * dt
        q_actual_log[step, i] = q_actual_log[step - 1, i] + q_velocity[i] * dt

        # 记录力矩
        torque_log[step, i] = torque


# === 计算性能指标 ===
# 1. 计算 RMSE
rmse = np.sqrt(np.mean((q_actual_log - q_target_log[:, np.newaxis]) ** 2, axis=0))

# 2. 计算能量消耗
energy_consumption = np.sum(torque_log * q_velocity * dt, axis=0)

# 3. 计算最大力矩
max_torque = np.max(np.abs(torque_log), axis=0)

# 4. 计算稳定性指标（力矩和角速度的标准差）
torque_std = np.std(torque_log, axis=0)
velocity_std = np.std(q_velocity, axis=0)

# === 组织结果并展示 ===
results_df = pd.DataFrame({
    "Joint": joint_names,
    "RMSE (rad)": rmse,
    "Energy (J)": energy_consumption,
    "Max Torque (Nm)": max_torque,
    "Torque Std": torque_std,
    "Velocity Std": velocity_std
})

# 显示表格
# tools.display_dataframe_to_user(name="B1 Performance Metrics (Improved)", dataframe=results_df)
print(tabulate(results_df, headers="keys", tablefmt="grid"))


# 创建 12 个子图，每个子图显示一个关节的跟踪情况
fig, axes = plt.subplots(4, 3, figsize=(12, 8))  # 4 行 3 列的子图布局
fig.suptitle("B1 Leg Joint Position Tracking with Real Data")

for i, ax in enumerate(axes.flatten()):
    ax.plot(time_log, q_actual_log[:, i], label=f"Actual Joint {i + 1}")
    ax.plot(time_log, q_target_log, '--', label="Target (Sine Wave)", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Angle (rad)")
    ax.set_title(f"Joint {i + 1}")
    ax.legend()
    ax.grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()







