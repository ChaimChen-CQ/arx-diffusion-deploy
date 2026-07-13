import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_experiment(exp_dir):
    zarr_path = os.path.join(exp_dir, "replay_buffer.zarr")
    if not os.path.exists(zarr_path):
        print(f"Error: 找不到 Zarr 数据文件 {zarr_path}")
        return

    print(f"正在加载实验数据: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    try:
        # 读取 Policy 下发的目标动作 (Target)
        actions = root['data/action'][:]
        # 读取底盘真实反馈的末端位姿 (Actual)
        actual_pos = root['data/robot0_eef_pos'][:]
        actual_gripper = root['data/robot0_gripper_width'][:]
        timestamps = root['data/timestamp'][:]
        
        # 将相对时间归零，方便查看
        timestamps = timestamps - timestamps[0]
    except KeyError as e:
        print(f"数据读取失败，缺少键值: {e}。请确保该 Episode 已完整录制。")
        return

    # Action 的前 3 维通常是 X, Y, Z 的目标笛卡尔坐标
    target_pos = actions[:, :3]
    # Action 的第 7 维是夹爪目标宽度
    target_gripper = actions[:, 6]

    # ========== 开始绘图 ==========
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Policy Tracking Analysis\n{os.path.basename(exp_dir)}', fontsize=16)

    # 1. 绘制 X, Y, Z 轴的时序跟踪对比
    axes_labels = ['X Axis', 'Y Axis', 'Z Axis']
    for i in range(3):
        ax = fig.add_subplot(2, 3, i+1)
        ax.plot(timestamps, target_pos[:, i], label='Target (Policy Output)', linestyle='--', color='red')
        ax.plot(timestamps, actual_pos[:, i], label='Actual (Robot Feedback)', alpha=0.7, color='blue')
        ax.set_title(axes_labels[i])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.legend()
        ax.grid(True)

    # 2. 绘制夹爪宽度的时序跟踪对比
    ax_g = fig.add_subplot(2, 3, 4)
    ax_g.plot(timestamps, target_gripper, label='Target Gripper', linestyle='--', color='red')
    ax_g.plot(timestamps, actual_gripper, label='Actual Gripper', alpha=0.7, color='blue')
    ax_g.set_title('Gripper Width')
    ax_g.set_xlabel('Time (s)')
    ax_g.set_ylabel('Width (m)')
    ax_g.legend()
    ax_g.grid(True)

    # 3. 绘制 3D 笛卡尔空间轨迹对比
    ax_3d = fig.add_subplot(2, 3, (5, 6), projection='3d')
    ax_3d.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], label='Target Trajectory', linestyle='--', color='red')
    ax_3d.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], label='Actual Trajectory', alpha=0.7, color='blue')
    
    # 标出起点和终点
    ax_3d.scatter(target_pos[0, 0], target_pos[0, 1], target_pos[0, 2], color='green', s=100, label='Start', marker='o')
    ax_3d.scatter(target_pos[-1, 0], target_pos[-1, 1], target_pos[-1, 2], color='purple', s=100, label='End', marker='x')

    ax_3d.set_title('3D End-Effector Trajectory')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表
    save_path = os.path.join(exp_dir, "tracking_analysis.png")
    plt.savefig(save_path)
    print(f"✅ 图表已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    # 你可以修改为你最新生成数据的文件夹
    EXP_DIR = "/home/yd/program/nyx/arx-difussion-deploy/umi-deploy/umi-arx/data/experiments/20260415_150754"
    analyze_experiment(EXP_DIR)