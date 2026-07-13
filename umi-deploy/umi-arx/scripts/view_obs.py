import os
import numpy as np
import matplotlib.pyplot as plt

# 请根据你的实际情况，替换为刚才跑出来的实验文件夹下具体的 npy 文件路径
# 随便挑一个中间步骤的，比如 12.npy 或 24.npy
FILE_PATH = "/home/yd/program/nyx/arx-difussion-deploy/umi-deploy/umi-arx/data/experiments/20260413_182136/obs/0/12.npy"

def view_obs(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return
        
    print(f"📂 正在读取: {file_path}")
    # 你的代码中使用 allow_pickle=True 保存了包含多个对象的 dict
    data = np.load(file_path, allow_pickle=True).item()
    
    print("\n🔍 === 数据结构概览 ===")
    print("顶层 Keys:", data.keys())
    
    # 提取网络真正看到的输入
    obs_dict_np = data.get('obs_dict_np', {})
    print("\n🧠 Policy 实际接收的 Obs 字典 Keys:", obs_dict_np.keys())
    
    # 寻找图像数据（UMI 中通常带有 'rgb' 或 'camera'）
    img_keys = [k for k in obs_dict_np.keys() if 'rgb' in k.lower() or 'image' in k.lower()]
    
    if not img_keys:
        print("⚠️ 在 obs_dict_np 中没有找到图像！退而求其次，查看环境返回的原始图像...")
        raw_obs = data.get('obs', {})
        img_keys = [k for k in raw_obs.keys() if 'rgb' in k.lower()]
        source = raw_obs
    else:
        source = obs_dict_np
        
    if not img_keys:
         print("❌ 彻底没找到任何图像数据！请检查摄像头是否正常工作。")
         return
         
    # 开始绘图
    fig, axes = plt.subplots(1, len(img_keys), figsize=(6 * len(img_keys), 6))
    if len(img_keys) == 1:
        axes = [axes]
        
    for ax, key in zip(axes, img_keys):
        img_array = source[key]
        print(f"\n🖼️ 图像 [{key}] 提取成功!")
        print(f"   -> 原始 Shape: {img_array.shape}, 数据类型: {img_array.dtype}")
        print(f"   -> 数值范围: Min={img_array.min():.3f}, Max={img_array.max():.3f}")
        
        # 兼容处理各种维度的图像数据
        img_to_show = img_array
        
        # 1. 如果带有时间维度/Horizon (例如 shape 为 [2, 3, 224, 224] 包含了过去2帧)
        if len(img_to_show.shape) == 4:
            img_to_show = img_to_show[-1] # 取最新的一帧
            print(f"   -> 存在时间维度，已提取最新一帧: {img_to_show.shape}")
            
        # 2. 如果通道在最前面 (例如 PyTorch 格式 [3, 224, 224])
        if len(img_to_show.shape) == 3 and img_to_show.shape[0] in [1, 3]:
            img_to_show = np.transpose(img_to_show, (1, 2, 0)) # 转成 matplotlib 认识的 [224, 224, 3]
            print(f"   -> 侦测到 Channel First，已转换为: {img_to_show.shape}")
            
        # 绘制图像
        ax.imshow(img_to_show)
        ax.set_title(f"Key: {key}\nShape: {img_to_show.shape}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    view_obs(FILE_PATH)