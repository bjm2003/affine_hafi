"""
全局配置文件 —— 所有超参数集中管理

v2: 连续动作空间 + 激光雷达观测 (替代离散关键点)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class Config:
    # ===================== 环境 =====================
    world_size: float = 12.0         # 世界尺寸 [-6, 6] x [-6, 6]
    enable_world_clip: bool = True   # 是否将车辆位置裁剪在世界边界内
    world_clip_margin: float = 0.1   # 边界裁剪内缩量 (m), 越小越接近墙边
    n_vehicles: int = 3               # 车辆数量
    d_form: float = 0.8               # 编队边长 (等边三角形, 车辆两两距离=0.8m, 质心到车≈0.462m)
    dt_mpc: float = 0.1               # MPC 控制周期 (s), 训练用 10Hz (加速采样)
    dt_rl: float = 0.5                # RL 决策周期 (s), = mpc_steps_per_rl * dt_mpc
    mpc_steps_per_rl: int = 5        # 每个 RL step 内 MPC 执行次数 (10×0.1s=1.0s)
    max_episode_steps: int = 150      # 最大 episode 步数 (RL steps), 与 20260419_194200 对齐

    # ===================== 子目标 =====================
    # 非对称纵向代价下, 约束放宽为 d_ss > 0 ⟺ α > v_max/(R0+v_max)
    # v_max=0.3, R0=0.5 → 阈值=0.375, α=0.50 > 0.375 ✅ (d_ss=0.20m)
    R0: float = 0.4                  # 子目标偏移半径 (与 20260419_194200 对齐)

    # ===================== 激光雷达 =====================
    n_lidar_rays: int = 72            # 扫描射线数 (每10°一条, 覆盖360°)
    lidar_max_dist: float = 1.5       # 最大探测距离 (m)
    lidar_yaw_offset: float = 0.0      # LiDAR frame 相对车体 +x 的静态 yaw 偏移 (rad)
    # RL 观测雷达来源: "leader"(车0传感器) 或 "center"(编队融合中心)
    rl_lidar_origin: str = "leader"

    # ===================== 感知驱动 MPC =====================
    # MPC 障碍输入来源: "sensor"=传感器可见障碍, "global"=环境真值(仅仿真对照)
    mpc_obstacle_source: str = "sensor"
    # LIDAR 命中阈值比例: 仅当距离 < max_dist * ratio 视为检测到障碍
    # (0.995→0.95: 留出 5% 远端噪声缓冲, 真实障碍最远 ≈2.85m 仍可见)
    perception_hit_ratio: float = 0.95
    # 单个障碍最少连续命中射线数
    perception_min_cluster_rays: int = 2
    # 同一 cluster 内相邻射线允许的最大径向跳变 (m):
    # 用于在仅按角度索引相邻聚类的基础上, 把不同距离的物体拆开,
    # 避免"近物 + 远墙"被并成一个幻影障碍 (落点会落到自由空间).
    perception_max_range_jump: float = 0.20
    # 从射线簇估计障碍半径的上下界
    perception_obs_radius_min: float = 0.10
    perception_obs_radius_max: float = 0.35
    # 障碍半径保守膨胀量 (覆盖检测噪声和未建模动态)
    perception_radius_inflation: float = 0.05
    # 线性度过滤: 区分墙壁/围栏 (线性) 与真障碍物 (紧凑)
    # PCA 特征值比 λ1/λ0 > 此阈值 → 视为墙壁并跳过 (不聚类为圆)
    perception_linearity_threshold: float = 5.0
    # 线性度检测至少需要的簇内射线数 (太少的簇 PCA 不可靠)
    perception_min_linearity_points: int = 5
    # 多帧速度估计开关 (消融实验用: False=所有障碍零速度输入 MPC)
    perception_velocity_estimation: bool = False
    # 感知更新周期 (s): 完整 LIDAR→检测→跟踪管线执行频率 (~5Hz)
    # 中间 MPC 步用速度线性外推, 减少约 50% 感知计算量
    # 训练 (dt_mpc=0.1): 每 2 MPC 步扫描一次; 评估 (dt_mpc=0.05): 每 4 步
    perception_dt: float = 0.2
    # perception_dt: float = dt_mpc

    # ===================== 车间通信 (V2V) =====================
    # MPC 其它车辆位置来源: "broadcast"=车间广播接收, "global"=环境真值(仿真对照)
    v2v_other_position_source: str = "broadcast"
    # 每条 V2V 消息独立丢包概率
    v2v_dropout_prob: float = 0.0
    # 位置测量高斯噪声标准差 (m)
    v2v_position_noise_std: float = 0.0
    # 通信延迟 (以 mpc step 为单位, 1 step = dt_mpc 秒)
    v2v_delay_steps: int = 0
    # 丢包后允许沿用的最大历史步数
    v2v_max_hold_steps: int = 5

    # ===================== 缩放 =====================
    s_min: float = 0.50                # 最小缩放因子 (0.65→0.50: 允许更小编队通过窄缝)
    s_max: float = 1.5                # 最大缩放因子
    max_delta_s: float = 0.5       # 每 RL 步最大缩放变化量 (与 20260419_194200 对齐)

    # ===================== 参考轨迹平滑 =====================
    # 非对称代价下约束: α > v_max/(R0+v_max) = 0.375 即可保证 d_ss > 0
    # α=0.50: d_ss = R0 − (1−α)/α × v_max = 0.5 − 1.0×0.3 = 0.20m (更平滑, 编队误差更小)
    ref_ema_alpha: float = 1.0       # 参考点EMA平滑系数 (越大subgoal越靠近raw)
    max_delta_ref: float = 0.15       # 每 RL 步参考点最大变化量 (与 wandb 20260414 对齐)

    # ===================== MPC =====================
    mpc_horizon: int = 10             # MPC 预测步长 H (与 20260419_194200 对齐)
    v_max: float = 0.1                # 速度上限 (m/s), 每RL步最大位移0.30m
    vehicle_radius: float = 0.18      # 车辆外接圆半径 (m), 20×30cm 车的外接圆
    d_safe: float = 0.15              # 额外安全余量 (m), 与 wandb 20260414 对齐
    # --- MPC + 非对称纵向代价 + 滑动参考 ---
    # 非对称: 纵向仅罚落后 (lag²), 超前不罚; 横向双向惩罚 (e_lat²)
    # p_ref(k) = p_ref + k·dt·slide·path_dir (滑动参考, 双保险)
    Q_p: float = 10.0                 # 位置跟踪权重 (各向同性, 匹配旧 MPC 验证值)
    R_u: float = 1.0                  # 控制输入权重 (与 wandb 20260414 对齐)
    R_du: float = 0.3                 # 控制平滑权重 (恢复旧 MPC 验证值)
    R_vs: float = 0.0                 # 沿路径速度正则; 0=关闭 (停顿主要由滑动参考+Q_p 处理)
    v_des_ratio: float = 1.0          # 期望沿路径速度 = v_des_ratio * v_max
    slide_ratio: float = 0.0           # 滑动参考速率比: 0=静态 (非对称代价已消除减速, 暂关闭观察效果)
    asymmetric_tracking: bool = True   # 非对称纵向代价: 仅惩罚落后, 超前不罚 → 消除 horizon 内减速
    Q_formation: float = 2.0          # 编队误差权重 (恢复旧 MPC 验证值, 允许灵活避障)
    slack_weight: float = 1e5         # slack 变量罚权 (保证可行性)
    n_obs_max: int = 3                # MPC 参数化最大障碍数 (单车 3m LIDAR 视野内通常 ≤4)
    ipopt_max_iter: int = 50          # IPOPT 最大迭代次数 (配合 acceptable_tol 提前终止, 50 足够)
    # 矩形场地边界 (map 系): MPC 直接用半平面约束保证轨迹不越界
    # None = 不启用; (xmin, xmax, ymin, ymax) = 启用
    arena_bounds: Optional[Tuple[float, float, float, float]] = None
    # 场地边界 LiDAR 过滤余量 (m): 端点落在边界 ± margin 内的射线视为围栏
    arena_lidar_margin: float = 0.15

    # ===================== 障碍 =====================
    n_static_obs_range: Tuple[int, int] = (2, 5)    # 静态障碍数量范围
    n_dynamic_obs_range: Tuple[int, int] = (0, 2)   # 动态障碍数量范围
    # 训练/评估场景中的静态障碍半径范围（动态与注入式保持原范围）
    static_obs_radius_range: Tuple[float, float] = (0.10, 0.30)
    # 动态/注入式障碍半径范围
    obs_radius_range: Tuple[float, float] = (0.10, 0.30)
    dynamic_obs_speed: float = 0.20   # 动态障碍速度上界 (m/s), 下界 0.1

    # ===================== RL 奖励 =====================
    w_progress: float = 5.0           # 前进奖励权重
    w_goal: float = 100.0             # 到达目标奖励权重 (50→100: 强化成功信号)
    w_scale: float = 1.5              # 缩放偏离惩罚权重 (非 corridor 场景)
    w_scale_corridor: float = 8.0     # corridor 专用缩放权重 (远强于 w_progress, 迫使缩小编队)
    w_delta_s: float = 0.3            # 缩放变化率惩罚 (|Δs|, 鼓励平滑过渡)
    w_collision: float = 50.0         # 碰撞惩罚权重 (30→50: 确保碰撞代价 > 绕路累计代价)
    w_proximity: float = 0.0          # 连续避障惩罚权重 (2.5→1.0: 轻柔预警, 避免过度绕路)
    proximity_threshold_factor: float = 2.5  # proximity 阈值: d_safe * factor; 例 0.10×2.5=0.25m
    w_formation: float = 2.0          # 编队形变惩罚权重 (惩罚三角形被挤压变形)
    # w_mpc_fail: float = 1.0           # MPC不可行惩罚权重 (已禁用)
    w_time: float = 0.01              # 每步时间惩罚 (0.25→0.01: 消除时间压力, 允许安全绕路)
    # w_truncation: float = 15.0        # 超时截断惩罚权重 (已禁用, 按剩余距离比例, 最大=-15)
    goal_tolerance: float = 0.1       # 到达目标判定距离 (m)
    delta_clip: float = 0.25          # 前进奖励裁剪值
    d_max: float = 5.0                # (保留兼容, lidar 不再使用此参数)

    # ===================== PPO 训练 =====================
    total_timesteps: int = 3_000_000  # 总训练步数
    lr: float = 3e-4                  # 学习率
    gamma: float = 0.99               # 折扣因子
    gae_lambda: float = 0.95          # GAE lambda
    clip_range: float = 0.2           # PPO clip range
    batch_size: int = 2048            # batch size
    n_epochs: int = 4                 # PPO 更新 epochs 
    n_steps: int = 8192               # 每次 rollout 采样步数
    ent_coef: float = 0.01            # 熵正则系数

    # ===================== 训练随机化 =====================
    start_distance_range: Tuple[float, float] = (6.0, 10.0)

    # 训练场景混合采样概率 [open, corridor, s_corridor, z_corridor, u_trap, dynamic]
    train_scenario_probs: Tuple[float, float, float, float, float, float] = (0.25, 0.15, 0.15, 0.15, 0.15, 0.15)

    # ===================== 课程学习 =====================
    # 第1阶段（早期）：着重简单场景（开放场景）和基础走廊
    curriculum_phase1_probs: Tuple[float, float, float, float, float, float] = (0.50, 0.10, 0.05, 0.05, 0.15, 0.15)
    # 第2阶段（后期）：均衡复杂走廊和动态障碍
    curriculum_phase2_probs: Tuple[float, float, float, float, float, float] = (0.15, 0.15, 0.20, 0.20, 0.15, 0.15)
    curriculum_switch_step: int = 500_000

    # Corridor gap 宽度课程
    # 正确通过条件: gap_half > 0.4×scale + vehicle_radius + d_safe = 0.4s + 0.28
    # target=0.70 → s ≤ 1.05, 加 -0.06 扰动后 gap=0.64 → s ≤ 0.90 (需 10% 缩放)
    corridor_gap_start: float = 1.0
    corridor_gap_target: float = 0.70
    corridor_gap_warmup_frac: float = 0.05
    corridor_gap_anneal_frac: float = 0.40

    # ===================== 失败重采样 =====================
    failure_replay_max_prob: float = 0.3
    failure_sr_low: float = 0.5
    failure_sr_high: float = 0.8
    failure_ema_alpha: float = 0.05
    failure_buffer_size: int = 200
    failure_decay_interval: int = 50
    failure_max_age: int = 300
    failure_max_replays: int = 15

    # ===================== 观测空间 (22 维) =====================
    # 空间感知 (16 维): 16 方向最近障碍归一化距离 (22.5° 间隔, 360° 覆盖)
    #   每个方向: 从 36 条 LiDAR 射线中取对应扇区 min, ∈ [0, 1]
    n_spatial_dirs: int = 16          # RL 空间感知方向数
    # 消融: 仍输出 22 维, 但将前 n_spatial_dirs 维置为 1.0 (RL 无有效障距; MPC 仍用传感器) — 对应 --no_spatial_obs
    no_spatial_obs: bool = False
    # 自身状态 (6 维): [gx, gy, vx, vy, scale, prev_scale]
    #   gx, gy: 目标相对位置 (归一化笛卡尔向量)
    #   vx, vy: 编队中心速度 (归一化)
    #   scale, prev_scale: 编队缩放因子
    n_self_features: int = 6

    # ===================== 特征提取器 =====================
    feature_dim: int = 128            # 最终特征维度 (256→128, 匹配缩小的观测)

    # ===================== 路径 =====================
    log_dir: str = "logs"
    model_dir: str = "models"
    video_dir: str = "videos"

    # ===================== 评估 =====================
    eval_episodes: int = 50
    video_record_freq: int = 10

    @property
    def obs_dim(self) -> int:
        """观测空间维度: n_spatial_dirs + n_self_features = 16 + 6 = 22

        空间感知 16 维: 16 方向最近障碍归一化距离
        自身状态  6 维: [gx, gy, vx, vy, scale, prev_scale]
        """
        return self.n_spatial_dirs + self.n_self_features
