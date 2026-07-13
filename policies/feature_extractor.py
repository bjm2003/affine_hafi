"""
轻量双流特征提取器 —— 空间感知 + 自身状态
=================================================
v4: 22 维结构化观测 (替代 36 LiDAR + 10 Global)。

结构:
  1. 空间流:  (n_spatial_dirs=16,) → Linear(64) → ReLU → spatial_feat (64)
  2. 自身流:  (n_self=6,)          → Linear(64) → ReLU → self_feat (64)
  3. 融合:    [spatial_feat, self_feat] → Linear(128) → ReLU → output (feature_dim=128)

空间感知 16 维: 16 方向最近障碍归一化距离 (从 36 条 LiDAR 扇区 min-pool)
自身状态  6 维: [gx, gy, vx, vy, scale, prev_scale]
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from config import Config


class FormationFeatureExtractor(BaseFeaturesExtractor):
    """
    轻量双流 MLP 特征提取器，用于 SB3 PPO。

    输入观测: (batch, n_spatial_dirs + n_self_features) = (batch, 22)
    输出特征: (batch, feature_dim) = (batch, 128)
    """

    def __init__(self, observation_space: spaces.Box, cfg: Config = None):
        cfg = cfg or Config()
        super().__init__(observation_space, features_dim=cfg.feature_dim)

        self.n_spatial = cfg.n_spatial_dirs    # 16
        self.n_self = cfg.n_self_features      # 6

        self.spatial_encoder = nn.Sequential(
            nn.Linear(self.n_spatial, 64),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            nn.Linear(self.n_self, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, cfg.feature_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        spatial = obs[:, :self.n_spatial]       # (batch, 16)
        self_feat = obs[:, self.n_spatial:]     # (batch, 6)

        spatial_feat = self.spatial_encoder(spatial)    # (batch, 64)
        self_feat = self.self_encoder(self_feat)        # (batch, 64)

        combined = torch.cat([spatial_feat, self_feat], dim=-1)  # (batch, 128)
        return self.fusion(combined)  # (batch, feature_dim=128)
