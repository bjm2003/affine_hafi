"""
双头动作输出策略 —— 共享 backbone + 分离动作头
=================================================
基于 SB3 ActorCriticPolicy，将默认的单线性 action_net 替换为:
  - dxdy_head:  latent → Linear(32) → ReLU → Linear(2)   导航子目标
  - scale_head: latent → Linear(16) → ReLU → Linear(1)   编队缩放

共享 backbone (FormationFeatureExtractor + pi MLP) 不变，
最终输出仍是 3 维连续动作 [dx, dy, scale]，与环境接口兼容。

scale_only_navigation (消融):
  dx, dy 由观测中归一化目标向量 (gx, gy) 与 pure_mpc 基线相同地给出，
  仅 scale 由 scale_head(latent) 输出；高斯 log_std 前两维固定为极小方差，
  避免 PPO 在 dx,dy 上产生错误梯度。
"""

from typing import Tuple

import torch
import torch.nn as nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class DualHeadActionNet(nn.Module):
    """双头动作网络：分别输出导航方向和编队缩放。"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.dxdy_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        dxdy = self.dxdy_head(latent_pi)
        scale = self.scale_head(latent_pi)
        return torch.cat([dxdy, scale], dim=-1)


class ScaleOnlyScaleHead(nn.Module):
    """仅输出 scale 均值 (与 DualHeadActionNet 的 scale_head 结构一致)。"""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.scale_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        return self.scale_head(latent_pi)


class DualHeadPolicy(ActorCriticPolicy):
    """继承 SB3 ActorCriticPolicy，仅替换 action_net 为双头结构。"""

    def __init__(self, *args, **kwargs):
        # 与 SB3 自定义策略惯例一致: 从 kwargs 取出扩展项再交给父类
        self.scale_only_navigation = bool(kwargs.pop("scale_only_navigation", False))
        self.n_spatial_dirs = int(kwargs.pop("n_spatial_dirs", 16))
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(
            scale_only_navigation=self.scale_only_navigation,
            n_spatial_dirs=self.n_spatial_dirs,
        )
        return data

    def _goal_dir_from_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """与 stage_eval.py 中 baseline pure_mpc 的 dx,dy 规则一致 (含阈值与零向量分支)。"""
        n = self.n_spatial_dirs
        gx = obs[..., n]
        gy = obs[..., n + 1]
        g_norm = torch.sqrt(gx * gx + gy * gy)
        big = g_norm > 1e-6
        safe = torch.where(big, g_norm, torch.ones_like(g_norm))
        dx_all = gx / safe
        dy_all = gy / safe
        dx = torch.where(big, dx_all, torch.zeros_like(gx))
        dy = torch.where(big, dy_all, torch.zeros_like(gy))
        return dx, dy

    def _effective_log_std_scale_only(self) -> torch.Tensor:
        """前两维近似确定性，第三维照常学习。"""
        ls = self.log_std
        head = torch.full_like(ls[..., :2], -20.0)
        return torch.cat([head, ls[..., 2:3]], dim=-1)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        if not self.scale_only_navigation:
            return super()._get_action_dist_from_latent(latent_pi)
        obs = self._pi_obs_ctx
        if obs is None:
            raise RuntimeError("scale_only_navigation requires _pi_obs_ctx (internal bug).")
        dx, dy = self._goal_dir_from_obs(obs)
        scale_mean = self.action_net(latent_pi)
        mean_actions = torch.stack([dx, dy, scale_mean.squeeze(-1)], dim=-1)
        eff_log_std = self._effective_log_std_scale_only()
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, eff_log_std)
        raise NotImplementedError("scale_only_navigation 仅支持 DiagGaussianDistribution")

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        if self.scale_only_navigation:
            self._pi_obs_ctx = obs
            try:
                return super().forward(obs, deterministic=deterministic)
            finally:
                self._pi_obs_ctx = None
        return super().forward(obs, deterministic=deterministic)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        if self.scale_only_navigation:
            self._pi_obs_ctx = obs
            try:
                return super().evaluate_actions(obs, actions)
            finally:
                self._pi_obs_ctx = None
        return super().evaluate_actions(obs, actions)

    def get_distribution(self, obs: torch.Tensor):
        if self.scale_only_navigation:
            self._pi_obs_ctx = obs
            try:
                return super().get_distribution(obs)
            finally:
                self._pi_obs_ctx = None
        return super().get_distribution(obs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        latent_dim = self.mlp_extractor.latent_dim_pi
        if self.scale_only_navigation:
            self.action_net = ScaleOnlyScaleHead(latent_dim)
        else:
            self.action_net = DualHeadActionNet(latent_dim)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
