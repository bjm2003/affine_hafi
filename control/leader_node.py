from typing import Optional, Tuple
import numpy as np

from config import Config
from .interfaces import ReferencePacket


class LeaderNode:
    """High-level leader logic wrapper used by both sim and future vehicle nodes."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def apply_scale_command(self, raw_scale: float, current_scale: float) -> float:
        """Map raw scale action [-1,1] to smoothed scale in [s_min, s_max]."""
        raw = float(np.clip(raw_scale, -1.0, 1.0))
        target = self.cfg.s_min + (raw + 1.0) / 2.0 * (self.cfg.s_max - self.cfg.s_min)
        delta_s = np.clip(target - current_scale, -self.cfg.max_delta_s, self.cfg.max_delta_s)
        return float(np.clip(current_scale + delta_s, self.cfg.s_min, self.cfg.s_max))

    def build_reference_packet(
        self,
        leader_idx: int,
        step_count: int,
        center_estimate: np.ndarray,
        goal: np.ndarray,
        dx: float,
        dy: float,
        scale: float,
        last_subgoal: Optional[np.ndarray],
    ) -> Tuple[ReferencePacket, np.ndarray]:
        """Create leader reference packet and updated smoothed subgoal."""
        subgoal_offset = self.cfg.R0 * np.array([dx, dy], dtype=np.float64)
        dist_to_goal = float(np.linalg.norm(center_estimate - goal))
        carrot = center_estimate + subgoal_offset
        band_outer = self.cfg.goal_homing_band * self.cfg.R0

        if dist_to_goal < self.cfg.R0:
            p_c_ref_raw = goal.copy()
        elif dist_to_goal < band_outer:
            # Blend carrot toward the true goal on the final approach so an
            # imperfect heading can't trap the centroid in a limit cycle at
            # radius R0 (w=0 at the R0 snap edge → goal, w=1 at band_outer → carrot).
            w = (dist_to_goal - self.cfg.R0) / (band_outer - self.cfg.R0)
            p_c_ref_raw = (1.0 - w) * goal + w * carrot
        else:
            p_c_ref_raw = carrot

        if last_subgoal is None:
            p_c_ref = p_c_ref_raw.copy()
        else:
            p_c_ref = (
                (1.0 - self.cfg.ref_ema_alpha) * last_subgoal
                + self.cfg.ref_ema_alpha * p_c_ref_raw
            )

        packet = ReferencePacket(
            leader_idx=int(leader_idx),
            p_c_ref=p_c_ref.copy(),
            scale=float(scale),
            step=int(step_count),
        )
        return packet, p_c_ref

    def build_vehicle_refs(
        self,
        packet: ReferencePacket,
        formation_offsets: np.ndarray,
        n_vehicles: int,
    ) -> np.ndarray:
        """Build per-vehicle target refs from leader packet."""
        refs = np.zeros((n_vehicles, 2), dtype=np.float64)
        for i in range(n_vehicles):
            refs[i] = packet.p_c_ref + packet.scale * formation_offsets[i]
        return refs
