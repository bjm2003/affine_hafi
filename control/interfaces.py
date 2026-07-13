from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ReferencePacket:
    """Leader high-level reference output for one RL step."""

    leader_idx: int
    p_c_ref: np.ndarray
    scale: float
    step: int


@dataclass
class VehicleMPCInput:
    """Per-vehicle MPC input payload."""

    current_pos: np.ndarray
    ref_pos: np.ndarray
    obstacles: List[Tuple[np.ndarray, float, np.ndarray]]
    other_positions: List[np.ndarray]
    current_scale: float
    nominal_dists: Optional[np.ndarray] = None
    path_dir: Optional[np.ndarray] = None
    arena_bounds: Optional[Tuple[float, float, float, float]] = None


@dataclass
class VehicleControlOutput:
    """Per-vehicle MPC solve result."""

    control: np.ndarray
    feasible: bool
    solve_time_s: float = 0.0
    # IPOPT / CasADi stats (可选, 供日志与分析)
    mpc_return_status: str = ""
    slack_sum: float = 0.0  # 障碍/编队松弛变量之和 (软约束违反程度)
