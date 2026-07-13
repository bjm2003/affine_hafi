from config import Config
from mpc.solver import MPCSolver
from .interfaces import VehicleMPCInput, VehicleControlOutput
from time import perf_counter


class MPCNode:
    """Per-vehicle MPC wrapper with a stable interface for deployment."""

    def __init__(self, cfg: Config):
        self.solver = MPCSolver(cfg)

    def reset_warm_start(self) -> None:
        self.solver.reset_warm_start()

    def compute_control(self, data: VehicleMPCInput) -> VehicleControlOutput:
        t0 = perf_counter()
        u, ok, slack_sum, ret_status = self.solver.solve(
            current_pos=data.current_pos,
            ref_pos=data.ref_pos,
            obstacles=data.obstacles,
            other_positions=data.other_positions,
            current_scale=data.current_scale,
            nominal_dists=data.nominal_dists,
            path_dir=data.path_dir,
            arena_bounds=data.arena_bounds,
        )
        solve_time_s = perf_counter() - t0
        return VehicleControlOutput(
            control=u,
            feasible=ok,
            solve_time_s=solve_time_s,
            mpc_return_status=ret_status,
            slack_sum=slack_sum,
        )
