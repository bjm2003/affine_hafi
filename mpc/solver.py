"""
MPC 求解器 —— 基于 CasADi + IPOPT (非对称纵向代价 + 滑动参考)
=============================================================
代价函数:
  J = Σ_k cost_track(p_k, p_ref(k))             (非对称位置跟踪)
      + Σ_k R_u·‖u_k‖²                         (控制输入正则)
      + Σ_k R_vs·‖u_k − v_ref‖²                (2D 向量速度跟踪)
      + Σ_k R_du·‖u_k − u_{k−1}‖²              (控制平滑)
      + Q_formation·Σ (√(d²+ε) − d_target)²    (编队距离约束)
      + slack_weight·‖slack‖²

  非对称位置跟踪 (asymmetric_tracking=True, 默认):
    dp = p_k − p_ref(k)
    e_lon = dp · path_dir           (纵向: 正=超前, 负=落后)
    e_lat² = ‖dp‖² − e_lon²        (横向误差平方)
    lag = smooth_relu(−e_lon)       (仅落后部分, C^∞ 光滑)
    cost_track = Q_p · (lag² + e_lat²)

    效果: 超前不罚 → MPC 在 horizon 内看到即将超越参考点时不减速 → 全程 v_max.
    smooth_relu(x) = (x + √(x²+ε)) / 2, ε=1e-4, 保证 IPOPT 二阶导数存在.

  对称跟踪 (asymmetric_tracking=False, 消融对照):
    cost_track = Q_p · ‖dp‖²       (各向同性, 超前与落后惩罚相同 → 会减速)

  其中:
    path_dir  = 路径单位方向 (由 RL 子目标确定)
    v_ref     = v_des · path_dir               (2D 期望速度向量)
    p_ref(k)  = p_ref + k·dt·slide·path_dir    (滑动参考)

约束：
  1. 离散全向动力学  x_{k+1} = x_k + vx_k·dt,  y_{k+1} = y_k + vy_k·dt
  2. 输入约束  |vx|, |vy| ≤ v_max
  3. 障碍物安全距离  ‖p_k − o_j(k)‖² ≥ (vehicle_radius + d_safe + r_j)²  − slack
     其中 o_j(k) = o_j(0) + k·dt·v_j  (动态障碍线性位置预测)
  4. 车间安全距离  ‖p_k − p_other‖² ≥ (2·vehicle_radius + d_safe)² − slack

性能优化:
  - CasADi JIT 编译: NLP 函数编译为 C 代码, 求值速度提升 2-5×
  - IPOPT 可接受容差提前终止: 达到 acceptable_tol 后快速退出
  - 完整 warm-start: 原始变量 + 对偶变量 + 边界推回参数
  - 时移初始化: 利用 MPC 滚动窗口结构, 用上一步解的时移版本做初始猜测
"""

import os
import shutil
import sys
import time
import numpy as np
import casadi as ca
from config import Config

_GCC_AVAILABLE: bool | None = None

def _check_gcc() -> bool:
    """检测系统是否安装了 gcc（仅检测一次，缓存结果）"""
    global _GCC_AVAILABLE
    if _GCC_AVAILABLE is None:
        _GCC_AVAILABLE = shutil.which("gcc") is not None
    return _GCC_AVAILABLE


def _ipopt_return_status(stats: dict) -> str:
    """CasADi 不同版本下 IPOPT return_status 可能在顶层或子 dict。"""
    if not isinstance(stats, dict):
        return ""
    rs = stats.get("return_status")
    if rs is not None:
        return str(rs)
    for v in stats.values():
        if isinstance(v, dict):
            rs = v.get("return_status")
            if rs is not None:
                return str(rs)
    return ""


class MPCSolver:
    """单车 MPC 求解器 (滑动参考点 + 各向同性跟踪, CasADi/IPOPT)。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.H = cfg.mpc_horizon
        self.dt = cfg.dt_mpc
        self.n_obs = cfg.n_obs_max
        self.n_others = cfg.n_vehicles - 1

        self.n_u = 2 * self.H
        self.n_slack_obs = self.H * self.n_obs
        self.n_slack_veh = self.H * self.n_others
        self.n_slack_arena = 4 * self.H
        self.n_dec = (self.n_u + self.n_slack_obs + self.n_slack_veh
                      + self.n_slack_arena)

        # 参数维度:
        #   x0(2) + p_ref(2) + path_dir(2) + v_des(1) + current_scale(1)
        #   + obs(5×n_obs) + others(2×n_others) + nominal_dists(n_others)
        #   + arena_bounds(4) [xmin, xmax, ymin, ymax]
        self.n_param = 8 + 5 * self.n_obs + 2 * self.n_others + self.n_others + 4

        self._solver = None
        self._lbw = None
        self._ubw = None
        self._lbg = None
        self._ubg = None
        self._warm_x = None
        self._warm_lam_g = None
        self._warm_lam_x = None

        self._build_solver()

    # ==================================================================
    #  构建 NLP
    # ==================================================================
    def _build_solver(self):
        H = self.H
        dt = self.dt
        cfg = self.cfg
        t_build0 = time.perf_counter()
        has_gcc = _check_gcc()
        print(
            f"[MPC] 正在构建 CasADi NLP + IPOPT "
            f"(H={H}, n_obs={self.n_obs}, gcc={'有' if has_gcc else '无'}) …",
            file=sys.stderr,
            flush=True,
        )

        w = ca.SX.sym("w", self.n_dec)
        p = ca.SX.sym("p", self.n_param)

        U_flat = w[: self.n_u]
        U = ca.reshape(U_flat, 2, H)

        slack_obs_flat = w[self.n_u : self.n_u + self.n_slack_obs]
        _sv_start = self.n_u + self.n_slack_obs
        slack_veh_flat = w[_sv_start : _sv_start + self.n_slack_veh]
        _sa_start = _sv_start + self.n_slack_veh
        slack_arena_flat = w[_sa_start : _sa_start + self.n_slack_arena]

        x0 = p[0:2]
        p_ref = p[2:4]
        path_dir = p[4:6]
        v_des = p[6]
        current_scale = p[7]
        obs_flat = p[8 : 8 + 5 * self.n_obs]
        others_start = 8 + 5 * self.n_obs
        others_flat = p[others_start : others_start + 2 * self.n_others]
        nd_start = others_start + 2 * self.n_others
        nominal_dists = p[nd_start : nd_start + self.n_others]
        arena_p = p[nd_start + self.n_others : nd_start + self.n_others + 4]
        arena_xmin = arena_p[0]
        arena_xmax = arena_p[1]
        arena_ymin = arena_p[2]
        arena_ymax = arena_p[3]

        # ---------- 正向模拟状态 ----------
        X = ca.SX.zeros(2, H + 1)
        X[:, 0] = x0
        for k in range(H):
            X[:, k + 1] = X[:, k] + U[:, k] * dt

        # ---------- 代价函数 ----------
        cost = 0.0

        # 位置跟踪 + 滑动参考
        slide = cfg.slide_ratio * v_des
        if cfg.asymmetric_tracking:
            # 非对称纵向代价: 分解为纵向 (沿 path_dir) + 横向
            #   纵向: 仅惩罚落后 (e_lon < 0), 超前不罚 → 消除 horizon 内减速
            #   横向: 双向惩罚 → 保持航线
            # smooth_relu(x) = (x + √(x²+ε)) / 2 ≈ max(0, x), C^∞ 光滑
            _eps_smooth = 1e-4
            for k in range(H + 1):
                p_ref_k = p_ref + k * dt * slide * path_dir
                dp = X[:, k] - p_ref_k
                e_lon = ca.dot(dp, path_dir)
                e_lat_sq = ca.dot(dp, dp) - e_lon * e_lon
                neg_lon = -e_lon
                lag = (neg_lon + ca.sqrt(neg_lon * neg_lon + _eps_smooth)) * 0.5
                cost += cfg.Q_p * (lag * lag + e_lat_sq)
        else:
            # 对称各向同性跟踪 (消融对照)
            for k in range(H + 1):
                p_ref_k = p_ref + k * dt * slide * path_dir
                dp = X[:, k] - p_ref_k
                cost += cfg.Q_p * ca.dot(dp, dp)

        # 控制输入 + 向量速度跟踪 (与 solver_mpc 一致, 同时约束横向/纵向)
        v_ref = v_des * path_dir
        for k in range(H):
            cost += cfg.R_u * ca.dot(U[:, k], U[:, k])
            if cfg.R_vs > 0.0:
                dv = U[:, k] - v_ref
                cost += cfg.R_vs * ca.dot(dv, dv)

        # 控制平滑
        for k in range(1, H):
            du = U[:, k] - U[:, k - 1]
            cost += cfg.R_du * ca.dot(du, du)

        cost += cfg.slack_weight * ca.dot(slack_obs_flat, slack_obs_flat)
        cost += cfg.slack_weight * ca.dot(slack_veh_flat, slack_veh_flat)
        cost += cfg.slack_weight * ca.dot(slack_arena_flat, slack_arena_flat)

        # 编队误差项 (队友沿本车 path_dir 以 slide 速度前进, 与滑动参考一致)
        for k in range(H + 1):
            p0 = X[:, k]
            for j in range(self.n_others):
                pj0 = others_flat[2 * j : 2 * j + 2]
                pj_k = pj0 + k * dt * slide * path_dir
                target_dist_j = nominal_dists[j] * current_scale
                dist_sq = ca.dot(p0 - pj_k, p0 - pj_k)
                error_dist = ca.sqrt(dist_sq + 1e-6) - target_dist_j
                cost += cfg.Q_formation * error_dist ** 2

        # ---------- 不等式约束 g ≥ 0 ----------
        g_list = []

        idx_obs = 0
        for k in range(1, H + 1):
            for j in range(self.n_obs):
                ox0 = obs_flat[5 * j]
                oy0 = obs_flat[5 * j + 1]
                orad = obs_flat[5 * j + 2]
                ovx = obs_flat[5 * j + 3]
                ovy = obs_flat[5 * j + 4]
                ox_k = ox0 + k * dt * ovx
                oy_k = oy0 + k * dt * ovy
                dp = X[:, k] - ca.vertcat(ox_k, oy_k)
                g_list.append(
                    ca.dot(dp, dp)
                    - (cfg.vehicle_radius + cfg.d_safe + orad) ** 2
                    + slack_obs_flat[idx_obs]
                )
                idx_obs += 1

        idx_veh = 0
        for k in range(1, H + 1):
            for j in range(self.n_others):
                op0 = others_flat[2 * j : 2 * j + 2]
                op_k = op0 + k * dt * slide * path_dir
                dp = X[:, k] - op_k
                g_list.append(
                    ca.dot(dp, dp)
                    - (2.0 * cfg.vehicle_radius + cfg.d_safe) ** 2
                    + slack_veh_flat[idx_veh]
                )
                idx_veh += 1

        # 场地边界半平面约束 (软): 保证轨迹不越过已知矩形围栏
        # 默认 arena = [-1000, 1000, -1000, 1000] → 约束平凡满足
        arena_safe = cfg.vehicle_radius + cfg.d_safe
        idx_arena = 0
        for k in range(1, H + 1):
            g_list.append(
                X[0, k] - (arena_xmin + arena_safe) + slack_arena_flat[idx_arena])
            idx_arena += 1
            g_list.append(
                (arena_xmax - arena_safe) - X[0, k] + slack_arena_flat[idx_arena])
            idx_arena += 1
            g_list.append(
                X[1, k] - (arena_ymin + arena_safe) + slack_arena_flat[idx_arena])
            idx_arena += 1
            g_list.append(
                (arena_ymax - arena_safe) - X[1, k] + slack_arena_flat[idx_arena])
            idx_arena += 1

        g = ca.vertcat(*g_list) if g_list else ca.SX(0, 1)
        n_g = g.shape[0]

        # ---------- 变量边界 ----------
        lbw = []
        ubw = []
        for _ in range(H):
            lbw.extend([-cfg.v_max, -cfg.v_max])
            ubw.extend([cfg.v_max, cfg.v_max])
        for _ in range(self.n_slack_obs + self.n_slack_veh + self.n_slack_arena):
            lbw.append(0.0)
            ubw.append(1e6)

        lbg = [0.0] * n_g
        ubg = [1e20] * n_g

        # ---------- IPOPT 求解器 ----------
        nlp = {"x": w, "f": cost, "g": g, "p": p}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": cfg.ipopt_max_iter,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.warm_start_bound_push": 1e-8,
            "ipopt.warm_start_bound_frac": 1e-8,
            "ipopt.warm_start_slack_bound_push": 1e-8,
            "ipopt.warm_start_slack_bound_frac": 1e-8,
            "ipopt.warm_start_mult_bound_push": 1e-8,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.mu_init": 1e-3,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.acceptable_iter": 3,
            "ipopt.acceptable_constr_viol_tol": 1e-2,
            "print_time": 0,
        }

        jit_ok = False
        if has_gcc:
            try:
                jit_opts = dict(opts)
                jit_opts["jit"] = True
                jit_opts["compiler"] = "shell"
                jit_opts["jit_options"] = {"flags": ["-O2"], "verbose": False}
                cache_dir = os.path.join(os.path.dirname(__file__), os.pardir, ".casadi_cache")
                cache_dir = os.path.abspath(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                prev_cwd = os.getcwd()
                try:
                    os.chdir(cache_dir)
                    print(
                        "[MPC] JIT 编译中（首次常需数十秒，属正常；"
                        "缓存命中后会快很多）…",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._solver = ca.nlpsol("mpc", "ipopt", nlp, jit_opts)
                    jit_ok = True
                finally:
                    os.chdir(prev_cwd)
            except Exception:
                pass
        if not jit_ok:
            print(
                "[MPC] 未使用 JIT（无 gcc 或编译失败），使用解释模式构建 …",
                file=sys.stderr,
                flush=True,
            )
            self._solver = ca.nlpsol("mpc", "ipopt", nlp, opts)

        dt_build = time.perf_counter() - t_build0
        print(
            f"[MPC] 求解器就绪: JIT={'是' if jit_ok else '否'}, 构建耗时 {dt_build:.1f}s",
            file=sys.stderr,
            flush=True,
        )

        self._lbw = np.array(lbw)
        self._ubw = np.array(ubw)
        self._lbg = np.array(lbg)
        self._ubg = np.array(ubg)

    # ==================================================================
    #  求解
    # ==================================================================
    def solve(
        self,
        current_pos: np.ndarray,
        ref_pos: np.ndarray,
        obstacles: list,
        other_positions: list,
        current_scale: float = 1.0,
        nominal_dists: np.ndarray = None,
        path_dir: np.ndarray = None,
        arena_bounds: "tuple | None" = None,
    ) -> tuple:
        """
        求解一次 MPC (滑动参考点)。

        Parameters
        ----------
        path_dir : (2,) 路径单位方向向量 (由 RL 子目标确定)。
                   若为 None, 回退到 normalize(ref_pos - current_pos)。
        nominal_dists : (n_others,) 该车到每个邻车的无缩放标称距离。

        Returns
        -------
        u : (2,) 最优控制 [vx, vy]
        feasible : bool 是否求解成功
        slack_sum : float 松弛变量分量之和 (≥0, 越大表示软约束越紧)
        return_status : str IPOPT/CasADi 返回状态字符串
        """
        # ---------- 路径方向 ----------
        if path_dir is None:
            rv = ref_pos - current_pos
            rd = float(np.linalg.norm(rv))
            path_dir = rv / rd if rd > 1e-6 else np.array([1.0, 0.0])

        v_des = float(np.clip(self.cfg.v_des_ratio, 0.0, 1.0) * self.cfg.v_max)

        # ---------- 构造参数向量 ----------
        param = np.zeros(self.n_param)
        param[0:2] = current_pos
        param[2:4] = ref_pos
        param[4:6] = path_dir
        param[6] = v_des
        param[7] = current_scale

        for j in range(self.n_obs):
            base = 8 + 5 * j
            if j < len(obstacles):
                pos_j, rad_j, vel_j = obstacles[j]
                param[base:base + 2] = pos_j
                param[base + 2] = rad_j
                param[base + 3:base + 5] = vel_j
            else:
                param[base:base + 2] = [1000.0, 1000.0]
                param[base + 2] = 0.0
                param[base + 3:base + 5] = [0.0, 0.0]

        offset = 8 + 5 * self.n_obs
        for j in range(self.n_others):
            if j < len(other_positions):
                param[offset + 2 * j : offset + 2 * j + 2] = other_positions[j]
            else:
                param[offset + 2 * j : offset + 2 * j + 2] = [1000.0, 1000.0]

        nd_offset = offset + 2 * self.n_others
        if nominal_dists is not None:
            param[nd_offset : nd_offset + self.n_others] = nominal_dists[:self.n_others]
        else:
            param[nd_offset : nd_offset + self.n_others] = self.cfg.d_form

        arena_offset = nd_offset + self.n_others
        if arena_bounds is not None:
            param[arena_offset : arena_offset + 4] = arena_bounds
        else:
            param[arena_offset : arena_offset + 4] = [-1000.0, 1000.0, -1000.0, 1000.0]

        # ---------- 初始猜测 ----------
        if self._warm_x is not None:
            x0_guess = self._warm_x
        else:
            x0_guess = np.zeros(self.n_dec)
            for k in range(self.H):
                t = (k + 1) / self.H
                target = current_pos + t * (ref_pos - current_pos)
                vel = (target - current_pos) / ((k + 1) * self.dt) if (k + 1) * self.dt > 0 else np.zeros(2)
                vel = np.clip(vel, -self.cfg.v_max, self.cfg.v_max)
                x0_guess[2 * k : 2 * k + 2] = vel

        # ---------- 调用求解器 (含完整 warm-start) ----------
        try:
            kwargs = dict(
                x0=x0_guess,
                p=param,
                lbx=self._lbw,
                ubx=self._ubw,
                lbg=self._lbg,
                ubg=self._ubg,
            )
            if self._warm_lam_g is not None:
                kwargs["lam_g0"] = self._warm_lam_g
            if self._warm_lam_x is not None:
                kwargs["lam_x0"] = self._warm_lam_x

            sol = self._solver(**kwargs)
            w_opt = np.array(sol["x"]).flatten()

            self._warm_lam_g = np.array(sol["lam_g"]).flatten()
            self._warm_lam_x = np.array(sol["lam_x"]).flatten()

            self._warm_x = self._shift_solution(w_opt)

            u_opt = w_opt[0:2]
            stats = self._solver.stats()
            feasible = bool(stats.get("success", False))
            slack_vec = w_opt[self.n_u :]
            slack_sum = float(np.sum(np.maximum(slack_vec, 0.0)))
            return_status = _ipopt_return_status(stats)
            if not return_status:
                return_status = "success" if feasible else "failed"
            return u_opt, feasible, slack_sum, return_status

        except Exception:
            self._warm_x = None
            self._warm_lam_g = None
            self._warm_lam_x = None
            return np.zeros(2), False, 0.0, "exception"

    def _shift_solution(self, w_opt: np.ndarray) -> np.ndarray:
        """时移 warm-start: 利用 MPC 滚动窗口结构优化初始猜测。"""
        shifted = w_opt.copy()
        n_u = self.n_u

        if n_u >= 4:
            shifted[:n_u - 2] = w_opt[2:n_u]
            shifted[n_u - 2:n_u] = w_opt[n_u - 2:n_u]

        n_so = self.n_slack_obs
        n_per_obs = self.n_obs
        if n_so > n_per_obs:
            start = n_u
            old = w_opt[start:start + n_so]
            shifted[start:start + n_so - n_per_obs] = old[n_per_obs:]
            shifted[start + n_so - n_per_obs:start + n_so] = old[-n_per_obs:]

        n_sv = self.n_slack_veh
        n_per_veh = self.n_others
        if n_sv > n_per_veh:
            start = n_u + n_so
            old = w_opt[start:start + n_sv]
            shifted[start:start + n_sv - n_per_veh] = old[n_per_veh:]
            shifted[start + n_sv - n_per_veh:start + n_sv] = old[-n_per_veh:]

        n_sa = self.n_slack_arena
        n_per_arena = 4
        if n_sa > n_per_arena:
            start = n_u + n_so + n_sv
            old = w_opt[start:start + n_sa]
            shifted[start:start + n_sa - n_per_arena] = old[n_per_arena:]
            shifted[start + n_sa - n_per_arena:start + n_sa] = old[-n_per_arena:]

        return shifted

    def reset_warm_start(self):
        """重置 warm-start 缓存 (episode 开始时调用)"""
        self._warm_x = None
        self._warm_lam_g = None
        self._warm_lam_x = None
