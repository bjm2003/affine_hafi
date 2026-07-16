"""
Open-only learning probe (Gate G1 pipeline sanity).

Question this answers: with the EXACT policy (DualHeadPolicy), feature extractor,
reward, and env used in the failing 40.7% baseline, can PPO learn the *trivial*
open scenario at all — with the curriculum + failure-replay callbacks REMOVED?

    * ep_rew_mean climbs toward ~120-140 and eval SR → high
        → the learning pipeline is SOUND; the full-run failure is caused by the
          scenario-distribution management (curriculum / failure-replay pinning
          the unsolvable u_trap at max_prob=0.3, starving/contaminating basics).
    * ep_rew_mean stays flat / eval SR stays low on OPEN
        → a fundamental pipeline bug (policy/PPO wiring) reproducible locally,
          debuggable WITHOUT burning training-machine hours.

We proved (scripts/oracle_probe.py) a greedy oracle solves open at 100%, so any
failure here is a LEARNING defect, not a mechanics one.

Usage:
    python scripts/open_learn_probe.py --steps 40000 --n_steps 2048
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config


def eval_sr(model, cfg, seed, n_eps=12):
    """Deterministic open-only success rate + mean steps."""
    from envs.formation_env import FormationEnv
    env = FormationEnv(cfg=cfg, action_type="hafi_3d", scenario_mode="open",
                       enable_projection=False, seed=seed)
    succ, steps_succ = 0, []
    for i in range(n_eps):
        obs, _ = env.reset(seed=seed + i)
        done = False
        s = 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            s += 1
            done = term or trunc
        if info.get("success"):
            succ += 1
            steps_succ.append(s)
    env.close()
    return succ / n_eps, (float(np.mean(steps_succ)) if steps_succ else float("nan"))


class _EvalPrinter:
    """Cheap periodic eval hook via SB3 callback."""
    def __init__(self, model, cfg, seed, every_rollouts=2):
        self.model, self.cfg, self.seed = model, cfg, seed
        self.every = every_rollouts
        self.n = 0

    def __call__(self, locals_, globals_):
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    cfg = Config()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from policies.feature_extractor import FormationFeatureExtractor
    from policies.dual_head_policy import DualHeadPolicy
    from envs.formation_env import FormationEnv

    # Open-only, single env (Windows: interpreted MPC → DummyVecEnv).
    def _make():
        return FormationEnv(cfg=cfg, action_type="hafi_3d", scenario_mode="open",
                            enable_projection=False, seed=args.seed)

    env = VecMonitor(DummyVecEnv([_make]))

    policy_kwargs = {
        "features_extractor_class": FormationFeatureExtractor,
        "features_extractor_kwargs": {"cfg": cfg},
        "n_spatial_dirs": cfg.n_spatial_dirs,
    }

    model = PPO(
        policy=DualHeadPolicy,
        env=env,
        learning_rate=cfg.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        policy_kwargs=policy_kwargs,
        device=args.device,
        verbose=0,
        seed=args.seed,
    )

    print(f"[open-probe] OPEN-ONLY, no curriculum, no failure-replay. "
          f"steps={args.steps} n_steps={args.n_steps} lr={cfg.lr} ent={cfg.ent_coef}")
    print(f"[open-probe] oracle reference on open = 100% SR (env is solvable)\n")

    # Baseline (untrained) eval
    sr0, st0 = eval_sr(model, cfg, seed=9000)
    print(f"[open-probe] t=0        eval_SR={sr0*100:4.0f}%  steps={st0}")

    # Train in chunks, eval + report ep_rew_mean between chunks.
    chunk = args.n_steps
    done = 0
    t_start = time.time()
    while done < args.steps:
        model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                    log_interval=1000)
        done += chunk
        # ep_rew_mean from Monitor
        erm = float("nan")
        if len(model.ep_info_buffer) > 0:
            erm = float(np.mean([e["r"] for e in model.ep_info_buffer]))
        sr, st = eval_sr(model, cfg, seed=9000)
        rate = done / max(time.time() - t_start, 1e-6)
        print(f"[open-probe] t={done:<7} eval_SR={sr*100:4.0f}%  steps={st if st==st else float('nan'):.0f}  "
              f"ep_rew_mean={erm:7.1f}  ({rate:.1f} steps/s)", flush=True)

    out = PROJECT_ROOT / "experiments" / "open_learn_probe.zip"
    model.save(str(out))
    print(f"\n[open-probe] saved {out}")
    print("[open-probe] VERDICT: SR high (>80%) → pipeline SOUND, fix distribution "
          "mgmt.  SR low → fundamental pipeline bug, debug locally.")


if __name__ == "__main__":
    main()
