"""Scenario generator registry.

Usage:
    from envs.scenario_generators import build_scenario, list_scenarios

    scen = build_scenario("curved_slot")
    instance = scen.sample(rng, cfg)
"""

from __future__ import annotations
from typing import Dict, Type

from .base import BaseScenario, ScenarioInstance, ObstacleStatic, ObstacleDynamic
from .open_scenario import OpenScenario
from .corridor import CorridorScenario
from .s_corridor import SCorridorScenario
from .z_corridor import ZCorridorScenario
from .u_trap import UTrapScenario
from .dynamic import DynamicScenario
from .curved_slot import CurvedSlotScenario
from .sequential_doorways import SequentialDoorwaysScenario
from .asymmetric_density import AsymmetricDensityScenario
from .interior_injection import InteriorInjectionScenario


# 6 training scenarios (match config.train_scenario_probs order):
#   open, corridor, s_corridor, z_corridor, u_trap, dynamic
TRAINING_SCENARIOS = [
    "open", "corridor", "s_corridor", "z_corridor", "u_trap", "dynamic",
]

# 4 killer scenarios (M2 gate + main experiments Table 1 new columns)
KILLER_SCENARIOS = [
    "curved_slot", "sequential_doorways", "asymmetric_density", "interior_injection",
]


_REGISTRY: Dict[str, Type[BaseScenario]] = {
    "open": OpenScenario,
    "corridor": CorridorScenario,
    "s_corridor": SCorridorScenario,
    "z_corridor": ZCorridorScenario,
    "u_trap": UTrapScenario,
    "dynamic": DynamicScenario,
    "curved_slot": CurvedSlotScenario,
    "sequential_doorways": SequentialDoorwaysScenario,
    "asymmetric_density": AsymmetricDensityScenario,
    "interior_injection": InteriorInjectionScenario,
}


def build_scenario(name: str, **kwargs) -> BaseScenario:
    """Instantiate a scenario generator by name.

    kwargs are forwarded to the generator's __init__ (for custom parameters like
    gap_width, arc_radius, etc.).
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown scenario: {name}. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def list_scenarios():
    return list(_REGISTRY.keys())


__all__ = [
    "BaseScenario",
    "ScenarioInstance",
    "ObstacleStatic",
    "ObstacleDynamic",
    "build_scenario",
    "list_scenarios",
    "TRAINING_SCENARIOS",
    "KILLER_SCENARIOS",
]
