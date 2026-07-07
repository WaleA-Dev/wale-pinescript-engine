"""Strategy registry for CLI and validation pipelines."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Dict, Type

from .base import BaseStrategy
from .donchian import DonchianBreakoutStrategy
from .ema_crossover import EMACrossoverStrategy
from .ndx_trader import NdxTraderStrategy

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "donchian": DonchianBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "ema_crossover": EMACrossoverStrategy,
    "ema": EMACrossoverStrategy,
    "ndx_trader": NdxTraderStrategy,
}

VALIDATION_STATUS: Dict[str, str] = {}

_DISCOVERED = False


def discover_strategies() -> Dict[str, Type[BaseStrategy]]:
    """
    Import every module in this package and register all BaseStrategy
    subclasses found. Broken modules (e.g. half-finished translations)
    are skipped so one bad file can't take the registry down.
    """
    global _DISCOVERED
    strategy_dir = Path(__file__).resolve().parent
    for candidate in sorted(strategy_dir.glob("*.py")):
        module_name = candidate.stem
        if module_name.startswith("_") or module_name == "base":
            continue
        if _normalize(module_name) in STRATEGY_REGISTRY:
            continue
        try:
            mod = importlib.import_module(f"src.strategies.{module_name}")
        except Exception:
            continue
        for obj in vars(mod).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseStrategy)
                and obj is not BaseStrategy
            ):
                register_strategy(module_name, obj)
                break
    _DISCOVERED = True
    return STRATEGY_REGISTRY


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")


def register_strategy(name: str, cls: Type[BaseStrategy]) -> None:
    STRATEGY_REGISTRY[_normalize(name)] = cls


def _try_dynamic_strategy(name: str) -> Type[BaseStrategy] | None:
    module_name = _normalize(name)
    strategy_dir = Path(__file__).resolve().parent
    candidate = strategy_dir / f"{module_name}.py"
    if not candidate.exists():
        return None

    mod = importlib.import_module(f"src.strategies.{module_name}")
    # Only classes *defined in* the module — not bases imported into it
    class_candidates = [
        obj for obj in vars(mod).values()
        if isinstance(obj, type) and issubclass(obj, BaseStrategy)
        and obj is not BaseStrategy and getattr(obj, "__module__", "") == mod.__name__
    ]
    if not class_candidates:
        return None
    cls = class_candidates[0]
    register_strategy(module_name, cls)
    return cls


def get_strategy_class(name: str) -> Type[BaseStrategy]:
    key = _normalize(name)
    if key in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[key]

    dynamic_cls = _try_dynamic_strategy(key)
    if dynamic_cls is not None:
        return dynamic_cls

    raise ValueError(f"Unknown strategy: {name}. Available: {sorted(STRATEGY_REGISTRY.keys())}")


def update_validation_status(name: str, status: str) -> None:
    VALIDATION_STATUS[_normalize(name)] = status


__all__ = [
    "BaseStrategy",
    "DonchianBreakoutStrategy",
    "EMACrossoverStrategy",
    "NdxTraderStrategy",
    "STRATEGY_REGISTRY",
    "VALIDATION_STATUS",
    "discover_strategies",
    "get_strategy_class",
    "register_strategy",
    "update_validation_status",
]
