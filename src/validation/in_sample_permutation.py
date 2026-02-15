"""In-sample permutation test (step 2)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Type

import numpy as np
import pandas as pd

from ..optimization import grid_search
from ..permutation import permute_bars


def in_sample_permutation_test(
    df: pd.DataFrame,
    strategy_class: Type,
    param_grid: Dict[str, Iterable] | List[Dict],
    n_perms: int = 1000,
) -> Dict:
    """
    Step 2 of validation framework.
    """
    if n_perms <= 0:
        raise ValueError("n_perms must be > 0")

    best_params, real_pf = grid_search(df, strategy_class, param_grid)

    perm_pfs: List[float] = []
    for _ in range(int(n_perms)):
        df_perm = permute_bars(df)
        _, perm_pf = grid_search(df_perm, strategy_class, param_grid)
        perm_pfs.append(float(perm_pf))

    # Correct permutation p-value per Phipson & Smyth (2010):
    # p = (1 + count(perm >= real)) / (1 + n_perms)
    # Ensures p is never exactly 0 and accounts for the real value itself.
    count_ge = int(np.sum(np.asarray(perm_pfs) >= real_pf))
    p_value = float((1 + count_ge) / (1 + len(perm_pfs)))
    return {
        "best_params": best_params,
        "real_pf": float(real_pf),
        "perm_pfs": perm_pfs,
        "p_value": p_value,
        "passed": bool(p_value < 0.05),
    }
