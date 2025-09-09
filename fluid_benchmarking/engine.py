from typing import Any, Callable, Dict, Literal, Optional, Tuple

import numpy as np

from fluid_benchmarking import estimators, irt_utils


def select_mfi(
    theta: float, 
    irt_model: np.ndarray, 
    used_mask: np.ndarray, 
    D: float,
) -> int:
    """
    Select the next item by maximum Fisher information under a 2PL IRT model.

    Parameters
    ----------
    theta : float
        Current ability estimate at which to compute information.
    irt_model : array-like, shape (n_items, 2)
        Item parameters: columns for discrimination and difficulty.
    used_mask : array-like of bool, shape (n_items,)
        Mask of items already administered (True means "used" and therefore excluded).
    D : float
        Logistic scaling constant (1.0 by default; 1.7 is also common).

    Returns
    -------
    int
        Index of the selected item (argmax of Fisher information among unused items).

    Notes
    -----
    Ties are broken by ``np.argmax`` (first maximal index).
    """
    a = irt_model[:, 0]
    b = irt_model[:, 1]
    fi = irt_utils.fisher_information(theta, a, b, D=D)
    fi_masked = np.where(~used_mask, fi, -np.inf)
    idx = int(np.argmax(fi_masked))
    if not np.isfinite(fi_masked[idx]):
        raise RuntimeError("No available items to select. All items administered?")
    return idx


def run_fluid_benchmarking(
    *,
    # Core inputs
    lm_responses: np.ndarray,
    irt_model: np.ndarray,
    start_ability: float = 0.0,
    n_max: int = 100,

    # Ability-estimation kwargs
    method: Literal["map", "mle", "MAP", "MLE"] = "map",
    D: float = 1.0,
    mu0: float = 0.0,
    sigma0: float = 1.0,
    theta_range: Tuple[float, float] = (-4.0, 4.0),
    tol: float = 1e-6,
    max_iter: int = 100,

    # Optional override for the estimator (defaults to estimators.ability_estimate)
    estimator: Optional[Callable[..., float]] = None,
) -> Dict[str, Any]:
    """
    Run fluid benchmarking for a 2PL IRT model using item selection by maximum 
    Fisher information at the current ability estimate.

    At each step, select the unused item that maximizes Fisher information at the
    current ability estimate, record its binary response from ``lm_responses``,
    and update the ability using the provided estimator (MAP or MLE). Unseen items
    remain NaN in the running response vector so the estimator ignores them.

    Parameters
    ----------
    lm_responses : array-like, shape (n_items,)
        Binary responses (0/1). NaN entries are ignored (items not administered).
    irt_model : array-like, shape (n_items, 2)
        Item parameters: columns for discrimination and difficulty.
    start_ability : float
        Ability used for the first selection (by maximum Fisher information) and as
        the initial ``theta0`` for the first ability update.
    n_max : int
        Maximum number of items to administer. If ``n_max < 1``, returns empty lists.
    method : {"map", "mle"} (case-insensitive)
        Estimation method (MAP with normal prior or MLE).
    D : float
        Logistic scaling constant (1.0 by default; 1.7 is also common).
    mu0, sigma0 : float
        Prior mean and standard deviation for MAP; ignored by MLE.
    theta_range : (low, high)
        Search/constraint interval for ability.
    tol : float
        Convergence tolerance on the score |T(theta)|.
    max_iter : int, default=100
        Maximum Newton iterations before falling back to bisection (if needed).

    estimator : callable or None
        Ability estimator. If ``None``, defaults to ``estimators.ability_estimate``.

    Returns
    -------
    dict
        A dictionary with:
        - ``"items_fb"`` : list[int]
            Indices of items administered in selection order.
        - ``"abilities_fb"`` : list[float]
            Ability estimates after each administered item.

    Notes
    -----
    - The running response vector passed to ``estimator`` is initialized with NaN and
      filled as items are administered; unseen items remain NaN so they are ignored by
      the default ``ability_estimate`` implementation.
    - Selection always chooses the unused item with the greatest Fisher information at
      the current estimate (``abilities_fb[-1]``). Ties are resolved by ``np.argmax``.
    - Iteration stops when ``n_max`` items have been administered or all items are used.
    """

    # Checks
    method = method.lower()
    if method not in {"map", "mle"}:
        raise ValueError("method must be 'map' or 'mle'.")

    if irt_model.ndim != 2 or irt_model.shape[1] != 2:
        raise ValueError("irt_model must be an (n_items, 2) array with columns [a, b].")
    n_items = irt_model.shape[0]

    if lm_responses.shape != (n_items,):
        raise ValueError("lm_responses must be shape (n_items,) aligned with irt_model.")

    if n_max < 1:
        return {"abilities_fb": [], "items_fb": []}

    # Estimator default
    if estimator is None:
        estimator = estimators.ability_estimate

    # State
    used_mask = np.zeros(n_items, dtype=bool)
    items = []
    abilities = []
    lm_responses_running = np.full(n_items, np.nan, dtype=float)

    # First item by MFI at start_ability
    idx0 = select_mfi(start_ability, irt_model, used_mask, D=D)
    used_mask[idx0] = True
    items.append(idx0)

    r0 = lm_responses[idx0]
    if not (r0 == 0.0 or r0 == 1.0):
        raise ValueError(f"Response for item {idx0} must be 0 or 1.")
    lm_responses_running[idx0] = r0

    # Initial ability estimate
    th = float(
        estimator(
            lm_responses=lm_responses_running,
            irt_model=irt_model,
            method=method,
            D=D,
            mu0=mu0,
            sigma0=sigma0,
            theta0=start_ability,
            theta_range=theta_range,
            tol=tol,
            max_iter=max_iter,
        )
    )
    abilities.append(th)

    # Loop: keep selecting by MFI at current ability
    while len(items) < n_max and len(items) < n_items:
        idx = select_mfi(abilities[-1], irt_model, used_mask, D=D)
        used_mask[idx] = True
        items.append(idx)

        r = lm_responses[idx]
        if not (r == 0.0 or r == 1.0):
            raise ValueError(f"Response for item {idx0} must be 0 or 1.")
        lm_responses_running[idx] = r

        th = float(
            estimator(
                lm_responses=lm_responses_running,
                irt_model=irt_model,
                method=method,
                D=D,
                mu0=mu0,
                sigma0=sigma0,
                theta0=abilities[-1],
                theta_range=theta_range,
                tol=tol,
                max_iter=max_iter,
            )
        )
        abilities.append(th)

    return {"abilities_fb": abilities, "items_fb": items}
