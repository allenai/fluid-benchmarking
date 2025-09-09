from typing import Literal, Optional, Tuple

import numpy as np

from fluid_benchmarking import irt_utils


def ability_estimate(
    lm_responses: np.ndarray,
    irt_model: np.ndarray,
    method: Literal["map", "mle", "MAP", "MLE"] = "map",
    D: float = 1.0,
    mu0: float = 0.0,
    sigma0: float = 1.0,
    theta0: Optional[float] = None,
    theta_range: Tuple[float, float] = (-4.0, 4.0),
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Ability estimate for a 2PL IRT model using either MAP with normal prior or MLE.

    Parameters
    ----------
    lm_responses : array-like, shape (n_items,)
        Binary responses (0/1). NaN entries are ignored (items not administered).
    irt_model : array-like, shape (n_items, 2)
        Item parameters: columns for discrimination and difficulty.
    method : {"map", "mle"} (case-insensitive)
        Estimation method (MAP with normal prior or MLE).
    D : float
        Logistic scaling constant (1.0 by default; 1.7 is also common).
    mu0, sigma0 : float
        Prior mean and standard deviation for MAP; ignored by MLE.
    theta0 : float or None
        Optional starting value; defaults to mu0, clipped to theta_range.
    theta_range : (low, high)
        Search/constraint interval for ability.
    tol : float
        Convergence tolerance on the score |T(theta)|.
    max_iter : int
        Maximum Newton iterations before falling back to bisection (if needed).

    Returns
    -------
    float
        The ability estimate.
    """

    # Checks
    method = method.lower()
    if method not in {"map", "mle"}:
        raise ValueError("method must be 'map' or 'mle'.")

    lm_responses = np.asarray(lm_responses, dtype=float)
    irt_model = np.asarray(irt_model, dtype=float)
    
    if irt_model.ndim != 2 or irt_model.shape[1] != 2:
        raise ValueError("irt_model must be an (n_items, 2) array with columns [a, b].")
    n_items = irt_model.shape[0]

    if lm_responses.shape != (n_items,):
        raise ValueError("lm_responses must be shape (n_items,) aligned with irt_model.")

    # Filter out NaN responses (items not administered)
    mask = ~np.isnan(lm_responses)
    a = irt_model[mask, 0]
    b = irt_model[mask, 1]
    lm_responses = lm_responses[mask]

    # Prior precision (1/sigma^2 for MAP; 0 for MLE)
    if method == "map":
        if sigma0 <= 0:
            raise ValueError("sigma0 must be positive for MAP.")
        inv_sigma2 = 1.0 / (sigma0 * sigma0)
    else:
        inv_sigma2 = 0.0

    low, high = float(theta_range[0]), float(theta_range[1])
    if low >= high:
        raise ValueError("theta_range must have low < high.")

    def score(
        theta: float
    ) -> float:
        z = D * a * (theta - b)
        P = irt_utils.sigmoid_stable(z)
        prior_term = (mu0 - theta) * inv_sigma2  # 0 for MLE
        likelihood_term = D * np.sum(a * (lm_responses - P))
        return prior_term + likelihood_term

    def score_prime(
        theta: float
    ) -> float:
        z = D * a * (theta - b)
        P = irt_utils.sigmoid_stable(z)
        PQ = P * (1.0 - P)
        prior_term = -inv_sigma2  # 0 for MLE
        likelihood_term = -(D**2) * np.sum(a * a * PQ)
        return prior_term + likelihood_term

    # Start at prior mean or provided theta0, clipped to range
    theta = mu0 if theta0 is None else theta0
    theta = float(np.clip(theta, low, high))

    # Newton–Raphson with simple backtracking and range projection
    for _ in range(max_iter):
        T = score(theta)
        if np.abs(T) < tol:
            return theta

        Tp = score_prime(theta)
        if not np.isfinite(Tp) or Tp == 0.0:
            break

        step = -T / Tp
        new_theta = theta + step

        # Project to range and backtrack if the score does not improve
        if new_theta < low or new_theta > high or not np.isfinite(new_theta):
            new_theta = float(np.clip(new_theta, low, high))

        # Backtracking: ensure |score| decreases
        T_abs = abs(T)
        for _bt in range(15):
            T_new = score(new_theta)
            if abs(T_new) < T_abs or not np.isfinite(T_new):
                break
            new_theta = 0.5 * (new_theta + theta)

        theta = new_theta

    # Fallback: try bisection within [low, high]
    sL = score(low)
    sH = score(high)

    # If we bracket a root, bisection will find it (T is strictly decreasing)
    if sL * sH <= 0:
        lo, hi = low, high
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            sM = score(mid)
            if abs(sM) < tol:
                return mid
            if sL * sM > 0:
                lo, sL = mid, sM
            else:
                hi = mid
        return 0.5 * (lo + hi)

    return high if (sL > 0 and sH > 0) else low
