# pyre-strict
"""Shared optimization primitives used across solvers and objectives."""

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def _fista(
    grad_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    project_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x0: npt.NDArray[np.float64],
    step_size: float,
    max_iter: int,
    tol: float,
    minimize: bool,
) -> tuple[npt.NDArray[np.float64], float, int, bool]:
    r"""FISTA projected gradient iteration (Beck & Teboulle, 2009).

    Runs accelerated proximal gradient on a smooth objective over a convex
    set.  Works for both minimization (``minimize=True``) and maximization
    (``minimize=False``).

    The iterates are

    .. math::

        x_{k+1} &= \Pi_C\!\bigl(y_k \pm \alpha\,\nabla f(y_k)\bigr), \\
        t_{k+1} &= \tfrac{1 + \sqrt{1 + 4t_k^2}}{2}, \\
        y_{k+1} &= x_{k+1} + \tfrac{t_k-1}{t_{k+1}}(x_{k+1} - x_k),

    where :math:`+` is used for maximization and :math:`-` for minimization.
    The best iterate seen is returned to guard against non-monotone steps
    caused by momentum.

    Parameters
    ----------
    grad_fn : Callable
        Gradient of the smooth objective at a point ``x``.
    obj_fn : Callable
        Value of the smooth objective at a point ``x``.
    project_fn : Callable
        Euclidean projection onto the feasible set :math:`C`.
    x0 : npt.NDArray[np.float64]
        Feasible starting point, shape ``(d,)``.
    step_size : float
        Constant step size ``alpha``.  Must satisfy ``alpha <= 1/L`` where
        ``L`` is the Lipschitz constant of ``grad_fn``.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the iterate-change norm.
    minimize : bool
        ``True`` for gradient descent (minimization),
        ``False`` for gradient ascent (maximization).

    Returns
    -------
    tuple
        ``(best_x, best_obj, n_iterations, converged)`` where ``best_x``
        is the iterate with the best objective value seen, ``n_iterations``
        is the count when the loop stopped, and ``converged`` is ``True``
        if the step-norm tolerance was reached before ``max_iter``.
    """
    sign = -1.0 if minimize else 1.0
    x = x0.copy()
    y = x.copy()
    t = 1.0
    best_x = x.copy()
    best_obj = obj_fn(x)

    for k in range(max_iter):
        x_new = project_fn(y + sign * step_size * grad_fn(y))

        obj = obj_fn(x_new)
        if (minimize and obj < best_obj) or (not minimize and obj > best_obj):
            best_obj = obj
            best_x = x_new.copy()

        if float(np.linalg.norm(x_new - x)) < tol:
            return best_x, best_obj, k + 1, True

        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x = x_new
        t = t_new

    return best_x, best_obj, max_iter, False
