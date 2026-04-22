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
    backtrack_factor: float | None = None,
    per_iter_callback: Callable[[npt.NDArray[np.float64], float], None]
    | None = None,
) -> tuple[npt.NDArray[np.float64], float, int, bool]:
    r"""FISTA projected gradient iteration (Beck & Teboulle, 2009).

    Runs accelerated proximal gradient on a smooth objective over a convex
    set.  Works for both minimization (``minimize=True``) and maximization
    (``minimize=False``).

    The iterates are

    .. math::

        x_{k+1} &= \Pi_C\!\bigl(y_k \pm \alpha_k\,\nabla f(y_k)\bigr), \\
        t_{k+1} &= \tfrac{1 + \sqrt{1 + 4t_k^2}}{2}, \\
        y_{k+1} &= x_{k+1} + \tfrac{t_k-1}{t_{k+1}}(x_{k+1} - x_k),

    where :math:`+` is used for maximization and :math:`-` for minimization.
    The best iterate seen is returned to guard against non-monotone steps
    caused by momentum.

    When ``backtrack_factor`` is given, the step size :math:`\alpha_k` is
    found each iteration via the backtracking rule of Parikh & Boyd,
    *Proximal Algorithms* §4.3: starting from the previous :math:`\alpha`,
    shrink by ``backtrack_factor`` until the sufficient-progress condition

    .. math::

        \operatorname{sign} \cdot f(x_{k+1})
        \;\ge\;
        \operatorname{sign} \cdot \bigl[
            f(y_k) + \nabla f(y_k)^\top (x_{k+1} - y_k)
        \bigr]
        - \frac{\|x_{k+1} - y_k\|^2}{2\alpha_k}

    holds (``sign = -1`` for minimization, ``+1`` for maximization).
    The accepted :math:`\alpha_k` is carried to the next iteration as the
    warm-start estimate.

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
        Initial step size :math:`\alpha_0`.  With backtracking this is an
        upper-bound estimate; the algorithm shrinks it as needed.  Without
        backtracking it is used as a constant and must satisfy
        :math:`\alpha \le 1/L`.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the iterate-change norm.
    minimize : bool
        ``True`` for gradient descent (minimization),
        ``False`` for gradient ascent (maximization).
    backtrack_factor : float or None
        Expansion factor :math:`\eta > 1` for the Lipschitz estimate
        (equivalently, the divisor applied to :math:`\alpha` on each failed
        backtracking step).  ``None`` disables backtracking and uses a
        constant step size.
    per_iter_callback : Callable or None
        Optional function called after each iterate update with
        ``(x_new, obj_new)``.  Used by outer solvers to record per-iteration
        diagnostics such as duality gaps.

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
    alpha = step_size

    for k in range(max_iter):
        grad_y = grad_fn(y)

        if backtrack_factor is not None:
            obj_y = obj_fn(y)
            while True:
                x_new = project_fn(y + sign * alpha * grad_y)
                step = x_new - y
                obj_new = obj_fn(x_new)
                step_sq = float(np.dot(step, step))
                rhs = sign * (obj_y + float(np.dot(grad_y, step))) - step_sq / (
                    2.0 * alpha
                )
                if sign * obj_new >= rhs:
                    break
                alpha /= backtrack_factor
        else:
            x_new = project_fn(y + sign * alpha * grad_y)
            obj_new = obj_fn(x_new)

        if per_iter_callback is not None:
            per_iter_callback(x_new, obj_new)

        if (minimize and obj_new < best_obj) or (not minimize and obj_new > best_obj):
            best_obj = obj_new
            best_x = x_new.copy()

        if float(np.linalg.norm(x_new - x)) < tol:
            return best_x, best_obj, k + 1, True

        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x = x_new
        t = t_new

    return best_x, best_obj, max_iter, False
