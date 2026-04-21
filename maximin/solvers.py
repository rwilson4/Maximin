# pyre-strict
"""Primal and dual solvers for maximin optimization."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import clarabel
import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.sparse as sp

from maximin._opt import _fista
from maximin.confidence_regions import ConfidenceRegion, Ellipsoid, Hypercube
from maximin.decision_spaces import AllocationDecision, DecisionSpace
from maximin.outcome_models import MatrixGame
from maximin.problem_objectives import (
    DualObjective,
    MatrixGameEllipsoidDualObjective,
    PrimalObjective,
)
from maximin.robust_constraints import MatrixGameEllipsoidRobustConstraint


@dataclass(frozen=True)
class SolverResult:
    r"""Result returned by a maximin solver.

    Parameters
    ----------
    x : npt.NDArray[np.float64]
        Optimal point found (``c`` for a DualSolver, ``beta`` for a
        PrimalSolver).
    objective : float
        Objective value at ``x``.
    n_iterations : int
        Number of gradient steps taken.
    converged : bool
        True if the step-size tolerance was reached before
        ``max_iter``.
    """

    x: npt.NDArray[np.float64]
    objective: float
    n_iterations: int
    converged: bool

    def __str__(self) -> str:
        status = "converged" if self.converged else "not converged"
        return (
            f"SolverResult({status}, "
            f"objective={self.objective:.6g}, "
            f"n_iterations={self.n_iterations})"
        )


class DualSolver(ABC):
    r"""Abstract base for solvers that maximize :math:`f(c)` over :math:`c \in C`.

    A DualSolver computes

    .. math::

        c^* = \arg\max_{c \in C}\, f(c)
            = \arg\max_{c \in C}\, \min_{\beta \in S} g(c;\, \beta).
    """

    @abstractmethod
    def solve(
        self,
        c0: npt.NDArray[np.float64],
    ) -> SolverResult:
        r"""Maximize the dual objective starting from ``c0``.

        Parameters
        ----------
        c0 : npt.NDArray[np.float64]
            Initial decision, shape ``(m,)``.

        Returns
        -------
        SolverResult
            Optimal point and convergence diagnostics.
        """


class PrimalSolver(ABC):
    r"""Abstract base for solvers that minimize :math:`h(\beta)` over :math:`\beta \in S`.

    A PrimalSolver computes

    .. math::

        \beta^* = \arg\min_{\beta \in S}\, h(\beta)
                = \arg\min_{\beta \in S}\, \max_{c \in C} g(c;\, \beta).
    """

    @abstractmethod
    def solve(
        self,
        beta0: npt.NDArray[np.float64],
    ) -> SolverResult:
        r"""Minimize the primal objective starting from ``beta0``.

        Parameters
        ----------
        beta0 : npt.NDArray[np.float64]
            Initial parameter, shape ``(n,)``.

        Returns
        -------
        SolverResult
            Optimal point and convergence diagnostics.
        """


class ProximalSubgradientDualSolver(DualSolver):
    r"""Projected subgradient ascent on the dual objective.

    Iterates

    .. math::

        c_{t+1} = \Pi_C\!\bigl(
            c_t + \alpha_t\, \nabla_c f(c_t)
        \bigr),

    with diminishing step sizes :math:`\alpha_t = \alpha_0 / \sqrt{t+1}`
    and Euclidean projection :math:`\Pi_C` onto the decision space.
    The best iterate (highest :math:`f` value seen) is returned.

    Parameters
    ----------
    objective : DualObjective
        Dual objective to maximize.
    space : DecisionSpace
        Feasible set for ``c``.
    max_iter : int
        Maximum number of gradient steps.
    tol : float
        Convergence tolerance on the step norm.
    step_size : float
        Initial step size :math:`\alpha_0`.
    """

    def __init__(
        self,
        objective: DualObjective,
        space: DecisionSpace,
        max_iter: int = 1_000,
        tol: float = 1e-6,
        step_size: float = 1e-2,
    ) -> None:
        self._objective = objective
        self._space = space
        self._max_iter = max_iter
        self._tol = tol
        self._step_size = step_size

    def solve(
        self,
        c0: npt.NDArray[np.float64],
    ) -> SolverResult:
        """Run projected subgradient ascent from ``c0``."""
        c = self._space.project(c0.copy())
        best_c = c.copy()
        best_obj = self._objective.evaluate(c)
        alpha0 = self._step_size

        for t in range(self._max_iter):
            alpha = alpha0 / math.sqrt(t + 1)
            grad = self._objective.grad_c(c)
            c_new = self._space.project(c + alpha * grad)

            obj = self._objective.evaluate(c_new)
            if obj > best_obj:
                best_obj = obj
                best_c = c_new.copy()

            if float(np.linalg.norm(c_new - c)) < self._tol:
                return SolverResult(
                    x=best_c,
                    objective=best_obj,
                    n_iterations=t + 1,
                    converged=True,
                )
            c = c_new

        return SolverResult(
            x=best_c,
            objective=best_obj,
            n_iterations=self._max_iter,
            converged=False,
        )


class ProximalSubgradientPrimalSolver(PrimalSolver):
    r"""Projected subgradient descent on the primal objective.

    Iterates

    .. math::

        \beta_{t+1} = \Pi_S\!\bigl(
            \beta_t - \alpha_t\, \nabla_\beta h(\beta_t)
        \bigr),

    with diminishing step sizes :math:`\alpha_t = \alpha_0 / \sqrt{t+1}`
    and Euclidean projection :math:`\Pi_S` onto the confidence region.
    The best iterate (lowest :math:`h` value seen) is returned.

    Parameters
    ----------
    objective : PrimalObjective
        Primal objective to minimize.
    region : ConfidenceRegion
        Feasible set for ``beta``.
    max_iter : int
        Maximum number of gradient steps.
    tol : float
        Convergence tolerance on the step norm.
    step_size : float
        Initial step size :math:`\alpha_0`.
    """

    def __init__(
        self,
        objective: PrimalObjective,
        region: ConfidenceRegion,
        max_iter: int = 1_000,
        tol: float = 1e-6,
        step_size: float = 1e-2,
    ) -> None:
        self._objective = objective
        self._region = region
        self._max_iter = max_iter
        self._tol = tol
        self._step_size = step_size

    def solve(
        self,
        beta0: npt.NDArray[np.float64],
    ) -> SolverResult:
        """Run projected subgradient descent from ``beta0``."""
        beta = self._region.project(beta0.copy())
        best_beta = beta.copy()
        best_obj = self._objective.evaluate(beta)
        alpha0 = self._step_size

        for t in range(self._max_iter):
            alpha = alpha0 / math.sqrt(t + 1)
            grad = self._objective.grad_beta(beta)
            beta_new = self._region.project(beta - alpha * grad)

            obj = self._objective.evaluate(beta_new)
            if obj < best_obj:
                best_obj = obj
                best_beta = beta_new.copy()

            if float(np.linalg.norm(beta_new - beta)) < self._tol:
                return SolverResult(
                    x=best_beta,
                    objective=best_obj,
                    n_iterations=t + 1,
                    converged=True,
                )
            beta = beta_new

        return SolverResult(
            x=best_beta,
            objective=best_obj,
            n_iterations=self._max_iter,
            converged=False,
        )


class AcceleratedProximalGradientDualSolver(DualSolver):
    r"""Nesterov-accelerated projected gradient ascent on the dual objective.

    Assumes :math:`f(c) = \min_{\beta \in S} g(c;\beta)` is differentiable,
    which holds whenever the infimum over :math:`\beta` is uniquely attained
    for every :math:`c` (e.g. any MatrixGame paired with an Ellipsoid). In
    that case the envelope theorem (Danskin's theorem) gives
    :math:`\nabla_c f(c) = \nabla_c g(c;\beta^*(c))`, which the objective's
    ``grad_c`` method should return.

    Iterates the FISTA rule (Beck & Teboulle, 2009):

    .. math::

        c_{k+1} &= \Pi_C\!\bigl(y_k + \alpha\,\nabla f(y_k)\bigr), \\
        t_{k+1} &= \tfrac{1 + \sqrt{1 + 4t_k^2}}{2}, \\
        y_{k+1} &= c_{k+1} + \tfrac{t_k - 1}{t_{k+1}}\,
                   (c_{k+1} - c_k),

    with constant step size :math:`\alpha \le 1/L` where :math:`L` is the
    Lipschitz constant of :math:`\nabla f`. Convergence is
    :math:`O(1/k^2)`, faster than the :math:`O(1/\sqrt{k})` rate of
    projected subgradient. The best iterate (highest :math:`f` value seen)
    is returned to guard against non-monotone steps caused by momentum.

    Parameters
    ----------
    objective : DualObjective
        Dual objective to maximize. Its ``grad_c`` must return the true
        gradient (not a subgradient).
    space : DecisionSpace
        Feasible set for ``c``.
    max_iter : int
        Maximum number of gradient steps.
    tol : float
        Convergence tolerance on the iterate-change norm.
    step_size : float
        Constant step size :math:`\alpha`. Must satisfy
        :math:`\alpha \le 1/L`; too large a value causes divergence.
    """

    def __init__(
        self,
        objective: DualObjective,
        space: DecisionSpace,
        max_iter: int = 1_000,
        tol: float = 1e-6,
        step_size: float = 1e-2,
    ) -> None:
        self._objective = objective
        self._space = space
        self._max_iter = max_iter
        self._tol = tol
        self._step_size = step_size

    def solve(
        self,
        c0: npt.NDArray[np.float64],
    ) -> SolverResult:
        """Run accelerated projected gradient ascent from ``c0``."""
        x0 = self._space.project(c0.copy())
        best_c, best_obj, n_iters, converged = _fista(
            grad_fn=self._objective.grad_c,
            obj_fn=self._objective.evaluate,
            project_fn=self._space.project,
            x0=x0,
            step_size=self._step_size,
            max_iter=self._max_iter,
            tol=self._tol,
            minimize=False,
        )
        return SolverResult(
            x=best_c,
            objective=best_obj,
            n_iterations=n_iters,
            converged=converged,
        )


def _build_markowitz_base_socp(
    game: MatrixGame,
    region: Ellipsoid,
) -> tuple[
    sp.csc_matrix,
    npt.NDArray[np.float64],
    sp.csc_matrix,
    npt.NDArray[np.float64],
]:
    r"""Assemble the base Clarabel SOCP matrices for the Markowitz problem.

    Variables: :math:`x = [c \;;\; t] \in \mathbb{R}^{m+1}`.

    Constraint layout (rows of ``A``):

    .. code-block:: text

        rows  0..n    SOC_{n+1}:  [t ; L^T A^T c]
        rows n+1..n+m  Nonneg_m:  c >= 0
        row  n+m+1     Nonneg_1:  sum(c) <= 1

    Returns
    -------
    tuple
        ``(P, q, A, b)``.
    """
    A_mat = game.A  # (m, n)
    beta_hat = region.beta_hat  # (n,)
    Sigma = region.Sigma  # (n, n)
    m, n = A_mat.shape

    L = np.linalg.cholesky(Sigma)  # Sigma = L L^T
    LT_AT = L.T @ A_mat.T  # (n, m)

    P = sp.csc_matrix((m + 1, m + 1))

    q = np.empty(m + 1)
    q[:m] = -(A_mat @ beta_hat)
    q[m] = 1.0

    # SOC block: [0..0 -1 ; -L^T A^T  0]
    soc_row0 = sp.hstack(
        [sp.csc_matrix((1, m)), sp.csc_matrix([[-1.0]])],
        format="csc",
    )
    soc_body = sp.hstack(
        [sp.csc_matrix(-LT_AT), sp.csc_matrix((n, 1))],
        format="csc",
    )
    A_soc = sp.vstack([soc_row0, soc_body], format="csc")  # (n+1) x (m+1)

    # Non-negativity block
    A_nn = sp.hstack(
        [-sp.eye(m, format="csc"), sp.csc_matrix((m, 1))],
        format="csc",
    )  # m x (m+1)

    # Budget block
    ones_row = sp.csc_matrix(
        (np.ones(m), (np.zeros(m, dtype=int), np.arange(m))),
        shape=(1, m),
    )
    A_bud = sp.hstack(
        [ones_row, sp.csc_matrix((1, 1))],
        format="csc",
    )  # 1 x (m+1)

    A_total = sp.vstack([A_soc, A_nn, A_bud], format="csc")
    b_total = np.zeros(n + 1 + m + 1)
    b_total[-1] = 1.0  # budget RHS

    return P, q, A_total, b_total


class MarkowitzSolver(DualSolver):
    r"""Exact SOCP solver for the MatrixGame--Ellipsoid maximin problem.

    Maximizes the dual objective

    .. math::

        f(c) = c^\top A \hat\beta - \bigl\| \Sigma^{1/2} A^\top c \bigr\|_2

    over the :class:`~maximin.decision_spaces.AllocationDecision` space
    :math:`C = \{c \ge 0,\, \sum_i c_i \le 1\}` by solving the
    equivalent second-order cone program:

    .. math::

        \begin{aligned}
        \min_{c,\, t}\quad & -c^\top A \hat\beta + t \\
        \text{s.t.}\quad
            & \bigl\| L^\top A^\top c \bigr\|_2 \le t \\
            & c \ge 0,\quad \textstyle\sum_i c_i \le 1
        \end{aligned}

    where :math:`L` is the lower Cholesky factor of :math:`\Sigma`
    (:math:`\Sigma = L L^\top`). The problem is solved globally and to
    high precision by Clarabel's interior-point method.

    The name reflects the structural similarity to Markowitz
    mean--variance portfolio optimization, where the payoff
    :math:`A\hat\beta` plays the role of expected returns and
    :math:`\Sigma` the covariance.

    Parameters
    ----------
    game : MatrixGame
        Bilinear outcome model with payoff matrix ``A``, shape
        ``(m, n)``.
    region : Ellipsoid
        Ellipsoidal uncertainty set parameterized by ``beta_hat``
        and ``Sigma``.
    space : AllocationDecision
        Feasible set for the decision ``c``.
    """

    def __init__(
        self,
        game: MatrixGame,
        region: Ellipsoid,
        space: AllocationDecision,
    ) -> None:
        if game.dim_c != space.dim:
            raise ValueError(
                f"game.dim_c ({game.dim_c}) must equal space.dim ({space.dim})"
            )
        if game.dim_beta != region.dim:
            raise ValueError(
                f"game.dim_beta ({game.dim_beta}) must equal "
                f"region.dim ({region.dim})"
            )
        self._game = game
        self._region = region
        self._space = space
        self._objective = MatrixGameEllipsoidDualObjective(game, region)

    def _build_socp(
        self,
    ) -> tuple[
        sp.csc_matrix,
        npt.NDArray[np.float64],
        sp.csc_matrix,
        npt.NDArray[np.float64],
    ]:
        """Assemble the Clarabel SOCP matrices; delegates to the module helper."""
        return _build_markowitz_base_socp(self._game, self._region)

    def solve(
        self,
        c0: npt.NDArray[np.float64],
    ) -> SolverResult:
        r"""Solve the SOCP to global optimality via Clarabel.

        Parameters
        ----------
        c0 : npt.NDArray[np.float64]
            Ignored; provided for API compatibility with
            :class:`DualSolver`. Clarabel does not require an initial
            point.

        Returns
        -------
        SolverResult
            Optimal decision ``c*``, objective value ``f(c*)``,
            interior-point iteration count, and convergence flag.
        """
        m = self._game.dim_c
        n = self._game.dim_beta

        P, q, A_total, b_total = self._build_socp()

        cones: list[object] = [
            clarabel.SecondOrderConeT(n + 1),
            clarabel.NonnegativeConeT(m + 1),
        ]

        settings = clarabel.DefaultSettings()
        settings.verbose = False

        solver = clarabel.DefaultSolver(P, q, A_total, b_total, cones, settings)
        solution = solver.solve()

        c_star = self._space.project(np.array(solution.x[:m]))
        converged = solution.status in (
            clarabel.SolverStatus.Solved,
            clarabel.SolverStatus.AlmostSolved,
        )

        return SolverResult(
            x=c_star,
            objective=self._objective.evaluate(c_star),
            n_iterations=solution.iterations,
            converged=converged,
        )


class ConstrainedMarkowitzSolver(DualSolver):
    r"""Exact SOCP solver for the Markowitz problem with robust constraints.

    Maximizes

    .. math::

        f(c) = c^\top A \hat\beta - \bigl\| \Sigma^{1/2} A^\top c \bigr\|_2

    over :math:`C = \{c \ge 0,\, \sum_i c_i \le 1\}` subject to a list of
    robust constraints

    .. math::

        q_k(c) = c^\top B_k \hat\gamma_k
                 - \bigl\| \Sigma_{T_k}^{1/2} B_k^\top c \bigr\|_2
                 \ge 0, \quad k = 1, \ldots, K.

    Each constraint contributes one additional
    :class:`clarabel.SecondOrderConeT` block to the SOCP solved by
    Clarabel's interior-point method.

    Parameters
    ----------
    game : MatrixGame
        Bilinear outcome model with payoff matrix ``A``, shape ``(m, n)``.
    region : Ellipsoid
        Ellipsoidal uncertainty set for the objective parameter ``beta``.
    space : AllocationDecision
        Feasible set for the decision ``c``.
    constraints : list[MatrixGameEllipsoidRobustConstraint]
        Robust constraints, each with ``dim_c == game.dim_c``.
    """

    def __init__(
        self,
        game: MatrixGame,
        region: Ellipsoid,
        space: AllocationDecision,
        constraints: list[MatrixGameEllipsoidRobustConstraint],
    ) -> None:
        if game.dim_c != space.dim:
            raise ValueError(
                f"game.dim_c ({game.dim_c}) must equal space.dim ({space.dim})"
            )
        if game.dim_beta != region.dim:
            raise ValueError(
                f"game.dim_beta ({game.dim_beta}) must equal "
                f"region.dim ({region.dim})"
            )
        for rc in constraints:
            if rc.dim_c != game.dim_c:
                raise ValueError(
                    f"constraint dim_c ({rc.dim_c}) must equal "
                    f"game.dim_c ({game.dim_c})"
                )
        self._game = game
        self._region = region
        self._space = space
        self._constraints = constraints
        self._objective = MatrixGameEllipsoidDualObjective(game, region)

    def _build_socp(
        self,
    ) -> tuple[
        sp.csc_matrix,
        npt.NDArray[np.float64],
        sp.csc_matrix,
        npt.NDArray[np.float64],
        list[object],
    ]:
        """Assemble SOCP matrices including all robust constraint blocks."""
        m = self._game.dim_c
        n = self._game.dim_beta

        P, q, A_base, b_base = _build_markowitz_base_socp(self._game, self._region)
        cones: list[object] = [
            clarabel.SecondOrderConeT(n + 1),
            clarabel.NonnegativeConeT(m + 1),
        ]

        A_blocks = [A_base]
        b_parts = [b_base]
        for rc in self._constraints:
            A_block, cone_size = rc.socp_block(m)
            A_blocks.append(A_block)
            b_parts.append(np.zeros(cone_size))
            cones.append(clarabel.SecondOrderConeT(cone_size))

        A_total = sp.vstack(A_blocks, format="csc")
        b_total = np.concatenate(b_parts)
        return P, q, A_total, b_total, cones

    def solve(
        self,
        c0: npt.NDArray[np.float64],
    ) -> SolverResult:
        r"""Solve the constrained SOCP to global optimality via Clarabel.

        Parameters
        ----------
        c0 : npt.NDArray[np.float64]
            Ignored; provided for API compatibility.

        Returns
        -------
        SolverResult
            Optimal decision ``c*``, objective value ``f(c*)``,
            interior-point iteration count, and convergence flag.
        """
        m = self._game.dim_c

        P, q, A_total, b_total, cones = self._build_socp()

        settings = clarabel.DefaultSettings()
        settings.verbose = False

        solver = clarabel.DefaultSolver(P, q, A_total, b_total, cones, settings)
        solution = solver.solve()

        c_star = self._space.project(np.array(solution.x[:m]))
        converged = solution.status in (
            clarabel.SolverStatus.Solved,
            clarabel.SolverStatus.AlmostSolved,
        )

        return SolverResult(
            x=c_star,
            objective=self._objective.evaluate(c_star),
            n_iterations=solution.iterations,
            converged=converged,
        )


class MaximinLinearSolver(DualSolver):
    r"""Exact LP solver for the MatrixGame--Hypercube maximin problem.

    Solves

    .. math::

        \max_{c \in C}\, \min_{\beta \in S}\, c^\top A \beta

    where :math:`C = \{c \ge 0,\, \sum_i c_i \le 1\}` and
    :math:`S = \{\beta : \ell \le \beta \le u\}` by reformulating via LP
    duality on the inner minimization:

    .. math::

        \begin{aligned}
        \max_{c,\, \mu_\ell,\, \mu_u \ge 0}\quad & \ell^\top \mu_\ell - u^\top \mu_u \\
        \text{s.t.}\quad
            & \mu_\ell - \mu_u = A^\top c \\
            & c \ge 0,\quad \textstyle\sum_i c_i \le 1
        \end{aligned}

    Parameters
    ----------
    game : MatrixGame
        Bilinear outcome model with payoff matrix ``A``, shape ``(m, n)``.
    region : Hypercube
        Box uncertainty set with lower bounds ``lo`` and upper bounds ``hi``.
    space : AllocationDecision
        Feasible set for the decision ``c``.
    """

    def __init__(
        self,
        game: MatrixGame,
        region: Hypercube,
        space: AllocationDecision,
    ) -> None:
        if game.dim_c != space.dim:
            raise ValueError(
                f"game.dim_c ({game.dim_c}) must equal space.dim ({space.dim})"
            )
        if game.dim_beta != region.dim:
            raise ValueError(
                f"game.dim_beta ({game.dim_beta}) must equal "
                f"region.dim ({region.dim})"
            )
        self._game = game
        self._region = region
        self._space = space

    def solve(
        self,
        c0: npt.NDArray[np.float64],
    ) -> SolverResult:
        r"""Solve the LP to global optimality via HiGHS.

        Parameters
        ----------
        c0 : npt.NDArray[np.float64]
            Ignored; provided for API compatibility with
            :class:`DualSolver`.

        Returns
        -------
        SolverResult
            Optimal decision ``c*``, objective value ``f(c*)``,
            LP simplex iteration count, and convergence flag.
        """
        A = self._game.A  # (m, n)
        lo = self._region.lo  # (n,)
        hi = self._region.hi  # (n,)
        m, n = A.shape

        # Variables: x = [c (m), mu_lo (n), mu_hi (n)]
        # Maximize lo^T mu_lo - hi^T mu_hi  =>  minimize -lo^T mu_lo + hi^T mu_hi
        c_obj = np.concatenate([np.zeros(m), -lo, hi])

        # Equality: mu_lo - mu_hi = A^T c  =>  [-A^T | I | -I] x = 0
        A_eq = np.hstack([-A.T, np.eye(n), -np.eye(n)])
        b_eq = np.zeros(n)

        # Inequality: sum(c) <= 1
        A_ub = np.concatenate([np.ones(m), np.zeros(2 * n)])[np.newaxis, :]
        b_ub = np.array([1.0])

        bounds = [(0.0, None)] * (m + 2 * n)

        result = scipy.optimize.linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        c_star = self._space.project(np.array(result.x[:m]))

        # Compute f(c*) = min_{beta in S} (A^T c*)^T beta directly.
        p = A.T @ c_star  # (n,)
        beta_star = np.where(p >= 0, lo, hi)
        objective = float(p @ beta_star)

        return SolverResult(
            x=c_star,
            objective=objective,
            n_iterations=result.nit,
            converged=(result.status == 0),
        )
