# pyre-strict
"""Primal and dual solvers for maximin optimization."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import clarabel
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from maximin.confidence_regions import ConfidenceRegion, Ellipsoid
from maximin.decision_spaces import AllocationDecision, DecisionSpace
from maximin.outcome_models import MatrixGame
from maximin.problem_objectives import (
    DualObjective,
    MatrixGameEllipsoidDualObjective,
    PrimalObjective,
)


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
        r"""Assemble the Clarabel SOCP matrices.

        Returns
        -------
        tuple
            ``(P, q, A, b)`` where ``P`` is the (zero) quadratic cost,
            ``q`` the linear cost, ``A`` the constraint matrix, and
            ``b`` the right-hand side.

        Notes
        -----
        Variables: :math:`x = [c \;;\; t] \in \mathbb{R}^{m+1}`.

        Constraint layout (rows of ``A``):

        .. code-block:: text

            rows  0..n    SOC_{n+1}:  [t ; L^T A^T c]
            rows n+1..n+m  Nonneg_m:  c >= 0
            row  n+m+1     Nonneg_1:  sum(c) <= 1
        """
        A_mat = self._game.A  # (m, n)
        beta_hat = self._region.beta_hat  # (n,)
        Sigma = self._region.Sigma  # (n, n)
        m, n = A_mat.shape

        L = np.linalg.cholesky(Sigma)  # Sigma = L L^T
        LT_AT = L.T @ A_mat.T  # (n, m)

        # Quadratic term: none.
        P = sp.csc_matrix((m + 1, m + 1))

        # Linear cost: minimize -c^T A beta_hat + t.
        q = np.empty(m + 1)
        q[:m] = -(A_mat @ beta_hat)
        q[m] = 1.0

        # SOC block ─────────────────────────────────────────────────────
        # row 0:    [0 … 0  -1]   (scalar t component)
        # rows 1..n: [-L^T A^T  0]  (vector component)
        soc_row0 = sp.hstack(
            [sp.csc_matrix((1, m)), sp.csc_matrix([[-1.0]])],
            format="csc",
        )
        soc_body = sp.hstack(
            [sp.csc_matrix(-LT_AT), sp.csc_matrix((n, 1))],
            format="csc",
        )
        A_soc = sp.vstack([soc_row0, soc_body], format="csc")  # (n+1) x (m+1)

        # Non-negativity block: [-I_m  0] ──────────────────────────────
        A_nn = sp.hstack(
            [-sp.eye(m, format="csc"), sp.csc_matrix((m, 1))],
            format="csc",
        )  # m x (m+1)

        # Budget block: [1 … 1  0] ─────────────────────────────────────
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

        cones: list[clarabel.SupportedConeT] = [
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
