# pyre-strict
"""Primal and dual solvers for maximin optimization."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from maximin.confidence_regions import ConfidenceRegion
from maximin.decision_spaces import DecisionSpace
from maximin.problem_objectives import DualObjective, PrimalObjective


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
    r"""Abstract base for solvers that maximize :math:`h(c)` over :math:`c \in C`.

    A DualSolver computes

    .. math::

        c^* = \arg\max_{c \in C}\, h(c)
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
    r"""Abstract base for solvers that minimize :math:`f(\beta)` over :math:`\beta \in S`.

    A PrimalSolver computes

    .. math::

        \beta^* = \arg\min_{\beta \in S}\, f(\beta)
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
            c_t + \alpha_t\, \nabla_c h(c_t)
        \bigr),

    with diminishing step sizes :math:`\alpha_t = \alpha_0 / \sqrt{t+1}`
    and Euclidean projection :math:`\Pi_C` onto the decision space.
    The best iterate (highest :math:`h` value seen) is returned.

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
            \beta_t - \alpha_t\, \nabla_\beta f(\beta_t)
        \bigr),

    with diminishing step sizes :math:`\alpha_t = \alpha_0 / \sqrt{t+1}`
    and Euclidean projection :math:`\Pi_S` onto the confidence region.
    The best iterate (lowest :math:`f` value seen) is returned.

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
