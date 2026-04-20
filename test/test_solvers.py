# pyre-unsafe
"""Tests for maximin solvers."""

import numpy as np
import pytest

from maximin.confidence_regions import Ellipsoid
from maximin.decision_spaces import AllocationDecision
from maximin.outcome_models import MatrixGame
from maximin.problem_objectives import MatrixGameEllipsoidDualObjective
from maximin.solvers import ProximalSubgradientDualSolver, SolverResult


class TestSolverResult:
    """Tests for the SolverResult dataclass."""

    @staticmethod
    def test_str_converged() -> None:
        """__str__ should include 'converged' and the objective value."""
        result = SolverResult(
            x=np.array([1.0, 0.0]),
            objective=0.9,
            n_iterations=2,
            converged=True,
        )
        s = str(result)
        assert "converged" in s
        assert "0.9" in s

    @staticmethod
    def test_str_not_converged() -> None:
        """__str__ should indicate failure to converge."""
        result = SolverResult(
            x=np.zeros(2),
            objective=0.0,
            n_iterations=1000,
            converged=False,
        )
        assert "not converged" in str(result)


class TestProximalSubgradientDualSolver:
    """Tests for ProximalSubgradientDualSolver."""

    @staticmethod
    def _simple_problem() -> tuple[
        MatrixGameEllipsoidDualObjective,
        AllocationDecision,
        ProximalSubgradientDualSolver,
    ]:
        r"""Set up a 2-option game with a known analytic optimum.

        With A = I_2, beta_hat = [1, 0], Sigma = 0.01 I_2:

        .. math::

            h(c) = c_0 - 0.1\,\|c\|,

        maximized over :math:`C = \{c \ge 0, c_0 + c_1 \le 1\}` at
        :math:`c^* = [1, 0]` with :math:`h^* = 0.9`.
        """
        game = MatrixGame(np.eye(2))
        region = Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2))
        obj = MatrixGameEllipsoidDualObjective(game, region)
        space = AllocationDecision(2)
        solver = ProximalSubgradientDualSolver(obj, space, step_size=1.0)
        return obj, space, solver

    @staticmethod
    def test_result_is_feasible() -> None:
        """The returned point must lie in the decision space."""
        obj, space, solver = TestProximalSubgradientDualSolver._simple_problem()
        result = solver.solve(np.array([0.5, 0.5]))
        assert space.contains(result.x), f"result.x={result.x} not in C"

    @staticmethod
    def test_known_optimum() -> None:
        """Solver should converge to c* = [1, 0] with objective ~0.9."""
        _, _, solver = TestProximalSubgradientDualSolver._simple_problem()
        result = solver.solve(np.array([0.5, 0.5]))
        assert result.converged
        np.testing.assert_allclose(result.x, [1.0, 0.0], atol=1e-8)
        assert pytest.approx(result.objective, abs=1e-8) == 0.9

    @staticmethod
    def test_objective_equals_evaluated() -> None:
        """result.objective must equal obj.evaluate(result.x)."""
        obj, _, solver = TestProximalSubgradientDualSolver._simple_problem()
        result = solver.solve(np.array([0.3, 0.7]))
        assert pytest.approx(result.objective, abs=1e-12) == obj.evaluate(result.x)

    @staticmethod
    def test_initial_point_projected() -> None:
        """An infeasible initial point should be projected before iteration."""
        obj, space, solver = TestProximalSubgradientDualSolver._simple_problem()
        result = solver.solve(np.array([5.0, 5.0]))
        assert space.contains(result.x)

    @staticmethod
    def test_result_not_worse_than_initial() -> None:
        """Best objective seen must be >= h(projected initial point)."""
        obj, space, solver = TestProximalSubgradientDualSolver._simple_problem()
        c0 = np.array([0.5, 0.5])
        initial_obj = obj.evaluate(space.project(c0))
        result = solver.solve(c0)
        assert result.objective >= initial_obj - 1e-12


@pytest.mark.parametrize(
    "seed,m,n",
    [
        (1401, 3, 3),
        (2401, 5, 4),
        (3401, 2, 5),
    ],
)
def test_dual_solver_feasible_random(seed: int, m: int, n: int) -> None:
    """For random problems the solver's output must lie in C."""
    np.random.seed(seed)
    A = np.random.randn(m, n)
    beta_hat = np.random.randn(n)
    R = np.random.randn(n, n)
    Sigma = R.T @ R + np.eye(n)
    game = MatrixGame(A)
    region = Ellipsoid(beta_hat, Sigma)
    obj = MatrixGameEllipsoidDualObjective(game, region)
    space = AllocationDecision(m)
    solver = ProximalSubgradientDualSolver(obj, space, max_iter=500, step_size=0.5)
    c0 = np.ones(m) / m
    result = solver.solve(c0)
    assert space.contains(result.x), f"result not feasible: {result.x}"
