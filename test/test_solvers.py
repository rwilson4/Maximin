# pyre-unsafe
"""Tests for maximin solvers."""

import time

import numpy as np
import pytest

from maximin.confidence_regions import Ellipsoid
from maximin.decision_spaces import AllocationDecision
from maximin.outcome_models import MatrixGame
from maximin.problem_objectives import MatrixGameEllipsoidDualObjective
from maximin.solvers import MarkowitzSolver, ProximalSubgradientDualSolver, SolverResult


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


class TestMarkowitzSolver:
    """Tests for the SOCP-based MarkowitzSolver."""

    @staticmethod
    def _simple_problem() -> tuple[MatrixGame, Ellipsoid, AllocationDecision]:
        """Return the same 2-option known-answer problem used elsewhere."""
        return (
            MatrixGame(np.eye(2)),
            Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2)),
            AllocationDecision(2),
        )

    @staticmethod
    def test_known_optimum() -> None:
        """Solver must recover c* = [1, 0] with h* = 0.9 exactly."""
        game, region, space = TestMarkowitzSolver._simple_problem()
        result = MarkowitzSolver(game, region, space).solve(np.zeros(2))
        assert result.converged
        np.testing.assert_allclose(result.x, [1.0, 0.0], atol=1e-6)
        assert pytest.approx(result.objective, abs=1e-6) == 0.9

    @staticmethod
    def test_result_feasible() -> None:
        """The returned point must lie in C."""
        game, region, space = TestMarkowitzSolver._simple_problem()
        result = MarkowitzSolver(game, region, space).solve(np.zeros(2))
        assert space.contains(result.x)

    @staticmethod
    def test_objective_equals_evaluated() -> None:
        """result.objective must equal obj.evaluate(result.x)."""
        game, region, space = TestMarkowitzSolver._simple_problem()
        obj = MatrixGameEllipsoidDualObjective(game, region)
        result = MarkowitzSolver(game, region, space).solve(np.zeros(2))
        assert pytest.approx(result.objective, abs=1e-10) == obj.evaluate(result.x)

    @staticmethod
    def test_dim_mismatch_raises() -> None:
        """Inconsistent dimensions must raise ValueError at construction."""
        game = MatrixGame(np.eye(3))  # dim_c=3, dim_beta=3
        region = Ellipsoid(np.zeros(3), np.eye(3))
        space = AllocationDecision(2)  # dim=2 ≠ dim_c=3
        with pytest.raises(ValueError, match="dim_c"):
            MarkowitzSolver(game, region, space)

    @staticmethod
    def test_dominates_subgradient_on_random_problem() -> None:
        r"""SOCP objective must be >= subgradient best on a random instance.

        The SOCP finds the global optimum, so no gradient method can
        exceed it (up to numerical tolerance).
        """
        np.random.seed(999)
        m, n = 20, 10
        A = np.random.randn(m, n)
        beta_hat = np.random.randn(n)
        R = np.random.randn(n, n)
        Sigma = R.T @ R + np.eye(n)
        game = MatrixGame(A)
        region = Ellipsoid(beta_hat, Sigma)
        space = AllocationDecision(m)
        obj = MatrixGameEllipsoidDualObjective(game, region)

        c0 = np.ones(m) / m
        socp_result = MarkowitzSolver(game, region, space).solve(c0)
        sg_result = ProximalSubgradientDualSolver(
            obj, space, max_iter=5_000, step_size=0.5
        ).solve(c0)

        assert socp_result.objective >= sg_result.objective - 1e-4


@pytest.mark.parametrize(
    "seed,m,n,sg_iters",
    [
        (42, 10, 5, 2_000),
        (42, 50, 20, 5_000),
        (42, 100, 50, 10_000),
    ],
)
def test_markowitz_solver_benchmark(
    seed: int, m: int, n: int, sg_iters: int, capsys
) -> None:
    r"""Benchmark MarkowitzSolver against ProximalSubgradientDualSolver.

    Compares wall-clock time and objective quality across three problem
    sizes. Run with ``pytest -s`` to see the timing table.
    """
    np.random.seed(seed)
    A = np.random.randn(m, n)
    beta_hat = np.random.randn(n)
    R = np.random.randn(n, n)
    Sigma = R.T @ R + np.eye(n)
    game = MatrixGame(A)
    region = Ellipsoid(beta_hat, Sigma)
    space = AllocationDecision(m)
    obj = MatrixGameEllipsoidDualObjective(game, region)
    c0 = np.ones(m) / m

    t0 = time.perf_counter()
    socp = MarkowitzSolver(game, region, space).solve(c0)
    t_socp = time.perf_counter() - t0

    t0 = time.perf_counter()
    sg = ProximalSubgradientDualSolver(
        obj, space, max_iter=sg_iters, step_size=0.5
    ).solve(c0)
    t_sg = time.perf_counter() - t0

    with capsys.disabled():
        print(
            f"\nm={m:>3d}, n={n:>3d} | "
            f"SOCP  {t_socp*1e3:7.1f} ms  obj={socp.objective:+.6f}  "
            f"iters={socp.n_iterations:>3d} | "
            f"SubGrad {t_sg*1e3:7.1f} ms  obj={sg.objective:+.6f}  "
            f"iters={sg.n_iterations:>6d}"
        )

    assert space.contains(socp.x), "SOCP result not feasible"
    assert socp.converged, "SOCP did not converge"
    # SOCP is globally optimal; subgradient cannot exceed it.
    assert socp.objective >= sg.objective - 1e-4
