# pyre-unsafe
"""Tests for maximin solvers."""

import time

import numpy as np
import pytest

from maximin.confidence_regions import Ellipsoid, Hypercube
from maximin.decision_spaces import AllocationDecision
from maximin.outcome_models import MatrixGame
from maximin.problem_objectives import (
    DefaultPrimalObjective,
    MatrixGameEllipsoidDualObjective,
)
from maximin.solvers import (
    AcceleratedProximalGradientDualSolver,
    MarkowitzSolver,
    MaximinLinearSolver,
    ProximalSubgradientDualSolver,
    ProximalSubgradientPrimalSolver,
    SolverResult,
)


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

    @staticmethod
    def test_str_includes_final_gap_when_present() -> None:
        """__str__ should include final_gap when duality_gaps is set."""
        result = SolverResult(
            x=np.array([1.0, 0.0]),
            objective=0.9,
            n_iterations=2,
            converged=True,
            duality_gaps=np.array([0.5, 0.02]),
        )
        s = str(result)
        assert "final_gap" in s
        assert "0.02" in s

    @staticmethod
    def test_plot_convergence_no_gaps_raises() -> None:
        """plot_convergence must raise ValueError when no gaps were recorded."""
        result = SolverResult(
            x=np.array([1.0, 0.0]),
            objective=0.9,
            n_iterations=2,
            converged=True,
        )
        with pytest.raises(ValueError, match="duality gaps"):
            result.plot_convergence()

    @staticmethod
    def test_plot_convergence_returns_axes() -> None:
        """plot_convergence must return an Axes object when gaps are present."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gaps = np.array([0.5, 0.3, 0.1, 0.01])
        result = SolverResult(
            x=np.array([1.0, 0.0]),
            objective=0.9,
            n_iterations=4,
            converged=True,
            duality_gaps=gaps,
        )
        ax = result.plot_convergence()
        assert ax is not None
        plt.close("all")


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

            f(c) = c_0 - 0.1\,\|c\|,

        maximized over :math:`C = \{c \ge 0, c_0 + c_1 \le 1\}` at
        :math:`c^* = [1, 0]` with :math:`f^* = 0.9`.
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
        """Best objective seen must be >= f(projected initial point)."""
        obj, space, solver = TestProximalSubgradientDualSolver._simple_problem()
        c0 = np.array([0.5, 0.5])
        initial_obj = obj.evaluate(space.project(c0))
        result = solver.solve(c0)
        assert result.objective >= initial_obj - 1e-12

    @staticmethod
    def test_duality_gaps_none_by_default() -> None:
        """Without primal_objective, duality_gaps must be None."""
        _, _, solver = TestProximalSubgradientDualSolver._simple_problem()
        result = solver.solve(np.array([0.5, 0.5]))
        assert result.duality_gaps is None

    @staticmethod
    def test_duality_gaps_tracked() -> None:
        """With primal_objective, gaps are non-negative and shrink."""
        game = MatrixGame(np.eye(2))
        region = Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2))
        space = AllocationDecision(2)
        obj = MatrixGameEllipsoidDualObjective(game, region)
        primal_obj = DefaultPrimalObjective(game, space)
        solver = ProximalSubgradientDualSolver(
            obj, space, step_size=1.0, primal_objective=primal_obj
        )
        # Start far from the optimum [1, 0] so the first gap is clearly positive.
        result = solver.solve(np.array([0.1, 0.9]))
        assert result.duality_gaps is not None
        assert len(result.duality_gaps) == result.n_iterations
        assert np.all(result.duality_gaps >= -1e-10)
        assert result.duality_gaps[-1] < result.duality_gaps[0]


class TestAcceleratedProximalGradientDualSolver:
    """Tests for AcceleratedProximalGradientDualSolver."""

    @staticmethod
    def _simple_problem() -> tuple[
        MatrixGameEllipsoidDualObjective,
        AllocationDecision,
        AcceleratedProximalGradientDualSolver,
    ]:
        r"""Same 2-option known-answer problem used in other solver tests.

        With A = I_2, beta_hat = [1, 0], Sigma = 0.01 I_2:

        .. math::

            f(c) = c_0 - 0.1\,\|c\|,

        maximized over :math:`C = \{c \ge 0, c_0 + c_1 \le 1\}` at
        :math:`c^* = [1, 0]` with :math:`f^* = 0.9`.
        """
        game = MatrixGame(np.eye(2))
        region = Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2))
        obj = MatrixGameEllipsoidDualObjective(game, region)
        space = AllocationDecision(2)
        solver = AcceleratedProximalGradientDualSolver(obj, space, step_size=1.0)
        return obj, space, solver

    @staticmethod
    def test_known_optimum() -> None:
        """Solver must converge to c* = [1, 0] with objective ~0.9."""
        _, _, solver = TestAcceleratedProximalGradientDualSolver._simple_problem()
        result = solver.solve(np.array([0.5, 0.5]))
        assert result.converged
        np.testing.assert_allclose(result.x, [1.0, 0.0], atol=1e-6)
        assert pytest.approx(result.objective, abs=1e-6) == 0.9

    @staticmethod
    def test_result_is_feasible() -> None:
        """The returned point must lie in the decision space."""
        obj, space, solver = TestAcceleratedProximalGradientDualSolver._simple_problem()
        result = solver.solve(np.array([0.5, 0.5]))
        assert space.contains(result.x), f"result.x={result.x} not in C"

    @staticmethod
    def test_objective_equals_evaluated() -> None:
        """result.objective must equal obj.evaluate(result.x)."""
        obj, _, solver = TestAcceleratedProximalGradientDualSolver._simple_problem()
        result = solver.solve(np.array([0.3, 0.7]))
        assert pytest.approx(result.objective, abs=1e-12) == obj.evaluate(result.x)

    @staticmethod
    def test_initial_point_projected() -> None:
        """An infeasible initial point should be projected before iteration."""
        obj, space, solver = TestAcceleratedProximalGradientDualSolver._simple_problem()
        result = solver.solve(np.array([5.0, 5.0]))
        assert space.contains(result.x)

    @staticmethod
    def test_matches_markowitz_on_random_problem() -> None:
        r"""APG objective must be close to the global SOCP optimum.

        MatrixGame + Ellipsoid yields a differentiable dual objective, so
        APG's true-gradient steps should converge to near the optimum found
        by the exact SOCP solver.
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
        apg_result = AcceleratedProximalGradientDualSolver(
            obj, space, max_iter=5_000, step_size=0.1
        ).solve(c0)

        assert space.contains(apg_result.x), "APG result not feasible"
        assert socp_result.objective >= apg_result.objective - 1e-4

    @staticmethod
    def test_result_not_worse_than_initial() -> None:
        """Best objective seen must be >= f(projected initial point)."""
        obj, space, solver = TestAcceleratedProximalGradientDualSolver._simple_problem()
        c0 = np.array([0.5, 0.5])
        initial_obj = obj.evaluate(space.project(c0))
        result = solver.solve(c0)
        assert result.objective >= initial_obj - 1e-12

    @staticmethod
    def test_duality_gaps_tracked() -> None:
        """With primal_objective, APGD gaps are non-negative and shrink."""
        game = MatrixGame(np.eye(2))
        region = Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2))
        space = AllocationDecision(2)
        obj = MatrixGameEllipsoidDualObjective(game, region)
        primal_obj = DefaultPrimalObjective(game, space)
        solver = AcceleratedProximalGradientDualSolver(
            obj, space, step_size=1.0, primal_objective=primal_obj
        )
        # Start far from the optimum [1, 0] so the first gap is clearly positive.
        result = solver.solve(np.array([0.1, 0.9]))
        assert result.duality_gaps is not None
        assert len(result.duality_gaps) == result.n_iterations
        assert np.all(result.duality_gaps >= -1e-10)
        assert result.duality_gaps[-1] < result.duality_gaps[0]

    @staticmethod
    def test_backtracking_with_large_step_size() -> None:
        r"""Backtracking must rescue a step size that would otherwise diverge.

        step_size=100.0 far exceeds 1/L; without backtracking the iterates
        diverge.  With backtracking (backtrack_factor=2.0, the default) the
        solver must still reach within 1e-3 of the SOCP optimum.
        """
        np.random.seed(42)
        m, n = 10, 5
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
        apg_result = AcceleratedProximalGradientDualSolver(
            obj, space, max_iter=5_000, step_size=100.0, backtrack_factor=2.0
        ).solve(c0)

        assert space.contains(apg_result.x), "APG result not feasible"
        assert socp_result.objective >= apg_result.objective - 1e-3


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


class TestProximalSubgradientPrimalSolver:
    """Tests for ProximalSubgradientPrimalSolver."""

    @staticmethod
    def _simple_problem() -> tuple[
        DefaultPrimalObjective,
        Ellipsoid,
        ProximalSubgradientPrimalSolver,
    ]:
        r"""Set up a 2-option game for the primal solver.

        With A = I_2, beta_hat = [1, 0], Sigma = 0.01 I_2:
        h(beta) = max(beta_1, beta_2), minimized over the ellipsoid
        at beta* near [0.9, 0] with h* = 0.9.
        """
        game = MatrixGame(np.eye(2))
        region = Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2))
        space = AllocationDecision(2)
        primal_obj = DefaultPrimalObjective(game, space)
        solver = ProximalSubgradientPrimalSolver(primal_obj, region, step_size=0.01)
        return primal_obj, region, solver

    @staticmethod
    def test_duality_gaps_none_by_default() -> None:
        """Without dual_objective, duality_gaps must be None."""
        _, _, solver = TestProximalSubgradientPrimalSolver._simple_problem()
        result = solver.solve(np.array([1.0, 0.0]))
        assert result.duality_gaps is None

    @staticmethod
    def test_duality_gaps_tracked() -> None:
        """With dual_objective, gaps are non-negative and have correct length."""
        game = MatrixGame(np.eye(2))
        region = Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2))
        space = AllocationDecision(2)
        primal_obj = DefaultPrimalObjective(game, space, max_iter=50)
        dual_obj = MatrixGameEllipsoidDualObjective(game, region)
        solver = ProximalSubgradientPrimalSolver(
            primal_obj, region, max_iter=10, step_size=0.01, dual_objective=dual_obj
        )
        result = solver.solve(np.array([1.0, 0.0]))
        assert result.duality_gaps is not None
        assert len(result.duality_gaps) == result.n_iterations
        assert np.all(result.duality_gaps >= -1e-10)


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
        """Solver must recover c* = [1, 0] with f* = 0.9 exactly."""
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


class TestMaximinLinearSolver:
    r"""Tests for the LP-based MaximinLinearSolver.

    Known-answer problem: A = I_2, lo = [0.8, 0], hi = [1, 0.2].

    For c >= 0 the inner minimum places beta at its lower bound when the
    corresponding gradient component is non-negative:

    .. math::

        f(c) = \min_{\substack{0.8 \le \beta_1 \le 1 \\ 0 \le \beta_2 \le 0.2}}
               c_1 \beta_1 + c_2 \beta_2 = 0.8\,c_1

    so the maximum over C = {c >= 0, c_1 + c_2 <= 1} is 0.8 at c* = [1, 0].
    """

    @staticmethod
    def _simple_problem() -> tuple[MatrixGame, Hypercube, AllocationDecision]:
        return (
            MatrixGame(np.eye(2)),
            Hypercube(np.array([0.8, 0.0]), np.array([1.0, 0.2])),
            AllocationDecision(2),
        )

    @staticmethod
    def test_known_optimum() -> None:
        """Solver must recover c* = [1, 0] with f* = 0.8."""
        game, region, space = TestMaximinLinearSolver._simple_problem()
        result = MaximinLinearSolver(game, region, space).solve(np.zeros(2))
        assert result.converged
        np.testing.assert_allclose(result.x, [1.0, 0.0], atol=1e-8)
        assert pytest.approx(result.objective, abs=1e-8) == 0.8

    @staticmethod
    def test_result_feasible() -> None:
        """The returned point must lie in C."""
        game, region, space = TestMaximinLinearSolver._simple_problem()
        result = MaximinLinearSolver(game, region, space).solve(np.zeros(2))
        assert space.contains(result.x)

    @staticmethod
    def test_objective_equals_evaluated() -> None:
        """result.objective must equal min_{beta in S} c*^T A beta."""
        game, region, space = TestMaximinLinearSolver._simple_problem()
        result = MaximinLinearSolver(game, region, space).solve(np.zeros(2))
        A = game.A
        lo, hi = region.lo, region.hi
        p = A.T @ result.x
        beta_star = np.where(p >= 0, lo, hi)
        expected = float(p @ beta_star)
        assert pytest.approx(result.objective, abs=1e-10) == expected

    @staticmethod
    def test_dim_mismatch_c_raises() -> None:
        """Mismatched game.dim_c and space.dim must raise ValueError."""
        game = MatrixGame(np.eye(3))
        region = Hypercube(np.zeros(3), np.ones(3))
        space = AllocationDecision(2)
        with pytest.raises(ValueError, match="dim_c"):
            MaximinLinearSolver(game, region, space)

    @staticmethod
    def test_dim_mismatch_beta_raises() -> None:
        """Mismatched game.dim_beta and region.dim must raise ValueError."""
        game = MatrixGame(np.ones((3, 3)))
        region = Hypercube(np.zeros(2), np.ones(2))
        space = AllocationDecision(3)
        with pytest.raises(ValueError, match="dim_beta"):
            MaximinLinearSolver(game, region, space)

    @staticmethod
    def test_globally_optimal_on_random_problem() -> None:
        """LP objective must dominate all randomly sampled feasible c."""
        np.random.seed(7777)
        m, n = 6, 4
        A = np.random.randn(m, n)
        lo = -np.random.rand(n)
        hi = lo + np.random.rand(n) + 0.1
        game = MatrixGame(A)
        region = Hypercube(lo, hi)
        space = AllocationDecision(m)

        result = MaximinLinearSolver(game, region, space).solve(np.zeros(m))
        assert result.converged

        # Verify no random feasible c beats the LP optimum.
        rng = np.random.default_rng(42)
        for _ in range(200):
            raw = rng.random(m)
            c = raw / max(raw.sum(), 1.0)
            p = A.T @ c
            beta = np.where(p >= 0, lo, hi)
            f_c = float(p @ beta)
            assert f_c <= result.objective + 1e-8
