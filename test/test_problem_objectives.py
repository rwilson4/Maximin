# pyre-unsafe
"""Tests for dual and primal objectives."""

import time

import numpy as np
import pytest

from maximin.confidence_regions import Ellipsoid
from maximin.decision_spaces import AllocationDecision
from maximin.outcome_models import MatrixGame
from maximin.problem_objectives import (
    DefaultDualObjective,
    DefaultPrimalObjective,
    MatrixGameEllipsoidDualObjective,
)
from maximin.solvers import (
    AcceleratedProximalGradientDualSolver,
    MarkowitzSolver,
    ProximalSubgradientPrimalSolver,
)


class TestMatrixGameEllipsoidDualObjective:
    """Tests for the analytic dual objective f(c) = c^T A b_hat - ||Sigma^{1/2} A^T c||."""

    @staticmethod
    def _make_problem(
        seed: int = 0,
    ) -> tuple[MatrixGame, Ellipsoid, MatrixGameEllipsoidDualObjective]:
        """Return a small 2x2 game/region/objective for shared tests."""
        np.random.seed(seed)
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        beta_hat = np.array([1.0, 0.0])
        Sigma = 0.01 * np.eye(2)
        game = MatrixGame(A)
        region = Ellipsoid(beta_hat, Sigma)
        obj = MatrixGameEllipsoidDualObjective(game, region)
        return game, region, obj

    @staticmethod
    def test_evaluate_matches_formula() -> None:
        """f(c) = c^T A b_hat - ||Sigma^{1/2} A^T c|| computed directly."""
        _, _, obj = TestMatrixGameEllipsoidDualObjective._make_problem()
        # A = I, beta_hat = [1, 0], Sigma = 0.01 I
        # f([1, 0]) = 1 - sqrt(0.01) = 1 - 0.1 = 0.9
        c = np.array([1.0, 0.0])
        assert pytest.approx(obj.evaluate(c), abs=1e-12) == 0.9

    @staticmethod
    def test_evaluate_zero_action() -> None:
        """f(0) = 0 since both terms vanish."""
        _, _, obj = TestMatrixGameEllipsoidDualObjective._make_problem()
        assert pytest.approx(obj.evaluate(np.zeros(2)), abs=1e-12) == 0.0

    @staticmethod
    def test_minimizer_lies_in_ellipsoid() -> None:
        """beta*(c) must always be in S."""
        _, region, obj = TestMatrixGameEllipsoidDualObjective._make_problem()
        for c in [
            np.array([1.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([0.3, 0.7]),
        ]:
            assert region.contains(obj.minimizer(c)), f"minimizer outside S for c={c}"

    @staticmethod
    def test_minimizer_achieves_dual_value() -> None:
        """g(c, beta*(c)) == f(c): the minimizer achieves the dual objective."""
        game, _, obj = TestMatrixGameEllipsoidDualObjective._make_problem()
        c = np.array([0.7, 0.3])
        beta_star = obj.minimizer(c)
        assert pytest.approx(game.evaluate(c, beta_star), abs=1e-10) == obj.evaluate(c)

    @staticmethod
    def test_minimizer_is_worst_case() -> None:
        """g(c, beta*(c)) <= g(c, beta) for any other beta in S."""
        game, region, obj = TestMatrixGameEllipsoidDualObjective._make_problem()
        np.random.seed(42)
        c = np.array([0.6, 0.4])
        beta_star = obj.minimizer(c)
        g_min = game.evaluate(c, beta_star)
        # Sample random points in the ellipsoid and verify none is worse.
        for _ in range(50):
            beta = region.project(np.random.randn(2) * 3.0)
            assert game.evaluate(c, beta) >= g_min - 1e-10

    @staticmethod
    def test_grad_c_equals_A_beta_star() -> None:
        """By the envelope theorem, grad_c f = A @ beta*(c)."""
        game, _, obj = TestMatrixGameEllipsoidDualObjective._make_problem()
        c = np.array([0.6, 0.4])
        expected = game.A @ obj.minimizer(c)
        np.testing.assert_allclose(obj.grad_c(c), expected, atol=1e-12)

    @staticmethod
    def test_zero_direction_gradient() -> None:
        """When A^T c = 0, beta*(c) = beta_hat and grad = A @ beta_hat."""
        # With A = [[1, -1], [0, 0]] and c = [0.5, 0.5], A^T c = [0.5-0.5, 0] = [0, 0].
        A = np.array([[1.0, 0.0], [-1.0, 0.0]])
        beta_hat = np.array([2.0, 1.0])
        Sigma = np.eye(2)
        game = MatrixGame(A)
        region = Ellipsoid(beta_hat, Sigma)
        obj = MatrixGameEllipsoidDualObjective(game, region)
        c = np.array([0.5, 0.5])
        # A^T c = [1*0.5 + (-1)*0.5, 0*0.5 + 0*0.5] = [0, 0]
        np.testing.assert_allclose(obj.minimizer(c), beta_hat, atol=1e-12)
        np.testing.assert_allclose(obj.grad_c(c), A @ beta_hat, atol=1e-12)

    @staticmethod
    def test_dimension_mismatch_raises() -> None:
        """Mismatched game/region dimensions should raise ValueError."""
        game = MatrixGame(np.eye(3))  # dim_beta = 3
        region = Ellipsoid(np.zeros(2), np.eye(2))  # dim = 2
        with pytest.raises(ValueError, match="dim_beta"):
            MatrixGameEllipsoidDualObjective(game, region)


@pytest.mark.parametrize(
    "seed,m,n",
    [
        (1301, 2, 2),
        (2301, 3, 4),
        (3301, 5, 3),
        (4301, 4, 4),
    ],
)
def test_dual_objective_grad_finite_difference(seed: int, m: int, n: int) -> None:
    """Finite-difference check of grad_c f against the analytic gradient."""
    np.random.seed(seed)
    A = np.random.randn(m, n)
    beta_hat = np.random.randn(n)
    R = np.random.randn(n, n)
    Sigma = 0.1 * (R.T @ R + np.eye(n))
    game = MatrixGame(A)
    region = Ellipsoid(beta_hat, Sigma)
    obj = MatrixGameEllipsoidDualObjective(game, region)

    # Random point in the interior of the simplex-like region.
    c = np.abs(np.random.randn(m))
    c /= np.sum(c) * 2.0  # ensure A^T c != 0 almost surely

    eps = 1e-7
    grad_fd = np.zeros(m)
    for i in range(m):
        c_plus = c.copy()
        c_plus[i] += eps
        c_minus = c.copy()
        c_minus[i] -= eps
        grad_fd[i] = (obj.evaluate(c_plus) - obj.evaluate(c_minus)) / (2 * eps)

    np.testing.assert_allclose(grad_fd, obj.grad_c(c), rtol=1e-4)


@pytest.mark.parametrize(
    "seed,m,n",
    [
        (1302, 2, 2),
        (2302, 4, 3),
        (3302, 3, 5),
    ],
)
def test_dual_objective_minimizer_always_feasible(seed: int, m: int, n: int) -> None:
    """beta*(c) lies in S for any c."""
    np.random.seed(seed)
    A = np.random.randn(m, n)
    beta_hat = np.random.randn(n)
    R = np.random.randn(n, n)
    Sigma = R.T @ R + np.eye(n)
    game = MatrixGame(A)
    region = Ellipsoid(beta_hat, Sigma)
    obj = MatrixGameEllipsoidDualObjective(game, region)

    for _ in range(20):
        c = np.abs(np.random.randn(m))
        c /= np.sum(c)
        assert region.contains(obj.minimizer(c))


class TestDefaultDualObjective:
    r"""Tests for DefaultDualObjective (inner-APG dual objective).

    Uses a 2-option identity game with a small ellipsoid where the
    analytic answer is known:

    .. math::

        f(c) = c_0 - 0.1\,\|c\|,

    maximized over :math:`C` at :math:`c^* = [1, 0]` with
    :math:`f^* = 0.9`.
    """

    @staticmethod
    def _make_problem(
        step_size: float = 0.5,
        max_iter: int = 1_000,
    ) -> tuple[
        MatrixGame,
        Ellipsoid,
        MatrixGameEllipsoidDualObjective,
        DefaultDualObjective,
    ]:
        game = MatrixGame(np.eye(2))
        region = Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2))
        analytic = MatrixGameEllipsoidDualObjective(game, region)
        default = DefaultDualObjective(
            game, region, max_iter=max_iter, step_size=step_size
        )
        return game, region, analytic, default

    @staticmethod
    def test_evaluate_matches_analytic() -> None:
        """DefaultDualObjective.evaluate should agree with the analytic formula."""
        _, _, analytic, default = TestDefaultDualObjective._make_problem()
        for c in [
            np.array([1.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([0.3, 0.7]),
        ]:
            assert pytest.approx(default.evaluate(c), abs=1e-3) == analytic.evaluate(c)

    @staticmethod
    def test_minimizer_in_region() -> None:
        """The APG minimizer must lie in S."""
        _, region, _, default = TestDefaultDualObjective._make_problem()
        for c in [np.array([1.0, 0.0]), np.array([0.5, 0.5]), np.array([0.3, 0.7])]:
            assert region.contains(default.minimizer(c)), f"minimizer outside S for c={c}"

    @staticmethod
    def test_evaluate_equals_g_at_minimizer() -> None:
        """evaluate(c) must equal g(c, minimizer(c))."""
        game, _, _, default = TestDefaultDualObjective._make_problem()
        c = np.array([0.7, 0.3])
        beta_star = default.minimizer(c)
        assert pytest.approx(default.evaluate(c), abs=1e-12) == game.evaluate(c, beta_star)

    @staticmethod
    def test_grad_c_by_envelope() -> None:
        """grad_c(c) must equal model.grad_c(c, minimizer(c))."""
        game, _, _, default = TestDefaultDualObjective._make_problem()
        c = np.array([0.6, 0.4])
        beta_star = default.minimizer(c)
        np.testing.assert_allclose(
            default.grad_c(c), game.grad_c(c, beta_star), atol=1e-12
        )

    @staticmethod
    def test_dimension_mismatch_raises() -> None:
        """Mismatched model/region dimensions must raise ValueError."""
        game = MatrixGame(np.eye(3))
        region = Ellipsoid(np.zeros(2), np.eye(2))
        with pytest.raises(ValueError, match="dim_beta"):
            DefaultDualObjective(game, region)

    @staticmethod
    def test_performance_comparison(capsys) -> None:
        r"""DefaultDualObjective vs analytic: accuracy and wall-clock timing.

        Both objectives are used inside AcceleratedProximalGradientDualSolver
        to maximize the outer dual :math:`\max_{c \in C} f(c)`.  The SOCP
        (MarkowitzSolver) provides a global upper bound.  The test verifies
        that DefaultDualObjective finds a solution within 1e-3 of the SOCP
        optimum and prints a timing table (visible with ``pytest -s``).
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
        analytic_obj = MatrixGameEllipsoidDualObjective(game, region)
        default_obj = DefaultDualObjective(
            game, region, max_iter=500, step_size=0.5
        )
        c0 = np.ones(m) / m

        t0 = time.perf_counter()
        analytic_result = AcceleratedProximalGradientDualSolver(
            analytic_obj, space, max_iter=500, step_size=0.1
        ).solve(c0)
        t_analytic = time.perf_counter() - t0

        t0 = time.perf_counter()
        default_result = AcceleratedProximalGradientDualSolver(
            default_obj, space, max_iter=100, step_size=0.1
        ).solve(c0)
        t_default = time.perf_counter() - t0

        socp_result = MarkowitzSolver(game, region, space).solve(c0)

        with capsys.disabled():
            print(
                f"\nAnalytic APG: {t_analytic * 1e3:7.1f} ms  "
                f"obj={analytic_result.objective:+.6f}  "
                f"outer_iters={analytic_result.n_iterations}"
            )
            print(
                f"Default APG:  {t_default * 1e3:7.1f} ms  "
                f"obj={default_result.objective:+.6f}  "
                f"outer_iters={default_result.n_iterations}"
            )
            print(f"SOCP exact:              obj={socp_result.objective:+.6f}")

        assert space.contains(analytic_result.x), "analytic result not feasible"
        assert space.contains(default_result.x), "default result not feasible"
        assert socp_result.objective >= analytic_result.objective - 1e-4
        assert socp_result.objective >= default_result.objective - 1e-3


@pytest.mark.parametrize(
    "seed,m,n",
    [
        (1501, 3, 3),
        (2501, 5, 4),
        (3501, 4, 2),
    ],
)
def test_default_vs_analytic_random(seed: int, m: int, n: int) -> None:
    """DefaultDualObjective matches MatrixGameEllipsoidDualObjective on random problems."""
    np.random.seed(seed)
    A = np.random.randn(m, n)
    beta_hat = np.random.randn(n)
    R = np.random.randn(n, n)
    Sigma = 0.1 * (R.T @ R + np.eye(n))
    game = MatrixGame(A)
    region = Ellipsoid(beta_hat, Sigma)
    analytic = MatrixGameEllipsoidDualObjective(game, region)
    default = DefaultDualObjective(game, region, max_iter=2_000, step_size=0.5)

    rng = np.random.default_rng(seed)
    for _ in range(10):
        c = np.abs(rng.standard_normal(m))
        c /= c.sum()
        assert region.contains(default.minimizer(c)), "minimizer not feasible"
        assert (
            pytest.approx(default.evaluate(c), abs=1e-3) == analytic.evaluate(c)
        ), f"value mismatch at c={c}"


class TestDefaultPrimalObjective:
    r"""Tests for DefaultPrimalObjective (inner-APG primal objective).

    Uses a 2-option identity game:
    :math:`h(\beta) = \max_{c \in C} c^\top \beta = \max(0, \beta_1, \beta_2)`.

    With :math:`\beta = [0.9, 0]` the primal value is :math:`0.9`.
    """

    @staticmethod
    def _make_problem(
        step_size: float = 0.5,
        max_iter: int = 1_000,
    ) -> tuple[MatrixGame, AllocationDecision, DefaultPrimalObjective]:
        game = MatrixGame(np.eye(2))
        space = AllocationDecision(2)
        obj = DefaultPrimalObjective(game, space, max_iter=max_iter, step_size=step_size)
        return game, space, obj

    @staticmethod
    def _h_analytic(A: np.ndarray, beta: np.ndarray) -> float:
        """Analytic h(beta) = max(0, max_i (A beta)_i) for AllocationDecision."""
        return float(max(0.0, float(np.max(A @ beta))))

    @staticmethod
    def test_evaluate_matches_analytic() -> None:
        """h(beta) = max(0, beta_1, beta_2) for the identity game."""
        game, _, obj = TestDefaultPrimalObjective._make_problem()
        for beta in [
            np.array([0.9, 0.0]),
            np.array([0.5, 0.3]),
            np.array([-0.1, -0.2]),
            np.array([0.0, 0.7]),
        ]:
            expected = TestDefaultPrimalObjective._h_analytic(game.A, beta)
            assert pytest.approx(obj.evaluate(beta), abs=1e-3) == expected

    @staticmethod
    def test_maximizer_in_space() -> None:
        """The APG maximizer must lie in C."""
        game, space, obj = TestDefaultPrimalObjective._make_problem()
        for beta in [np.array([0.9, 0.0]), np.array([0.5, 0.5]), np.array([0.2, 0.8])]:
            assert space.contains(obj.maximizer(beta)), f"maximizer outside C for beta={beta}"

    @staticmethod
    def test_evaluate_equals_g_at_maximizer() -> None:
        """evaluate(beta) must equal g(maximizer(beta), beta)."""
        game, _, obj = TestDefaultPrimalObjective._make_problem()
        beta = np.array([0.7, 0.3])
        c_star = obj.maximizer(beta)
        assert pytest.approx(obj.evaluate(beta), abs=1e-12) == game.evaluate(c_star, beta)

    @staticmethod
    def test_grad_beta_by_envelope() -> None:
        """grad_beta(beta) must equal model.grad_beta(maximizer(beta), beta)."""
        game, _, obj = TestDefaultPrimalObjective._make_problem()
        beta = np.array([0.8, 0.2])
        c_star = obj.maximizer(beta)
        np.testing.assert_allclose(
            obj.grad_beta(beta), game.grad_beta(c_star, beta), atol=1e-12
        )

    @staticmethod
    def test_dimension_mismatch_raises() -> None:
        """Mismatched model/space dimensions must raise ValueError."""
        game = MatrixGame(np.eye(3))
        space = AllocationDecision(2)
        with pytest.raises(ValueError, match="dim_c"):
            DefaultPrimalObjective(game, space)

    @staticmethod
    def test_performance_comparison(capsys) -> None:
        r"""Primal APG vs analytic dual APG: same maximin value, different speed.

        By the minimax theorem,
        :math:`\max_c f(c) = \min_\beta h(\beta) = v^*`.
        This test drives both paths to :math:`v^*` and compares wall-clock
        times.  The MarkowitzSolver provides the exact reference.
        """
        np.random.seed(7)
        m, n = 8, 4
        A = np.random.randn(m, n)
        beta_hat = np.random.randn(n)
        R = np.random.randn(n, n)
        Sigma = 0.1 * (R.T @ R + np.eye(n))
        game = MatrixGame(A)
        region = Ellipsoid(beta_hat, Sigma)
        space = AllocationDecision(m)
        analytic_dual = MatrixGameEllipsoidDualObjective(game, region)
        default_primal = DefaultPrimalObjective(game, space, max_iter=500, step_size=0.5)

        c0 = np.ones(m) / m
        beta0 = region.project(np.zeros(n))

        t0 = time.perf_counter()
        dual_result = AcceleratedProximalGradientDualSolver(
            analytic_dual, space, max_iter=2_000, step_size=0.1
        ).solve(c0)
        t_dual = time.perf_counter() - t0

        t0 = time.perf_counter()
        primal_result = ProximalSubgradientPrimalSolver(
            default_primal, region, max_iter=500, step_size=0.5
        ).solve(beta0)
        t_primal = time.perf_counter() - t0

        socp_result = MarkowitzSolver(game, region, space).solve(c0)

        with capsys.disabled():
            print(
                f"\nAnalytic dual APG:  {t_dual * 1e3:7.1f} ms  "
                f"obj={dual_result.objective:+.6f}  "
                f"outer_iters={dual_result.n_iterations}"
            )
            print(
                f"Default primal APG: {t_primal * 1e3:7.1f} ms  "
                f"obj={primal_result.objective:+.6f}  "
                f"outer_iters={primal_result.n_iterations}"
            )
            print(f"SOCP exact:                    obj={socp_result.objective:+.6f}")

        assert space.contains(dual_result.x), "dual result not feasible"
        assert region.contains(primal_result.x), "primal result not feasible"
        # SOCP is globally optimal; both gradient methods cannot exceed it.
        assert socp_result.objective >= dual_result.objective - 1e-3
        # Primal h(beta) >= minimax value; the primal solver minimizes h.
        assert primal_result.objective >= socp_result.objective - 1e-3


@pytest.mark.parametrize(
    "seed,m,n",
    [
        (1601, 3, 2),
        (2601, 4, 3),
        (3601, 2, 4),
    ],
)
def test_default_primal_vs_formula_random(seed: int, m: int, n: int) -> None:
    """DefaultPrimalObjective matches max(0, max_i (A beta)_i) on random problems."""
    np.random.seed(seed)
    A = np.random.randn(m, n)
    game = MatrixGame(A)
    space = AllocationDecision(m)
    obj = DefaultPrimalObjective(game, space, max_iter=2_000, step_size=0.5)

    rng = np.random.default_rng(seed)
    for _ in range(10):
        beta = rng.standard_normal(n)
        assert space.contains(obj.maximizer(beta)), "maximizer not feasible"
        expected = float(max(0.0, float(np.max(A @ beta))))
        assert (
            pytest.approx(obj.evaluate(beta), abs=1e-3) == expected
        ), f"value mismatch at beta={beta}"
