# pyre-unsafe
"""Tests for dual and primal objectives."""

import numpy as np
import pytest

from maximin.confidence_regions import Ellipsoid
from maximin.outcome_models import MatrixGame
from maximin.problem_objectives import MatrixGameEllipsoidDualObjective


class TestMatrixGameEllipsoidDualObjective:
    """Tests for the analytic dual objective h(c) = c^T A b_hat - ||Sigma^{1/2} A^T c||."""

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
        """h(c) = c^T A b_hat - ||Sigma^{1/2} A^T c|| computed directly."""
        _, _, obj = TestMatrixGameEllipsoidDualObjective._make_problem()
        # A = I, beta_hat = [1, 0], Sigma = 0.01 I
        # h([1, 0]) = 1 - sqrt(0.01) = 1 - 0.1 = 0.9
        c = np.array([1.0, 0.0])
        assert pytest.approx(obj.evaluate(c), abs=1e-12) == 0.9

    @staticmethod
    def test_evaluate_zero_action() -> None:
        """h(0) = 0 since both terms vanish."""
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
        """g(c, beta*(c)) == h(c): the minimizer achieves the dual objective."""
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
        """By the envelope theorem, grad_c h = A @ beta*(c)."""
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
    """Finite-difference check of grad_c h against the analytic gradient."""
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
