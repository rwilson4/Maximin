# pyre-unsafe
"""Tests for CobbDouglas and CobbDouglasEllipsoidDualObjective."""

import numpy as np
import pytest

from maximin.confidence_regions import Ellipsoid
from maximin.decision_spaces import AllocationDecision
from maximin.outcome_models import CobbDouglas
from maximin.problem_objectives import CobbDouglasEllipsoidDualObjective
from maximin.solvers import AcceleratedProximalGradientDualSolver


class TestCobbDouglas:
    """Tests for the CobbDouglas outcome model."""

    @staticmethod
    def test_dim_c() -> None:
        assert CobbDouglas(3).dim_c == 3

    @staticmethod
    def test_dim_beta() -> None:
        """dim_beta is m + 1 (one baseline + m elasticities)."""
        assert CobbDouglas(3).dim_beta == 4

    @staticmethod
    def test_invalid_m() -> None:
        with pytest.raises(ValueError, match="m must be"):
            CobbDouglas(0)

    @staticmethod
    def test_evaluate_known_value() -> None:
        r"""g([0.5, 0.5]; [0, 0.3, 0.3]) = 1.5^{0.6}."""
        model = CobbDouglas(2)
        c = np.array([0.5, 0.5])
        beta = np.array([0.0, 0.3, 0.3])
        assert pytest.approx(model.evaluate(c, beta), rel=1e-12) == 1.5**0.6

    @staticmethod
    def test_evaluate_at_zero_allocation() -> None:
        """g(0; beta) = exp(beta_0) regardless of elasticities."""
        model = CobbDouglas(3)
        beta = np.array([0.5, 0.2, 0.3, 0.1])
        assert pytest.approx(model.evaluate(np.zeros(3), beta), rel=1e-12) == np.exp(
            0.5
        )

    @staticmethod
    def test_evaluate_nonzero_baseline() -> None:
        """Nonzero beta_0 scales the output by exp(beta_0)."""
        model = CobbDouglas(1)
        c = np.array([1.0])
        beta0 = np.array([0.0, 0.5])
        beta1 = np.array([2.0, 0.5])
        ratio = model.evaluate(c, beta1) / model.evaluate(c, beta0)
        assert pytest.approx(ratio, rel=1e-12) == np.exp(2.0)

    @staticmethod
    @pytest.mark.parametrize("seed,m", [(7, 1), (13, 2), (42, 4)])
    def test_grad_c_finite_difference(seed: int, m: int) -> None:
        """Analytic grad_c must match central finite differences."""
        np.random.seed(seed)
        model = CobbDouglas(m)
        c = np.random.rand(m) * 0.8
        beta = np.array([0.0] + list(np.random.rand(m) * 0.2 + 0.05))
        eps = 1e-7
        grad_fd = np.zeros(m)
        for i in range(m):
            cp, cm = c.copy(), c.copy()
            cp[i] += eps
            cm[i] -= eps
            grad_fd[i] = (model.evaluate(cp, beta) - model.evaluate(cm, beta)) / (
                2 * eps
            )
        np.testing.assert_allclose(grad_fd, model.grad_c(c, beta), rtol=1e-5)

    @staticmethod
    @pytest.mark.parametrize("seed,m", [(7, 1), (13, 2), (42, 4)])
    def test_grad_beta_finite_difference(seed: int, m: int) -> None:
        """Analytic grad_beta must match central finite differences."""
        np.random.seed(seed)
        model = CobbDouglas(m)
        c = np.random.rand(m) * 0.8
        beta = np.array([0.0] + list(np.random.rand(m) * 0.2 + 0.05))
        n = m + 1
        eps = 1e-7
        grad_fd = np.zeros(n)
        for i in range(n):
            bp, bm = beta.copy(), beta.copy()
            bp[i] += eps
            bm[i] -= eps
            grad_fd[i] = (model.evaluate(c, bp) - model.evaluate(c, bm)) / (2 * eps)
        np.testing.assert_allclose(grad_fd, model.grad_beta(c, beta), rtol=1e-5)


class TestCobbDouglasEllipsoidDualObjective:
    """Tests for CobbDouglasEllipsoidDualObjective."""

    @staticmethod
    def _setup(
        m: int = 2, sigma: float = 0.01
    ) -> tuple[CobbDouglas, Ellipsoid, CobbDouglasEllipsoidDualObjective]:
        model = CobbDouglas(m)
        beta_hat = np.array([0.0] + [0.3] * m)
        Sigma = (sigma**2) * np.eye(m + 1)
        region = Ellipsoid(beta_hat, Sigma)
        return model, region, CobbDouglasEllipsoidDualObjective(model, region)

    @staticmethod
    def test_dim_mismatch_raises() -> None:
        with pytest.raises(ValueError, match="dim_beta"):
            CobbDouglasEllipsoidDualObjective(
                CobbDouglas(2), Ellipsoid(np.zeros(5), np.eye(5))
            )

    @staticmethod
    def test_minimizer_in_region() -> None:
        _, region, obj = TestCobbDouglasEllipsoidDualObjective._setup()
        beta_star = obj.minimizer(np.array([0.5, 0.5]))
        assert region.contains(beta_star)

    @staticmethod
    def test_minimizer_is_worst_case() -> None:
        """g(c; beta*(c)) must be <= g(c; beta_hat) for any c in C."""
        model, _, obj = TestCobbDouglasEllipsoidDualObjective._setup()
        c = np.array([0.5, 0.5])
        beta_hat = np.array([0.0, 0.3, 0.3])
        assert (
            model.evaluate(c, obj.minimizer(c)) <= model.evaluate(c, beta_hat) + 1e-12
        )

    @staticmethod
    def test_evaluate_equals_g_at_minimizer() -> None:
        """f(c) must equal g(c; beta*(c))."""
        model, _, obj = TestCobbDouglasEllipsoidDualObjective._setup()
        c = np.array([0.5, 0.5])
        assert pytest.approx(obj.evaluate(c), rel=1e-12) == model.evaluate(
            c, obj.minimizer(c)
        )

    @staticmethod
    def test_evaluate_at_zero_allocation() -> None:
        """At c=0 the dual objective is exp(beta_hat_0 - sigma_0)."""
        sigma = 0.01
        _, _, obj = TestCobbDouglasEllipsoidDualObjective._setup(sigma=sigma)
        c = np.zeros(2)
        # v = [1, 0, 0], so v^T Sigma v = sigma^2, sqrt = sigma
        # a = beta_hat_0 = 0, so f = exp(0 - sigma)
        assert pytest.approx(obj.evaluate(c), rel=1e-12) == np.exp(-sigma)

    @staticmethod
    @pytest.mark.parametrize("seed", [7, 13, 42])
    def test_grad_c_finite_difference(seed: int) -> None:
        """Envelope-theorem grad_c must match central finite differences."""
        np.random.seed(seed)
        m = 3
        model = CobbDouglas(m)
        beta_hat = np.array([0.0] + list(np.random.rand(m) * 0.15 + 0.1))
        region = Ellipsoid(beta_hat, 0.0001 * np.eye(m + 1))
        obj = CobbDouglasEllipsoidDualObjective(model, region)
        c = np.random.rand(m) * 0.5
        eps = 1e-7
        grad_fd = np.zeros(m)
        for i in range(m):
            cp, cm = c.copy(), c.copy()
            cp[i] += eps
            cm[i] -= eps
            grad_fd[i] = (obj.evaluate(cp) - obj.evaluate(cm)) / (2 * eps)
        np.testing.assert_allclose(grad_fd, obj.grad_c(c), rtol=1e-5)


class TestCobbDouglasAPGSolver:
    r"""Test AcceleratedProximalGradientDualSolver with CobbDouglasEllipsoidDualObjective.

    Problem: 2 goods, beta_hat = [0, 0.3, 0.3], Sigma = 0.0001 * I_3.

    With a tiny ellipsoid the dual objective approximates
    g(c; beta_hat) = (1+c_1)^{0.3} (1+c_2)^{0.3}.  By symmetry and
    first-order conditions (beta_i/(1+c_i) equal for all active goods),
    the maximizer over AllocationDecision(2) is c* = [0.5, 0.5].

    The small ellipsoid ensures the worst-case beta always has non-negative
    elasticities (beta_hat_i - sigma = 0.3 - 0.01 = 0.29 > 0) summing to
    less than 1 (worst-case sum <= 2*0.3 = 0.6 < 1).
    """

    @staticmethod
    def _setup() -> tuple[
        CobbDouglasEllipsoidDualObjective,
        AllocationDecision,
        AcceleratedProximalGradientDualSolver,
    ]:
        model = CobbDouglas(2)
        beta_hat = np.array([0.0, 0.3, 0.3])
        Sigma = 0.0001 * np.eye(3)
        region = Ellipsoid(beta_hat, Sigma)
        obj = CobbDouglasEllipsoidDualObjective(model, region)
        space = AllocationDecision(2)
        solver = AcceleratedProximalGradientDualSolver(
            obj, space, max_iter=5_000, step_size=0.1
        )
        return obj, space, solver

    @staticmethod
    def test_result_feasible() -> None:
        """Returned c* must lie in AllocationDecision(2)."""
        _, space, solver = TestCobbDouglasAPGSolver._setup()
        result = solver.solve(np.array([0.5, 0.5]))
        assert space.contains(result.x), f"result.x = {result.x} not feasible"

    @staticmethod
    def test_converges_to_symmetric_optimum() -> None:
        """APG must find c* ≈ [0.5, 0.5] for the symmetric problem."""
        _, _, solver = TestCobbDouglasAPGSolver._setup()
        result = solver.solve(np.array([0.5, 0.5]))
        assert result.converged
        np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-3)

    @staticmethod
    def test_converges_from_asymmetric_start() -> None:
        """APG should reach the same optimum from an off-center start."""
        _, _, solver = TestCobbDouglasAPGSolver._setup()
        result = solver.solve(np.array([0.1, 0.9]))
        np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-3)

    @staticmethod
    def test_objective_not_worse_than_uniform() -> None:
        """APG result must be at least as good as the uniform allocation."""
        obj, _, solver = TestCobbDouglasAPGSolver._setup()
        c_uniform = np.array([0.5, 0.5])
        result = solver.solve(c_uniform)
        assert result.objective >= obj.evaluate(c_uniform) - 1e-8

    @staticmethod
    def test_objective_matches_evaluate() -> None:
        """result.objective must equal obj.evaluate(result.x)."""
        obj, _, solver = TestCobbDouglasAPGSolver._setup()
        result = solver.solve(np.array([0.3, 0.7]))
        assert pytest.approx(result.objective, rel=1e-8) == obj.evaluate(result.x)

    @staticmethod
    def test_worst_case_elasticities_nonnegative() -> None:
        """Worst-case beta must have non-negative elasticities at c*."""
        obj, _, solver = TestCobbDouglasAPGSolver._setup()
        result = solver.solve(np.array([0.5, 0.5]))
        beta_star = obj.minimizer(result.x)
        assert np.all(beta_star[1:] >= 0), f"Negative elasticity: {beta_star[1:]}"

    @staticmethod
    def test_worst_case_elasticities_sum_less_than_one() -> None:
        """Worst-case elasticities must sum to less than 1 at c*."""
        obj, _, solver = TestCobbDouglasAPGSolver._setup()
        result = solver.solve(np.array([0.5, 0.5]))
        beta_star = obj.minimizer(result.x)
        total = float(beta_star[1:].sum())
        assert total < 1.0, f"Elasticities sum = {total:.4f} >= 1"
