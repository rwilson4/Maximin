# pyre-unsafe
"""Tests for RobustConstraint and ConstrainedMarkowitzSolver."""

import numpy as np
import pytest

from maximin.confidence_regions import Ellipsoid
from maximin.decision_spaces import AllocationDecision
from maximin.outcome_models import MatrixGame
from maximin.robust_constraints import (
    MatrixGameEllipsoidRobustConstraint,
    RobustConstraint,
)
from maximin.solvers import ConstrainedMarkowitzSolver, MarkowitzSolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_constraint() -> MatrixGameEllipsoidRobustConstraint:
    """B=I_3, gamma_hat=0, Sigma_T=I_3  =>  q(c) = -||c||."""
    return MatrixGameEllipsoidRobustConstraint(
        np.eye(3), Ellipsoid(np.zeros(3), np.eye(3))
    )


# ---------------------------------------------------------------------------
# MatrixGameEllipsoidRobustConstraint unit tests
# ---------------------------------------------------------------------------


class TestMatrixGameEllipsoidRobustConstraint:
    def test_infimum_known_value(self) -> None:
        rc = _identity_constraint()
        c = np.array([0.3, 0.5, 0.2])
        assert pytest.approx(rc.infimum(c), abs=1e-12) == -np.linalg.norm(c)

    def test_infimum_at_zero(self) -> None:
        rc = _identity_constraint()
        assert pytest.approx(rc.infimum(np.zeros(3)), abs=1e-12) == 0.0

    @pytest.mark.parametrize(
        "c",
        [
            np.array([0.3, 0.5, 0.2]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.0]),
        ],
    )
    def test_worst_case_gamma_in_ellipsoid(self, c: np.ndarray) -> None:
        rc = _identity_constraint()
        region = Ellipsoid(np.zeros(3), np.eye(3))
        gamma_star = rc.worst_case_gamma(c)
        assert region.contains(gamma_star, atol=1e-9)

    @pytest.mark.parametrize(
        "c",
        [np.array([0.3, 0.5, 0.2]), np.array([0.6, 0.4, 0.0])],
    )
    def test_infimum_equals_r_at_worst_case_gamma(self, c: np.ndarray) -> None:
        B = np.eye(3)
        rc = _identity_constraint()
        gamma_star = rc.worst_case_gamma(c)
        r_val = float(np.dot(c, B @ gamma_star))
        assert pytest.approx(rc.infimum(c), abs=1e-12) == r_val

    def test_is_satisfied_at_zero(self) -> None:
        rc = _identity_constraint()
        assert rc.is_satisfied(np.zeros(3))

    def test_is_satisfied_nonzero(self) -> None:
        rc = _identity_constraint()
        c = np.array([0.5, 0.5, 0.0])
        assert not rc.is_satisfied(c)
        assert rc.is_satisfied(c, atol=1.0)

    def test_dim_c_property(self) -> None:
        rc = _identity_constraint()
        assert rc.dim_c == 3

    def test_constructor_dimension_mismatch_raises(self) -> None:
        B = np.eye(3)  # p=3 columns
        region = Ellipsoid(np.zeros(2), np.eye(2))  # dim=2
        with pytest.raises(ValueError):
            MatrixGameEllipsoidRobustConstraint(B, region)

    def test_constructor_bad_ndim_raises(self) -> None:
        with pytest.raises(ValueError):
            MatrixGameEllipsoidRobustConstraint(
                np.ones(3), Ellipsoid(np.zeros(3), np.eye(3))
            )

    def test_socp_block_shape(self) -> None:
        rc = _identity_constraint()  # m=3, p=3
        A_block, cone_size = rc.socp_block(3)
        assert A_block.shape == (3 + 1, 3 + 1)
        assert cone_size == 4

    def test_socp_block_wrong_m_raises(self) -> None:
        rc = _identity_constraint()
        with pytest.raises(ValueError):
            rc.socp_block(5)

    def test_worst_case_gamma_zero_c(self) -> None:
        B = np.eye(2)
        gamma_hat = np.array([1.0, 0.5])
        rc = MatrixGameEllipsoidRobustConstraint(
            B, Ellipsoid(gamma_hat, 0.25 * np.eye(2))
        )
        np.testing.assert_allclose(
            rc.worst_case_gamma(np.zeros(2)), gamma_hat, atol=1e-12
        )

    def test_non_square_B(self) -> None:
        # B shape (4, 2), gamma lives in R^2
        B = np.random.default_rng(42).standard_normal((4, 2))
        region = Ellipsoid(np.zeros(2), np.eye(2))
        rc = MatrixGameEllipsoidRobustConstraint(B, region)
        assert rc.dim_c == 4
        c = np.array([0.25, 0.25, 0.25, 0.25])
        gamma_star = rc.worst_case_gamma(c)
        assert region.contains(gamma_star, atol=1e-9)
        r_val = float(np.dot(c, B @ gamma_star))
        assert pytest.approx(rc.infimum(c), abs=1e-10) == r_val

    def test_is_abstract(self) -> None:
        assert issubclass(MatrixGameEllipsoidRobustConstraint, RobustConstraint)


# ---------------------------------------------------------------------------
# ConstrainedMarkowitzSolver tests
# ---------------------------------------------------------------------------


def _base_problem(
    seed: int = 0,
) -> tuple[MatrixGame, Ellipsoid, AllocationDecision]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 2))
    beta_hat = rng.standard_normal(2)
    R = rng.standard_normal((2, 2))
    Sigma = R.T @ R + np.eye(2)
    return MatrixGame(A), Ellipsoid(beta_hat, Sigma), AllocationDecision(3)


class TestConstrainedMarkowitzSolver:
    def test_empty_constraints_matches_unconstrained(self) -> None:
        game, region, space = _base_problem(0)
        c0 = np.zeros(3)
        unconstrained = MarkowitzSolver(game, region, space).solve(c0)
        constrained = ConstrainedMarkowitzSolver(game, region, space, []).solve(c0)
        assert constrained.converged
        np.testing.assert_allclose(constrained.x, unconstrained.x, atol=1e-5)
        assert pytest.approx(constrained.objective, abs=1e-5) == unconstrained.objective

    def test_trivially_satisfied_constraint_matches_unconstrained(
        self,
    ) -> None:
        # q(c) = 10*(c[0]+c[1]+c[2]) - small_penalty >> 0 always
        game = MatrixGame(np.eye(2)[:, :2])
        region = Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2))
        space = AllocationDecision(2)
        B = np.eye(2)
        constraint = MatrixGameEllipsoidRobustConstraint(
            B, Ellipsoid(np.array([10.0, 10.0]), 0.01 * np.eye(2))
        )
        c0 = np.zeros(2)
        unconstrained = MarkowitzSolver(game, region, space).solve(c0)
        constrained = ConstrainedMarkowitzSolver(
            game, region, space, [constraint]
        ).solve(c0)
        assert constrained.converged
        np.testing.assert_allclose(constrained.x, unconstrained.x, atol=1e-4)

    def test_binding_constraint_yields_lower_objective(self) -> None:
        # Objective: maximize c[0] - 0.1*||c||  (beta_hat=[1,0], A=I, small Sigma)
        # Constraint: q(c) = c[1] - 0.1*||c|| >= 0 (forces c[1] > 0)
        game = MatrixGame(np.eye(2))
        region = Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2))
        space = AllocationDecision(2)
        B = np.eye(2)
        constraint = MatrixGameEllipsoidRobustConstraint(
            B, Ellipsoid(np.array([0.0, 1.0]), 0.01 * np.eye(2))
        )
        c0 = np.zeros(2)
        unconstrained = MarkowitzSolver(game, region, space).solve(c0)
        constrained = ConstrainedMarkowitzSolver(
            game, region, space, [constraint]
        ).solve(c0)
        assert constrained.converged
        assert constrained.objective < unconstrained.objective - 1e-4
        assert constraint.is_satisfied(constrained.x, atol=1e-6)
        assert space.contains(constrained.x)

    def test_degenerate_constraint_forces_near_zero_c0(self) -> None:
        # Constraint: q(c) = c[0]*0 - tiny*||c[0]|| >= 0 forces c[0] ~ 0
        game = MatrixGame(np.eye(2))
        region = Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2))
        space = AllocationDecision(2)
        # B = [[1],[0]], gamma_hat=[0], Sigma_T tiny => c[0] <= tiny*||c||
        B = np.array([[1.0], [0.0]])
        constraint = MatrixGameEllipsoidRobustConstraint(
            B, Ellipsoid(np.array([0.0]), 1e-6 * np.eye(1))
        )
        result = ConstrainedMarkowitzSolver(game, region, space, [constraint]).solve(
            np.zeros(2)
        )
        assert result.x[0] < 1e-3
        assert result.objective < 0.01

    def test_multiple_constraints_all_satisfied(self) -> None:
        game = MatrixGame(np.eye(2))
        region = Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2))
        space = AllocationDecision(2)
        B = np.eye(2)
        c1 = MatrixGameEllipsoidRobustConstraint(
            B, Ellipsoid(np.array([0.0, 1.0]), 0.01 * np.eye(2))
        )
        c2 = MatrixGameEllipsoidRobustConstraint(
            B, Ellipsoid(np.array([1.0, 0.0]), 0.01 * np.eye(2))
        )
        result = ConstrainedMarkowitzSolver(game, region, space, [c1, c2]).solve(
            np.zeros(2)
        )
        assert result.converged
        assert c1.is_satisfied(result.x, atol=1e-6)
        assert c2.is_satisfied(result.x, atol=1e-6)
        assert space.contains(result.x)

    def test_dim_mismatch_raises(self) -> None:
        game = MatrixGame(np.eye(2))
        region = Ellipsoid(np.zeros(2), np.eye(2))
        space = AllocationDecision(2)
        # constraint expects m=3, but game has dim_c=2
        constraint = MatrixGameEllipsoidRobustConstraint(
            np.eye(3), Ellipsoid(np.zeros(3), np.eye(3))
        )
        with pytest.raises(ValueError):
            ConstrainedMarkowitzSolver(game, region, space, [constraint])

    @pytest.mark.parametrize("seed", [101, 202, 303])
    def test_constrained_result_feasible_random(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        m, n, p = 4, 3, 2
        A = rng.standard_normal((m, n))
        beta_hat = rng.standard_normal(n)
        R = rng.standard_normal((n, n))
        Sigma = R.T @ R + np.eye(n)
        game = MatrixGame(A)
        region = Ellipsoid(beta_hat, Sigma)
        space = AllocationDecision(m)

        B = rng.standard_normal((m, p))
        gamma_hat = rng.standard_normal(p)
        R_T = rng.standard_normal((p, p))
        Sigma_T = R_T.T @ R_T + np.eye(p)
        constraint = MatrixGameEllipsoidRobustConstraint(
            B, Ellipsoid(gamma_hat, Sigma_T)
        )

        result = ConstrainedMarkowitzSolver(game, region, space, [constraint]).solve(
            np.zeros(m)
        )

        if result.converged:
            assert space.contains(result.x)
            assert constraint.is_satisfied(result.x, atol=1e-5)
