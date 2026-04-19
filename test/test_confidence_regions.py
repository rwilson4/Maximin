# pyre-unsafe
"""Tests for confidence regions."""

import numpy as np
import pytest

from maximin.confidence_regions import Ellipsoid


class TestEllipsoid:
    """Tests for the Ellipsoid confidence region."""

    @staticmethod
    def test_center_is_inside() -> None:
        """The center beta_hat must always be in S."""
        beta_hat = np.array([1.0, -0.5])
        Sigma = np.array([[3.0, 0.5], [0.5, 2.0]])
        region = Ellipsoid(beta_hat, Sigma)
        assert region.contains(beta_hat)

    @staticmethod
    def test_project_inside_unchanged() -> None:
        """A point already inside S is returned as-is."""
        beta_hat = np.zeros(2)
        Sigma = np.eye(2)
        region = Ellipsoid(beta_hat, Sigma)
        beta = np.array([0.3, 0.4])  # ||beta||^2 = 0.25 < 1
        np.testing.assert_allclose(region.project(beta), beta)

    @staticmethod
    def test_project_unit_ball_known_value() -> None:
        """For the unit ball (Sigma=I), project([3, 4]) = [0.6, 0.8]."""
        region = Ellipsoid(np.zeros(2), np.eye(2))
        proj = region.project(np.array([3.0, 4.0]))
        np.testing.assert_allclose(proj, [0.6, 0.8], atol=1e-10)

    @staticmethod
    def test_project_axis_aligned_known_value() -> None:
        r"""For Sigma=diag([4, 1]), project([3, 0]) = [2, 0].

        The ellipsoid is :math:`\beta_0^2/4 + \beta_1^2 \le 1`. The
        secular equation reduces to :math:`36/(4+\nu)^2 = 1`, giving
        :math:`\nu = 2` and projection :math:`[2, 0]`.
        """
        region = Ellipsoid(np.zeros(2), np.diag([4.0, 1.0]))
        proj = region.project(np.array([3.0, 0.0]))
        np.testing.assert_allclose(proj, [2.0, 0.0], atol=1e-10)

    @staticmethod
    def test_projected_outside_point_is_on_boundary() -> None:
        """A point projected from outside should have Mahalanobis distance 1."""
        beta_hat = np.array([0.5, -0.5])
        Sigma = np.array([[2.0, 0.8], [0.8, 1.5]])
        region = Ellipsoid(beta_hat, Sigma)
        outside = np.array([5.0, -5.0])
        proj = region.project(outside)
        # Verify proj is on the boundary.
        delta = proj - beta_hat
        mahal = float(delta @ np.linalg.solve(Sigma, delta))
        assert pytest.approx(mahal, abs=1e-8) == 1.0

    @staticmethod
    def test_project_is_closer_than_center() -> None:
        """The projection of an outside point is closer than the center."""
        region = Ellipsoid(np.zeros(2), np.eye(2))
        outside = np.array([3.0, 4.0])
        proj = region.project(outside)
        d_proj = float(np.linalg.norm(proj - outside))
        d_center = float(np.linalg.norm(outside))
        assert d_proj < d_center

    @staticmethod
    def test_contains_outside() -> None:
        """A point well outside the ellipsoid should not be contained."""
        region = Ellipsoid(np.zeros(2), np.eye(2))
        assert not region.contains(np.array([2.0, 0.0]))

    @staticmethod
    def test_contains_boundary() -> None:
        """A point on the boundary should be contained (within tolerance)."""
        region = Ellipsoid(np.zeros(2), np.eye(2))
        assert region.contains(np.array([1.0, 0.0]))

    @staticmethod
    def test_invalid_non_positive_definite_sigma() -> None:
        """A singular Sigma should raise ValueError."""
        with pytest.raises(ValueError, match="positive definite"):
            Ellipsoid(np.zeros(2), np.zeros((2, 2)))

    @staticmethod
    def test_invalid_shape_mismatch() -> None:
        """Sigma shape inconsistent with beta_hat should raise ValueError."""
        with pytest.raises(ValueError, match="shape"):
            Ellipsoid(np.zeros(2), np.eye(3))

    @staticmethod
    def test_properties_return_copies() -> None:
        """Mutating returned arrays must not affect the stored region."""
        beta_hat = np.array([1.0, 2.0])
        Sigma = np.eye(2)
        region = Ellipsoid(beta_hat, Sigma)
        region.beta_hat[0] = 999.0
        assert region.beta_hat[0] == 1.0


@pytest.mark.parametrize(
    "seed,n",
    [
        (1201, 2),
        (2201, 5),
        (3201, 10),
        (4201, 3),
    ],
)
def test_ellipsoid_project_always_feasible(seed: int, n: int) -> None:
    """Projection of any point lies in S."""
    np.random.seed(seed)
    beta_hat = np.random.randn(n)
    # Random PD matrix via S = R^T R + I.
    R = np.random.randn(n, n)
    Sigma = R.T @ R + np.eye(n)
    region = Ellipsoid(beta_hat, Sigma)
    for _ in range(20):
        beta = np.random.randn(n) * 5.0
        assert region.contains(region.project(beta)), "projection not feasible"


@pytest.mark.parametrize(
    "seed,n",
    [
        (1202, 2),
        (2202, 5),
        (3202, 4),
    ],
)
def test_ellipsoid_project_idempotent(seed: int, n: int) -> None:
    """Projecting a second time leaves the point unchanged."""
    np.random.seed(seed)
    beta_hat = np.random.randn(n)
    R = np.random.randn(n, n)
    Sigma = R.T @ R + np.eye(n)
    region = Ellipsoid(beta_hat, Sigma)
    for _ in range(20):
        beta = np.random.randn(n) * 5.0
        proj = region.project(beta)
        np.testing.assert_allclose(
            region.project(proj), proj, atol=1e-10, err_msg="project not idempotent"
        )


@pytest.mark.parametrize(
    "seed,n",
    [
        (1203, 2),
        (2203, 5),
        (3203, 3),
    ],
)
def test_ellipsoid_project_outside_lands_on_boundary(seed: int, n: int) -> None:
    """Outside-point projections should have Mahalanobis distance 1."""
    np.random.seed(seed)
    beta_hat = np.random.randn(n)
    R = np.random.randn(n, n)
    Sigma = R.T @ R + np.eye(n)
    region = Ellipsoid(beta_hat, Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    for _ in range(10):
        beta = beta_hat + np.random.randn(n) * 10.0
        if region.contains(beta):
            continue
        proj = region.project(beta)
        delta = proj - beta_hat
        mahal = float(delta @ Sigma_inv @ delta)
        assert pytest.approx(mahal, abs=1e-8) == 1.0
