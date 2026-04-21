# pyre-unsafe
"""Tests for confidence regions."""

import numpy as np
import pytest
from scipy.stats import chi2

from maximin.confidence_regions import (
    BinomialRegion,
    Ellipsoid,
    GammaRegion,
    Hypercube,
    PoissonRegion,
)


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


class TestHypercube:
    """Tests for the Hypercube confidence region."""

    @staticmethod
    def test_center_is_inside() -> None:
        """The midpoint of the hypercube must be contained."""
        lo = np.array([-1.0, 0.0, 2.0])
        hi = np.array([1.0, 3.0, 5.0])
        region = Hypercube(lo, hi)
        mid = (lo + hi) / 2.0
        assert region.contains(mid)

    @staticmethod
    def test_project_inside_unchanged() -> None:
        """A point already inside S is returned unchanged."""
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        region = Hypercube(lo, hi)
        beta = np.array([0.3, 0.7])
        np.testing.assert_allclose(region.project(beta), beta)

    @staticmethod
    def test_project_outside_clamps() -> None:
        """A point outside is clamped component-wise to the bounds."""
        lo = np.array([-1.0, -1.0])
        hi = np.array([1.0, 1.0])
        region = Hypercube(lo, hi)
        beta = np.array([3.0, -5.0])
        np.testing.assert_allclose(region.project(beta), [1.0, -1.0])

    @staticmethod
    def test_project_mixed_components() -> None:
        """Components inside bounds are unchanged; those outside are clamped."""
        lo = np.array([0.0, 2.0, -3.0])
        hi = np.array([1.0, 5.0, 3.0])
        region = Hypercube(lo, hi)
        beta = np.array([-1.0, 3.5, 7.0])
        np.testing.assert_allclose(region.project(beta), [0.0, 3.5, 3.0])

    @staticmethod
    def test_contains_boundary() -> None:
        """Points exactly on the boundary should be contained."""
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        region = Hypercube(lo, hi)
        assert region.contains(np.array([0.0, 1.0]))
        assert region.contains(np.array([1.0, 0.0]))

    @staticmethod
    def test_contains_outside() -> None:
        """A point outside should not be contained."""
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        region = Hypercube(lo, hi)
        assert not region.contains(np.array([1.1, 0.5]))

    @staticmethod
    def test_dim() -> None:
        """dim property returns the number of components."""
        lo = np.zeros(5)
        hi = np.ones(5)
        assert Hypercube(lo, hi).dim == 5

    @staticmethod
    def test_properties_return_copies() -> None:
        """Mutating returned arrays must not affect the stored region."""
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        region = Hypercube(lo, hi)
        region.lo[0] = 999.0
        assert region.lo[0] == 0.0
        region.hi[1] = 999.0
        assert region.hi[1] == 1.0

    @staticmethod
    def test_invalid_lo_gt_hi() -> None:
        """lo > hi for any component should raise ValueError."""
        with pytest.raises(ValueError, match="lo must be <= hi"):
            Hypercube(np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    @staticmethod
    def test_invalid_shape_mismatch() -> None:
        """lo and hi with different shapes should raise ValueError."""
        with pytest.raises(ValueError, match="shape"):
            Hypercube(np.zeros(3), np.ones(2))

    @staticmethod
    def test_invalid_not_1d() -> None:
        """2-D lo should raise ValueError."""
        with pytest.raises(ValueError, match="1-dimensional"):
            Hypercube(np.zeros((2, 2)), np.ones((2, 2)))


@pytest.mark.parametrize("seed,n", [(10, 2), (20, 5), (30, 10)])
def test_hypercube_project_always_feasible(seed: int, n: int) -> None:
    """Projection of any point lies in S."""
    np.random.seed(seed)
    lo = np.random.randn(n)
    hi = lo + np.abs(np.random.randn(n)) + 0.1
    region = Hypercube(lo, hi)
    for _ in range(30):
        beta = np.random.randn(n) * 5.0
        assert region.contains(region.project(beta))


@pytest.mark.parametrize("seed,n", [(11, 2), (21, 4)])
def test_hypercube_project_idempotent(seed: int, n: int) -> None:
    """Projecting a second time leaves the point unchanged."""
    np.random.seed(seed)
    lo = np.random.randn(n)
    hi = lo + np.abs(np.random.randn(n)) + 0.1
    region = Hypercube(lo, hi)
    for _ in range(20):
        beta = np.random.randn(n) * 5.0
        proj = region.project(beta)
        np.testing.assert_allclose(region.project(proj), proj, atol=1e-15)


# ---------------------------------------------------------------------------
# BinomialRegion
# ---------------------------------------------------------------------------


class TestBinomialRegion:
    """Tests for BinomialRegion."""

    @staticmethod
    def _make(m: int = 1) -> "BinomialRegion":
        n = np.full(m, 100.0)
        k = np.full(m, 50.0)
        threshold = float(chi2.ppf(0.95, df=m))
        return BinomialRegion(n, k, threshold)

    def test_mle_is_inside(self) -> None:
        region = self._make(m=2)
        assert region.contains(region.beta_hat)

    def test_known_outside(self) -> None:
        region = self._make(m=1)
        # LRT at p=0.1 ≫ threshold for n=100, k=50.
        assert not region.contains(np.array([0.1]))

    def test_known_inside(self) -> None:
        # n=100, k=50, threshold=chi2(0.95,1)≈3.84.  LRT at p=0.41 ≈ 3.29 < 3.84.
        region = self._make(m=1)
        assert region.contains(np.array([0.41]))

    def test_known_outside_near_boundary(self) -> None:
        # LRT at p=0.40 ≈ 4.08 > 3.84.
        region = self._make(m=1)
        assert not region.contains(np.array([0.40]))

    def test_project_inside_unchanged(self) -> None:
        region = self._make(m=1)
        beta = np.array([0.45])
        np.testing.assert_allclose(region.project(beta), beta)

    def test_project_outside_feasible(self) -> None:
        region = self._make(m=2)
        outside = np.array([0.1, 0.9])
        assert region.contains(region.project(outside))

    def test_project_outside_on_boundary(self) -> None:
        region = self._make(m=1)
        outside = np.array([0.1])
        proj = region.project(outside)
        lrt = region._lrt(proj)
        assert pytest.approx(lrt, abs=1e-4) == region._threshold

    def test_project_idempotent(self) -> None:
        region = self._make(m=2)
        outside = np.array([0.05, 0.95])
        proj = region.project(outside)
        np.testing.assert_allclose(region.project(proj), proj, atol=1e-6)

    def test_dim(self) -> None:
        assert self._make(m=3).dim == 3

    def test_beta_hat_returns_copy(self) -> None:
        region = self._make(m=2)
        bh = region.beta_hat
        bh[0] = 999.0
        assert region.beta_hat[0] != 999.0

    def test_invalid_n_not_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            BinomialRegion(np.array([0.0]), np.array([0.0]), 3.84)

    def test_invalid_k_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="0 <= k"):
            BinomialRegion(np.array([10.0]), np.array([11.0]), 3.84)

    def test_invalid_k_negative(self) -> None:
        with pytest.raises(ValueError, match="0 <= k"):
            BinomialRegion(np.array([10.0]), np.array([-1.0]), 3.84)

    def test_invalid_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            BinomialRegion(np.array([10.0, 10.0]), np.array([5.0]), 3.84)

    def test_invalid_n_not_1d(self) -> None:
        with pytest.raises(ValueError, match="1-dimensional"):
            BinomialRegion(np.ones((2, 2)), np.ones((2, 2)), 3.84)

    def test_edge_k_zero(self) -> None:
        """k=0 (MLE on boundary) should not raise."""
        region = BinomialRegion(np.array([20.0]), np.array([0.0]), 3.84)
        assert region.contains(region.beta_hat, atol=1e-6)

    def test_joint_region_not_product(self) -> None:
        """A point on the boundary of each marginal may be outside the joint region."""
        # For m=2, the joint threshold (chi2(0.95,2)≈5.99) is larger than
        # each marginal (chi2(0.95,1)≈3.84), so the joint region is larger
        # along the diagonal.  A point at the marginal boundary along one axis
        # while the other component is at the MLE is strictly inside the joint region.
        n = np.array([100.0, 100.0])
        k = np.array([50.0, 50.0])
        t_joint = float(chi2.ppf(0.95, df=2))
        region = BinomialRegion(n, k, t_joint)
        # p=[0.40, 0.50]: 0.40 is just outside the 1-d marginal boundary,
        # but its LRT contribution is only ~4.08, well below t_joint≈5.99.
        assert region.contains(np.array([0.40, 0.50]))


# ---------------------------------------------------------------------------
# PoissonRegion
# ---------------------------------------------------------------------------


class TestPoissonRegion:
    """Tests for PoissonRegion."""

    @staticmethod
    def _make(m: int = 1) -> "PoissonRegion":
        n = np.full(m, 50.0)
        x_sum = np.full(m, 100.0)  # MLE = 2.0 per component
        threshold = float(chi2.ppf(0.95, df=m))
        return PoissonRegion(n, x_sum, threshold)

    def test_mle_is_inside(self) -> None:
        region = self._make(m=2)
        assert region.contains(region.beta_hat)

    def test_known_outside(self) -> None:
        # LRT at lambda=10 is large.
        region = self._make(m=1)
        assert not region.contains(np.array([10.0]))

    def test_known_inside(self) -> None:
        # MLE = 2.0; a nearby value should be inside.
        region = self._make(m=1)
        assert region.contains(np.array([2.1]))

    def test_project_inside_unchanged(self) -> None:
        region = self._make(m=1)
        beta = np.array([2.05])
        np.testing.assert_allclose(region.project(beta), beta)

    def test_project_outside_feasible(self) -> None:
        region = self._make(m=2)
        outside = np.array([0.01, 20.0])
        assert region.contains(region.project(outside))

    def test_project_outside_on_boundary(self) -> None:
        region = self._make(m=1)
        outside = np.array([10.0])
        proj = region.project(outside)
        lrt = region._lrt(proj)
        assert pytest.approx(lrt, abs=1e-4) == region._threshold

    def test_project_idempotent(self) -> None:
        region = self._make(m=2)
        outside = np.array([0.01, 20.0])
        proj = region.project(outside)
        np.testing.assert_allclose(region.project(proj), proj, atol=1e-6)

    def test_dim(self) -> None:
        assert self._make(m=4).dim == 4

    def test_invalid_n_not_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PoissonRegion(np.array([0.0]), np.array([5.0]), 3.84)

    def test_invalid_x_sum_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            PoissonRegion(np.array([10.0]), np.array([-1.0]), 3.84)

    def test_invalid_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            PoissonRegion(np.array([10.0, 10.0]), np.array([5.0]), 3.84)

    def test_edge_x_sum_zero(self) -> None:
        """x_sum=0 (MLE=0, boundary) should not raise."""
        region = PoissonRegion(np.array([10.0]), np.array([0.0]), 3.84)
        assert region.contains(region.beta_hat, atol=1e-6)


# ---------------------------------------------------------------------------
# GammaRegion
# ---------------------------------------------------------------------------


class TestGammaRegion:
    """Tests for GammaRegion."""

    @staticmethod
    def _make(m: int = 1) -> "GammaRegion":
        alpha = np.full(m, 2.0)
        n = np.full(m, 30.0)
        x_sum = np.full(m, 60.0)  # MLE = n*alpha/x_sum = 1.0 per component
        threshold = float(chi2.ppf(0.95, df=m))
        return GammaRegion(alpha, n, x_sum, threshold)

    def test_mle_is_inside(self) -> None:
        region = self._make(m=2)
        assert region.contains(region.beta_hat)

    def test_known_outside(self) -> None:
        region = self._make(m=1)
        assert not region.contains(np.array([10.0]))

    def test_project_inside_unchanged(self) -> None:
        region = self._make(m=1)
        beta = np.array([1.05])
        np.testing.assert_allclose(region.project(beta), beta)

    def test_project_outside_feasible(self) -> None:
        region = self._make(m=2)
        outside = np.array([0.01, 10.0])
        assert region.contains(region.project(outside))

    def test_project_outside_on_boundary(self) -> None:
        region = self._make(m=1)
        outside = np.array([10.0])
        proj = region.project(outside)
        lrt = region._lrt(proj)
        assert pytest.approx(lrt, abs=1e-4) == region._threshold

    def test_project_idempotent(self) -> None:
        region = self._make(m=2)
        outside = np.array([0.01, 10.0])
        proj = region.project(outside)
        np.testing.assert_allclose(region.project(proj), proj, atol=1e-6)

    def test_dim(self) -> None:
        assert self._make(m=3).dim == 3

    def test_invalid_alpha_not_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            GammaRegion(np.array([0.0]), np.array([10.0]), np.array([5.0]), 3.84)

    def test_invalid_x_sum_not_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            GammaRegion(np.array([2.0]), np.array([10.0]), np.array([0.0]), 3.84)

    def test_invalid_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            GammaRegion(np.array([2.0, 2.0]), np.array([10.0]), np.array([5.0]), 3.84)


# ---------------------------------------------------------------------------
# Parametric tests for all three LR region classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed,m",
    [(100, 1), (200, 2), (300, 3)],
)
def test_binomial_project_always_feasible(seed: int, m: int) -> None:
    """BinomialRegion.project always yields a point in S."""
    rng = np.random.default_rng(seed)
    n = rng.integers(20, 100, size=m).astype(float)
    k = np.array([rng.integers(0, int(ni) + 1) for ni in n], dtype=float)
    threshold = float(chi2.ppf(0.95, df=m))
    region = BinomialRegion(n, k, threshold)
    for _ in range(15):
        beta = rng.uniform(0.05, 0.95, size=m)
        assert region.contains(region.project(beta)), "projection not feasible"


@pytest.mark.parametrize(
    "seed,m",
    [(101, 1), (201, 2), (301, 3)],
)
def test_binomial_project_idempotent(seed: int, m: int) -> None:
    """BinomialRegion.project is idempotent."""
    rng = np.random.default_rng(seed)
    n = rng.integers(20, 100, size=m).astype(float)
    k = np.array([rng.integers(1, int(ni)) for ni in n], dtype=float)
    threshold = float(chi2.ppf(0.95, df=m))
    region = BinomialRegion(n, k, threshold)
    for _ in range(10):
        beta = rng.uniform(0.05, 0.95, size=m)
        proj = region.project(beta)
        np.testing.assert_allclose(
            region.project(proj), proj, atol=1e-6, err_msg="not idempotent"
        )


@pytest.mark.parametrize(
    "seed,m",
    [(102, 1), (202, 2)],
)
def test_poisson_project_always_feasible(seed: int, m: int) -> None:
    """PoissonRegion.project always yields a point in S."""
    rng = np.random.default_rng(seed)
    n = rng.integers(10, 50, size=m).astype(float)
    x_sum = n * rng.uniform(0.5, 5.0, size=m)
    threshold = float(chi2.ppf(0.95, df=m))
    region = PoissonRegion(n, x_sum, threshold)
    for _ in range(15):
        beta = rng.uniform(0.1, 10.0, size=m)
        assert region.contains(region.project(beta)), "projection not feasible"


@pytest.mark.parametrize(
    "seed,m",
    [(103, 1), (203, 2)],
)
def test_gamma_project_always_feasible(seed: int, m: int) -> None:
    """GammaRegion.project always yields a point in S."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.5, 3.0, size=m)
    n = rng.integers(10, 50, size=m).astype(float)
    x_sum = n * alpha / rng.uniform(0.5, 3.0, size=m)
    threshold = float(chi2.ppf(0.95, df=m))
    region = GammaRegion(alpha, n, x_sum, threshold)
    for _ in range(15):
        beta = rng.uniform(0.1, 5.0, size=m)
        assert region.contains(region.project(beta)), "projection not feasible"
