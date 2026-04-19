# pyre-unsafe
"""Tests for decision spaces."""

import numpy as np
import pytest

from maximin.decision_spaces import AllocationDecision


class TestAllocationDecision:
    """Tests for the AllocationDecision simplex-constrained space."""

    @staticmethod
    def test_project_inside_unchanged() -> None:
        """A point already in C is returned as-is."""
        space = AllocationDecision(3)
        c = np.array([0.2, 0.3, 0.1])
        np.testing.assert_allclose(space.project(c), c)

    @staticmethod
    def test_project_clips_negative_to_feasible() -> None:
        """Negative components are clipped; result stays in C."""
        space = AllocationDecision(2)
        c = np.array([-1.0, 0.5])
        proj = space.project(c)
        np.testing.assert_allclose(proj, [0.0, 0.5])
        assert space.contains(proj)

    @staticmethod
    def test_project_onto_simplex_equal_components() -> None:
        """[0.6, 0.6] maps to [0.5, 0.5] on the probability simplex."""
        space = AllocationDecision(2)
        proj = space.project(np.array([0.6, 0.6]))
        np.testing.assert_allclose(proj, [0.5, 0.5], atol=1e-12)

    @staticmethod
    def test_project_onto_simplex_corner() -> None:
        """[2, 1, 0] maps to the simplex corner [1, 0, 0]."""
        space = AllocationDecision(3)
        proj = space.project(np.array([2.0, 1.0, 0.0]))
        np.testing.assert_allclose(proj, [1.0, 0.0, 0.0], atol=1e-12)

    @staticmethod
    def test_project_is_feasible_after_large_input() -> None:
        """Projection of a large vector is in C."""
        space = AllocationDecision(4)
        proj = space.project(np.array([5.0, 3.0, 2.0, 1.0]))
        assert space.contains(proj)

    @staticmethod
    def test_contains_interior() -> None:
        """Interior point with sum < 1 and non-negative components is in C."""
        space = AllocationDecision(3)
        assert space.contains(np.array([0.1, 0.2, 0.3]))

    @staticmethod
    def test_contains_boundary() -> None:
        """Point on the probability simplex face is in C."""
        space = AllocationDecision(3)
        assert space.contains(np.array([0.5, 0.3, 0.2]))

    @staticmethod
    def test_not_contains_negative() -> None:
        """A point with a negative component is not in C."""
        space = AllocationDecision(3)
        assert not space.contains(np.array([-0.1, 0.5, 0.4]))

    @staticmethod
    def test_not_contains_sum_exceeds_budget() -> None:
        """A point with sum > 1 is not in C."""
        space = AllocationDecision(3)
        assert not space.contains(np.array([0.4, 0.4, 0.4]))

    @staticmethod
    def test_invalid_dimension() -> None:
        """Dimension zero should raise ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            AllocationDecision(0)


@pytest.mark.parametrize(
    "seed,m",
    [
        (1101, 2),
        (2101, 5),
        (3101, 10),
        (4101, 20),
    ],
)
def test_allocation_project_always_feasible(seed: int, m: int) -> None:
    """Projection of any random vector lies in C."""
    np.random.seed(seed)
    space = AllocationDecision(m)
    for _ in range(20):
        c = np.random.randn(m) * 3.0
        assert space.contains(space.project(c)), f"projection infeasible for c={c}"


@pytest.mark.parametrize(
    "seed,m",
    [
        (1102, 2),
        (2102, 5),
        (3102, 10),
    ],
)
def test_allocation_project_idempotent(seed: int, m: int) -> None:
    """Projecting a second time leaves the point unchanged."""
    np.random.seed(seed)
    space = AllocationDecision(m)
    for _ in range(20):
        c = np.random.randn(m) * 3.0
        proj = space.project(c)
        np.testing.assert_allclose(
            space.project(proj), proj, atol=1e-12, err_msg="project not idempotent"
        )
