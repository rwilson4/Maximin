# pyre-unsafe
"""Tests for outcome models."""

import numpy as np
import pytest

from maximin.outcome_models import MatrixGame


class TestMatrixGame:
    """Tests for the MatrixGame bilinear outcome model."""

    @staticmethod
    def test_evaluate_known_value() -> None:
        """Verify c^T A beta against a hand-computed result."""
        A = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
        c = np.array([0.5, 0.3, 0.2])
        beta = np.array([1.0, 0.5])
        # g = 0.5*(1+0) + 0.3*(0+1) + 0.2*(1-0.5) = 0.5 + 0.3 + 0.1 = 0.9
        game = MatrixGame(A)
        assert pytest.approx(game.evaluate(c, beta)) == 0.9

    @staticmethod
    def test_grad_c_is_A_beta() -> None:
        """Verify that grad_c g = A beta."""
        A = np.array([[2.0, -1.0], [0.5, 3.0]])
        c = np.array([0.4, 0.6])
        beta = np.array([1.0, -0.5])
        game = MatrixGame(A)
        np.testing.assert_array_equal(game.grad_c(c, beta), A @ beta)

    @staticmethod
    def test_grad_beta_is_At_c() -> None:
        """Verify that grad_beta g = A^T c."""
        A = np.array([[2.0, -1.0], [0.5, 3.0]])
        c = np.array([0.4, 0.6])
        beta = np.array([1.0, -0.5])
        game = MatrixGame(A)
        np.testing.assert_array_equal(game.grad_beta(c, beta), A.T @ c)

    @staticmethod
    def test_dimensions() -> None:
        """Verify dim_c and dim_beta match the payoff matrix shape."""
        A = np.zeros((5, 3))
        game = MatrixGame(A)
        assert game.dim_c == 5
        assert game.dim_beta == 3

    @staticmethod
    def test_invalid_1d_array() -> None:
        """A 1-d array should raise ValueError."""
        with pytest.raises(ValueError, match="2-dimensional"):
            MatrixGame(np.zeros(4))

    @staticmethod
    def test_payoff_matrix_is_copied() -> None:
        """Mutating the original array must not affect the stored matrix."""
        A = np.eye(2)
        game = MatrixGame(A)
        A[0, 0] = 999.0
        assert game.A[0, 0] == 1.0


@pytest.mark.parametrize(
    "seed,m,n",
    [
        (1001, 3, 4),
        (2001, 5, 2),
        (3001, 10, 10),
        (4001, 2, 8),
    ],
)
def test_matrix_game_grad_c_finite_difference(seed: int, m: int, n: int) -> None:
    """Finite-difference check of grad_c against the analytic formula."""
    np.random.seed(seed)
    A = np.random.randn(m, n)
    c = np.random.rand(m)
    beta = np.random.randn(n)
    game = MatrixGame(A)

    eps = 1e-7
    grad_fd = np.zeros(m)
    for i in range(m):
        c_plus = c.copy()
        c_plus[i] += eps
        c_minus = c.copy()
        c_minus[i] -= eps
        grad_fd[i] = (game.evaluate(c_plus, beta) - game.evaluate(c_minus, beta)) / (
            2 * eps
        )

    np.testing.assert_allclose(grad_fd, game.grad_c(c, beta), rtol=1e-5)


@pytest.mark.parametrize(
    "seed,m,n",
    [
        (1002, 3, 4),
        (2002, 5, 2),
        (3002, 10, 10),
        (4002, 2, 8),
    ],
)
def test_matrix_game_grad_beta_finite_difference(seed: int, m: int, n: int) -> None:
    """Finite-difference check of grad_beta against the analytic formula."""
    np.random.seed(seed)
    A = np.random.randn(m, n)
    c = np.random.rand(m)
    beta = np.random.randn(n)
    game = MatrixGame(A)

    eps = 1e-7
    grad_fd = np.zeros(n)
    for i in range(n):
        b_plus = beta.copy()
        b_plus[i] += eps
        b_minus = beta.copy()
        b_minus[i] -= eps
        grad_fd[i] = (game.evaluate(c, b_plus) - game.evaluate(c, b_minus)) / (2 * eps)

    np.testing.assert_allclose(grad_fd, game.grad_beta(c, beta), rtol=1e-5)
