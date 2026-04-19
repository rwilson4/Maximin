# pyre-strict
"""Outcome models g(c; beta) for maximin optimization."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class OutcomeModel(ABC):
    r"""Abstract base for outcome models :math:`g(c; \beta)`.

    An OutcomeModel describes the objective function together with its
    gradients, enabling both evaluation and gradient-based optimization.
    """

    @property
    @abstractmethod
    def dim_c(self) -> int:
        """Dimension of the decision variable c."""

    @property
    @abstractmethod
    def dim_beta(self) -> int:
        """Dimension of the parameter vector beta."""

    @abstractmethod
    def evaluate(
        self,
        c: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> float:
        r"""Evaluate :math:`g(c; \beta)`.

        Parameters
        ----------
        c : npt.NDArray[np.float64]
            Decision variable, shape ``(m,)``.
        beta : npt.NDArray[np.float64]
            Parameter vector, shape ``(n,)``.

        Returns
        -------
        float
            Scalar objective value.
        """

    @abstractmethod
    def grad_c(
        self,
        c: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Gradient of :math:`g` with respect to ``c``, shape ``(m,)``.

        Parameters
        ----------
        c : npt.NDArray[np.float64]
            Decision variable, shape ``(m,)``.
        beta : npt.NDArray[np.float64]
            Parameter vector, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Gradient, shape ``(m,)``.
        """

    @abstractmethod
    def grad_beta(
        self,
        c: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Gradient of :math:`g` with respect to ``beta``, shape ``(n,)``.

        Parameters
        ----------
        c : npt.NDArray[np.float64]
            Decision variable, shape ``(m,)``.
        beta : npt.NDArray[np.float64]
            Parameter vector, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Gradient, shape ``(n,)``.
        """


class MatrixGame(OutcomeModel):
    r"""Bilinear outcome model :math:`g(c; \beta) = c^\top A \beta`.

    Parameters
    ----------
    A : npt.NDArray[np.float64]
        Payoff matrix of shape ``(m, n)``.

    Notes
    -----
    Gradients are

    .. math::

        \nabla_c g = A \beta, \qquad \nabla_\beta g = A^\top c.

    For an ellipsoidal uncertainty set, the dual objective
    :math:`h(c) = \min_{\beta \in S} g(c; \beta)` has a closed form;
    see :class:`~maximin.problem_objectives.MatrixGameEllipsoidDualObjective`.
    """

    def __init__(self, A: npt.NDArray[np.float64]) -> None:
        if A.ndim != 2:
            raise ValueError(f"A must be 2-dimensional, got shape {A.shape}")
        self._A = A.copy()
        self._m, self._n = A.shape

    @property
    def dim_c(self) -> int:
        """Dimension of the decision variable c."""
        return self._m

    @property
    def dim_beta(self) -> int:
        """Dimension of the parameter vector beta."""
        return self._n

    @property
    def A(self) -> npt.NDArray[np.float64]:
        """Payoff matrix, shape ``(m, n)``."""
        return self._A

    def evaluate(
        self,
        c: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> float:
        r"""Evaluate :math:`c^\top A \beta`."""
        return float(np.dot(c, self._A @ beta))

    def grad_c(
        self,
        c: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Return :math:`A \beta`, the gradient with respect to ``c``."""
        return self._A @ beta

    def grad_beta(
        self,
        c: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Return :math:`A^\top c`, the gradient with respect to ``beta``."""
        return self._A.T @ c
