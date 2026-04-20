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


class CobbDouglas(OutcomeModel):
    r"""Cobb--Douglas outcome model.

    .. math::

        g(c;\,\beta)
            = e^{\beta_0}\,\prod_{i=1}^{m}(1 + c_i)^{\beta_i}

    ``c`` represents resource allocations across ``m`` goods.  ``beta[0]``
    is the log baseline output and ``beta[1:]`` are the output elasticities.

    Parameters
    ----------
    m : int
        Number of goods; dimension of ``c``.  Must be >= 1.

    Notes
    -----
    ``dim_c = m`` and ``dim_beta = m + 1``.

    Gradients:

    .. math::

        \frac{\partial g}{\partial c_i}
            = \frac{\beta_i}{1 + c_i}\,g, \qquad
        \nabla_\beta g
            = g\,\bigl[1,\,\log(1+c_1),\,\dots,\,\log(1+c_m)\bigr]^\top.

    The model is concave in ``c`` when every ``beta[i] >= 0`` (for
    ``i >= 1``) and ``sum(beta[1:]) < 1``, and log-linear (hence convex)
    in ``beta`` for every fixed ``c >= 0``.
    """

    def __init__(self, m: int) -> None:
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        self._m = m

    @property
    def dim_c(self) -> int:
        """Dimension of the decision variable c."""
        return self._m

    @property
    def dim_beta(self) -> int:
        """Dimension of the parameter vector beta."""
        return self._m + 1

    def evaluate(
        self,
        c: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> float:
        r"""Evaluate :math:`e^{\beta_0}\prod_i(1+c_i)^{\beta_i}`."""
        log_g = beta[0] + float(np.dot(beta[1:], np.log1p(c)))
        return float(np.exp(log_g))

    def grad_c(
        self,
        c: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Return :math:`g \cdot \beta_{1:m} / (1 + c)`."""
        return self.evaluate(c, beta) * beta[1:] / (1.0 + c)

    def grad_beta(
        self,
        c: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Return :math:`g \cdot [1, \log(1+c_1), \dots, \log(1+c_m)]^\top`."""
        g = self.evaluate(c, beta)
        result = np.empty(self._m + 1)
        result[0] = g
        result[1:] = g * np.log1p(c)
        return result


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
