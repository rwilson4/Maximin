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
    r"""Cobb--Douglas outcome model with affine-transformed inputs.

    .. math::

        g(c;\,\beta)
            = e^{\beta_0}\,\prod_{i=1}^{m}
              (\delta_i + \gamma_i\,c_i)^{\beta_i}

    ``c`` represents resource allocations across ``m`` goods.  ``beta[0]``
    is the log baseline output and ``beta[1:]`` are the output elasticities.
    ``delta`` and ``gamma`` are fixed known quantities that shift and scale
    each input; they default to ``delta = gamma = 1``, giving the standard
    form :math:`(1 + c_i)^{\beta_i}`.  Setting ``delta = 0`` gives a pure
    power-law model :math:`(\gamma_i c_i)^{\beta_i}`.

    Parameters
    ----------
    m : int
        Number of goods; dimension of ``c``.  Must be >= 1.
    delta : npt.NDArray[np.float64], optional
        Intercept vector, shape ``(m,)``.  Defaults to ``ones(m)``.
    gamma : npt.NDArray[np.float64], optional
        Slope vector, shape ``(m,)``.  Defaults to ``ones(m)``.

    Notes
    -----
    ``dim_c = m`` and ``dim_beta = m + 1``.

    Gradients:

    .. math::

        \frac{\partial g}{\partial c_i}
            = \frac{\gamma_i\,\beta_i}{\delta_i + \gamma_i c_i}\,g,
        \qquad
        \nabla_\beta g
            = g\,\bigl[1,\,\log(\delta_1+\gamma_1 c_1),\,\dots\bigr]^\top.

    The model is concave in ``c`` when every ``beta[i] >= 0`` (for
    ``i >= 1``) and ``sum(beta[1:]) < 1``, and log-linear (hence convex)
    in ``beta`` for every fixed ``c`` with ``delta + gamma * c > 0``.
    """

    def __init__(
        self,
        m: int,
        delta: npt.NDArray[np.float64] | None = None,
        gamma: npt.NDArray[np.float64] | None = None,
    ) -> None:
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        self._m = m
        self._delta = np.ones(m) if delta is None else np.array(delta, dtype=np.float64)
        self._gamma = np.ones(m) if gamma is None else np.array(gamma, dtype=np.float64)
        if self._delta.shape != (m,):
            raise ValueError(f"delta must have shape ({m},), got {self._delta.shape}")
        if self._gamma.shape != (m,):
            raise ValueError(f"gamma must have shape ({m},), got {self._gamma.shape}")

    @property
    def dim_c(self) -> int:
        """Dimension of the decision variable c."""
        return self._m

    @property
    def dim_beta(self) -> int:
        """Dimension of the parameter vector beta."""
        return self._m + 1

    @property
    def delta(self) -> npt.NDArray[np.float64]:
        """Intercept vector, shape ``(m,)``."""
        return self._delta.copy()

    @property
    def gamma(self) -> npt.NDArray[np.float64]:
        """Slope vector, shape ``(m,)``."""
        return self._gamma.copy()

    def _base(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return delta + gamma * c, the argument of each power."""
        return self._delta + self._gamma * c

    def evaluate(
        self,
        c: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> float:
        r"""Evaluate :math:`e^{\beta_0}\prod_i(\delta_i+\gamma_i c_i)^{\beta_i}`."""
        log_g = beta[0] + float(np.dot(beta[1:], np.log(self._base(c))))
        return float(np.exp(log_g))

    def grad_c(
        self,
        c: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Return :math:`g \cdot \gamma\,\beta_{1:m}\,/\,(\delta + \gamma c)`."""
        return self.evaluate(c, beta) * self._gamma * beta[1:] / self._base(c)

    def grad_beta(
        self,
        c: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Return :math:`g \cdot [1,\,\log(\delta_i+\gamma_i c_i),\,\dots]^\top`."""
        g = self.evaluate(c, beta)
        result = np.empty(self._m + 1)
        result[0] = g
        result[1:] = g * np.log(self._base(c))
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
