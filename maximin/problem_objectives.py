# pyre-strict
"""Dual and primal objectives derived from an OutcomeModel."""

import math
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from maximin.confidence_regions import Ellipsoid
from maximin.outcome_models import MatrixGame


class DualObjective(ABC):
    r"""Abstract base for the dual objective :math:`h(c)`.

    The dual objective aggregates over the uncertainty set:

    .. math::

        h(c) = \min_{\beta \in S} g(c;\, \beta).

    By the minimax theorem, maximizing :math:`h` over :math:`c \in C`
    yields the maximin value.
    """

    @abstractmethod
    def evaluate(self, c: npt.NDArray[np.float64]) -> float:
        r"""Evaluate :math:`h(c) = \min_{\beta \in S} g(c; \beta)`.

        Parameters
        ----------
        c : npt.NDArray[np.float64]
            Decision variable, shape ``(m,)``.

        Returns
        -------
        float
            Worst-case objective value.
        """

    @abstractmethod
    def grad_c(
        self, c: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Gradient (or supergradient) of :math:`h` with respect to ``c``.

        By the envelope theorem,
        :math:`\nabla_c h(c) = \nabla_c g(c;\, \beta^*(c))`
        when :math:`\beta^*(c)` is unique.

        Parameters
        ----------
        c : npt.NDArray[np.float64]
            Decision variable, shape ``(m,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Gradient, shape ``(m,)``.
        """

    @abstractmethod
    def minimizer(
        self, c: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Return :math:`\beta^*(c) = \arg\min_{\beta \in S} g(c;\, \beta)`.

        Parameters
        ----------
        c : npt.NDArray[np.float64]
            Decision variable, shape ``(m,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Worst-case parameter, shape ``(n,)``.
        """


class PrimalObjective(ABC):
    r"""Abstract base for the primal objective :math:`f(\beta)`.

    The primal objective is the best response over the decision space:

    .. math::

        f(\beta) = \max_{c \in C} g(c;\, \beta).

    By the minimax theorem, minimizing :math:`f` over :math:`\beta \in S`
    yields the same maximin value as maximizing :math:`h`.
    """

    @abstractmethod
    def evaluate(self, beta: npt.NDArray[np.float64]) -> float:
        r"""Evaluate :math:`f(\beta) = \max_{c \in C} g(c;\, \beta)`.

        Parameters
        ----------
        beta : npt.NDArray[np.float64]
            Parameter vector, shape ``(n,)``.

        Returns
        -------
        float
            Best-case objective value.
        """

    @abstractmethod
    def grad_beta(
        self, beta: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Gradient (or subgradient) of :math:`f` with respect to ``beta``.

        By the envelope theorem,
        :math:`\nabla_\beta f(\beta) = \nabla_\beta g(c^*(\beta);\, \beta)`
        when :math:`c^*(\beta)` is unique.

        Parameters
        ----------
        beta : npt.NDArray[np.float64]
            Parameter vector, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Gradient, shape ``(n,)``.
        """

    @abstractmethod
    def maximizer(
        self, beta: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Return :math:`c^*(\beta) = \arg\max_{c \in C} g(c;\, \beta)`.

        Parameters
        ----------
        beta : npt.NDArray[np.float64]
            Parameter vector, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Best-response decision, shape ``(m,)``.
        """


class MatrixGameEllipsoidDualObjective(DualObjective):
    r"""Analytic dual objective for a matrix game with ellipsoidal uncertainty.

    For :math:`g(c;\, \beta) = c^\top A \beta` and the ellipsoidal
    uncertainty set

    .. math::

        S = \bigl\{ \beta :
            (\beta - \hat\beta)^\top \Sigma^{-1}
            (\beta - \hat\beta) \le 1 \bigr\},

    minimizing :math:`g` over :math:`S` at fixed :math:`c` amounts to
    minimizing a linear function over an ellipsoid, which has the
    closed form

    .. math::

        h(c) = c^\top A \hat\beta
               - \bigl\| \Sigma^{1/2} A^\top c \bigr\|_2.

    The worst-case parameter is

    .. math::

        \beta^*(c) = \hat\beta
                   - \frac{\Sigma A^\top c}
                     {\bigl\| \Sigma^{1/2} A^\top c \bigr\|_2},

    and by the envelope theorem
    :math:`\nabla_c h(c) = A \beta^*(c)`.

    Parameters
    ----------
    game : MatrixGame
        Bilinear outcome model with payoff matrix ``A``.
    region : Ellipsoid
        Ellipsoidal uncertainty set.

    Notes
    -----
    When :math:`A^\top c = 0`, the objective equals :math:`c^\top A\hat\beta`
    for every :math:`\beta \in S`. In that case :math:`\beta^*(c) = \hat\beta`
    is returned as a canonical choice.
    """

    def __init__(self, game: MatrixGame, region: Ellipsoid) -> None:
        if game.dim_beta != region.dim:
            raise ValueError(
                f"game.dim_beta ({game.dim_beta}) must equal "
                f"region.dim ({region.dim})"
            )
        self._game = game
        self._region = region

    def _at_c_quantities(
        self, c: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        r"""Compute :math:`A^\top c`, :math:`\Sigma A^\top c`, and
        :math:`\|\Sigma^{1/2} A^\top c\|_2`.

        Returns
        -------
        tuple
            ``(At_c, Sigma_At_c, norm)`` where ``norm`` is the scalar
            :math:`\|\Sigma^{1/2} A^\top c\|_2 \ge 0`.
        """
        A = self._game.A
        Sigma = self._region.Sigma
        At_c = A.T @ c  # shape (n,)
        Sigma_At_c = Sigma @ At_c  # shape (n,)
        # Numerically clamp to avoid negative radicand from rounding.
        norm = math.sqrt(max(float(np.dot(At_c, Sigma_At_c)), 0.0))
        return At_c, Sigma_At_c, norm

    def minimizer(
        self, c: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Return :math:`\beta^*(c) = \hat\beta - \Sigma A^\top c \,/\, \|\Sigma^{1/2} A^\top c\|`."""
        beta_hat = self._region.beta_hat
        _, Sigma_At_c, norm = self._at_c_quantities(c)
        if norm == 0.0:
            return beta_hat.copy()
        return beta_hat - Sigma_At_c / norm

    def evaluate(self, c: npt.NDArray[np.float64]) -> float:
        r"""Evaluate :math:`h(c) = c^\top A \hat\beta - \|\Sigma^{1/2} A^\top c\|`."""
        A = self._game.A
        beta_hat = self._region.beta_hat
        _, _, norm = self._at_c_quantities(c)
        return float(np.dot(c, A @ beta_hat)) - norm

    def grad_c(
        self, c: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Return :math:`A \beta^*(c)`, the gradient with respect to ``c``."""
        return self._game.A @ self.minimizer(c)
