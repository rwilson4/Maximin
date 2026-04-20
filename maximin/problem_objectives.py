# pyre-strict
"""Dual and primal objectives derived from an OutcomeModel."""

import math
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from maximin.confidence_regions import ConfidenceRegion, Ellipsoid
from maximin.decision_spaces import DecisionSpace
from maximin.outcome_models import CobbDouglas, MatrixGame, OutcomeModel


class DualObjective(ABC):
    r"""Abstract base for the dual objective :math:`f(c)`.

    The dual objective aggregates over the uncertainty set:

    .. math::

        f(c) = \min_{\beta \in S} g(c;\, \beta).

    By the minimax theorem, maximizing :math:`f` over :math:`c \in C`
    yields the maximin value.
    """

    @abstractmethod
    def evaluate(self, c: npt.NDArray[np.float64]) -> float:
        r"""Evaluate :math:`f(c) = \min_{\beta \in S} g(c; \beta)`.

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
    def grad_c(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Gradient (or supergradient) of :math:`f` with respect to ``c``.

        By the envelope theorem,
        :math:`\nabla_c f(c) = \nabla_c g(c;\, \beta^*(c))`
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
    def minimizer(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
    r"""Abstract base for the primal objective :math:`h(\beta)`.

    The primal objective is the best response over the decision space:

    .. math::

        h(\beta) = \max_{c \in C} g(c;\, \beta).

    By the minimax theorem, minimizing :math:`h` over :math:`\beta \in S`
    yields the same maximin value as maximizing :math:`f`.
    """

    @abstractmethod
    def evaluate(self, beta: npt.NDArray[np.float64]) -> float:
        r"""Evaluate :math:`h(\beta) = \max_{c \in C} g(c;\, \beta)`.

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
    def grad_beta(self, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Gradient (or subgradient) of :math:`h` with respect to ``beta``.

        By the envelope theorem,
        :math:`\nabla_\beta h(\beta) = \nabla_\beta g(c^*(\beta);\, \beta)`
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
    def maximizer(self, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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

        f(c) = c^\top A \hat\beta
               - \bigl\| \Sigma^{1/2} A^\top c \bigr\|_2.

    The worst-case parameter is

    .. math::

        \beta^*(c) = \hat\beta
                   - \frac{\Sigma A^\top c}
                     {\bigl\| \Sigma^{1/2} A^\top c \bigr\|_2},

    and by the envelope theorem
    :math:`\nabla_c f(c) = A \beta^*(c)`.

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

    def minimizer(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Return :math:`\beta^*(c) = \hat\beta - \Sigma A^\top c \,/\, \|\Sigma^{1/2} A^\top c\|`."""
        beta_hat = self._region.beta_hat
        _, Sigma_At_c, norm = self._at_c_quantities(c)
        if norm == 0.0:
            return beta_hat.copy()
        return beta_hat - Sigma_At_c / norm

    def evaluate(self, c: npt.NDArray[np.float64]) -> float:
        r"""Evaluate :math:`f(c) = c^\top A \hat\beta - \|\Sigma^{1/2} A^\top c\|`."""
        A = self._game.A
        beta_hat = self._region.beta_hat
        _, _, norm = self._at_c_quantities(c)
        return float(np.dot(c, A @ beta_hat)) - norm

    def grad_c(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Return :math:`A \beta^*(c)`, the gradient with respect to ``c``."""
        return self._game.A @ self.minimizer(c)


class CobbDouglasEllipsoidDualObjective(DualObjective):
    r"""Closed-form dual objective for Cobb--Douglas with ellipsoidal uncertainty.

    For :math:`g(c;\,\beta) = e^{\beta_0}\prod_i(1+c_i)^{\beta_i}` and the
    ellipsoidal uncertainty set
    :math:`S = \{\beta:(\beta-\hat\beta)^\top\Sigma^{-1}(\beta-\hat\beta)\le 1\}`,

    the inner minimization has a closed form.  Writing
    :math:`v(c) = [1,\log(1+c_1),\dots,\log(1+c_m)]^\top` gives
    :math:`g = \exp(v^\top\beta)`, so minimizing :math:`g` over :math:`S` is
    equivalent to minimizing the linear function :math:`v^\top\beta` over the
    ellipsoid, which yields

    .. math::

        \beta^*(c) = \hat\beta
                   - \frac{\Sigma\,v(c)}{\sqrt{v(c)^\top\Sigma\,v(c)}},
        \qquad
        f(c) = \exp\!\Bigl(v(c)^\top\hat\beta
                          - \sqrt{v(c)^\top\Sigma\,v(c)}\Bigr).

    By the envelope theorem,
    :math:`\nabla_c f(c) = \nabla_c g(c;\,\beta^*(c))
    = f(c)\,\beta^*(c)_{1:m}\,/\,(1+c)`.

    Parameters
    ----------
    model : CobbDouglas
        Cobb--Douglas outcome model with ``m`` goods.
    region : Ellipsoid
        Ellipsoidal uncertainty set for ``beta``.
    """

    def __init__(self, model: CobbDouglas, region: Ellipsoid) -> None:
        if model.dim_beta != region.dim:
            raise ValueError(
                f"model.dim_beta ({model.dim_beta}) must equal "
                f"region.dim ({region.dim})"
            )
        self._model = model
        self._region = region

    def _v(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        v = np.empty(self._model.dim_beta)
        v[0] = 1.0
        v[1:] = np.log(self._model.delta + self._model.gamma * c)
        return v

    def _quantities(
        self, c: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
        r"""Compute ``v``, ``Sigma @ v``, ``||Sigma^{1/2} v||``, and ``v^T beta_hat``."""
        v = self._v(c)
        Sigma = self._region.Sigma
        Sigma_v = Sigma @ v
        norm = math.sqrt(max(float(np.dot(v, Sigma_v)), 0.0))
        a = float(np.dot(v, self._region.beta_hat))
        return v, Sigma_v, norm, a

    def minimizer(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Return :math:`\beta^*(c) = \hat\beta - \Sigma v\,/\,\|\Sigma^{1/2}v\|`."""
        _, Sigma_v, norm, _ = self._quantities(c)
        beta_hat = self._region.beta_hat
        if norm == 0.0:
            return beta_hat.copy()
        return beta_hat - Sigma_v / norm

    def evaluate(self, c: npt.NDArray[np.float64]) -> float:
        r"""Return :math:`f(c) = \exp(v^\top\hat\beta - \|\Sigma^{1/2}v\|)`."""
        _, _, norm, a = self._quantities(c)
        return float(math.exp(a - norm))

    def grad_c(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Return :math:`f(c)\,\gamma\,\beta^*(c)_{1:m}\,/\,(\delta+\gamma c)` by the envelope theorem."""
        beta_star = self.minimizer(c)
        gamma = self._model.gamma
        base = self._model.delta + gamma * c
        return self.evaluate(c) * gamma * beta_star[1:] / base


class DefaultDualObjective(DualObjective):
    r"""General dual objective via inner APG on :math:`\beta`.

    For any :class:`~maximin.outcome_models.OutcomeModel` and
    :class:`~maximin.confidence_regions.ConfidenceRegion`, approximates

    .. math::

        f(c) = \min_{\beta \in S} g(c;\, \beta)

    by running FISTA (Beck & Teboulle, 2009) projected gradient descent
    on :math:`\beta \mapsto g(c;\, \beta)` over :math:`S`.  The outer
    gradient is recovered by the envelope theorem:
    :math:`\nabla_c f(c) = \nabla_c g(c;\, \beta^*(c))`.

    When a closed-form dual is available (e.g.
    :class:`MatrixGameEllipsoidDualObjective`), prefer it for speed and
    precision.  This class is useful when no analytic formula exists.

    Parameters
    ----------
    model : OutcomeModel
        Outcome function providing ``evaluate``, ``grad_c``, and
        ``grad_beta``.
    region : ConfidenceRegion
        Uncertainty set :math:`S` for ``beta`` with a ``project`` method.
    max_iter : int
        Maximum FISTA iterations for each inner minimization call.
    tol : float
        Inner convergence tolerance on the iterate-change norm.
    step_size : float
        FISTA step size ``alpha``.  Must satisfy ``alpha <= 1/L`` where
        ``L`` is the Lipschitz constant of
        :math:`\nabla_\beta g(c;\, \cdot)`.  For
        :class:`~maximin.outcome_models.MatrixGame` the gradient is
        constant in ``beta`` (Lipschitz constant zero), so any positive
        value works.
    """

    def __init__(
        self,
        model: OutcomeModel,
        region: ConfidenceRegion,
        max_iter: int = 500,
        tol: float = 1e-8,
        step_size: float = 1e-2,
    ) -> None:
        if model.dim_beta != region.dim:
            raise ValueError(
                f"model.dim_beta ({model.dim_beta}) must equal "
                f"region.dim ({region.dim})"
            )
        self._model = model
        self._region = region
        self._max_iter = max_iter
        self._tol = tol
        self._step_size = step_size

    def minimizer(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Find :math:`\beta^*(c)` via FISTA descent on :math:`g(c;\,\cdot)` over :math:`S`.

        The FISTA iterates are

        .. math::

            \beta_{k+1} &= \Pi_S\!\bigl(y_k - \alpha\,\nabla_\beta g(c;\,y_k)\bigr), \\
            t_{k+1}     &= \tfrac{1 + \sqrt{1 + 4t_k^2}}{2}, \\
            y_{k+1}     &= \beta_{k+1}
                           + \tfrac{t_k-1}{t_{k+1}}(\beta_{k+1} - \beta_k).

        The best iterate (lowest :math:`g` value seen) is returned.
        """
        alpha = self._step_size
        beta = self._region.project(np.zeros(self._region.dim))
        y = beta.copy()
        t = 1.0
        best_beta = beta.copy()
        best_obj = self._model.evaluate(c, beta)

        for _ in range(self._max_iter):
            grad = self._model.grad_beta(c, y)
            beta_new = self._region.project(y - alpha * grad)

            obj = self._model.evaluate(c, beta_new)
            if obj < best_obj:
                best_obj = obj
                best_beta = beta_new.copy()

            if float(np.linalg.norm(beta_new - beta)) < self._tol:
                break

            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            y = beta_new + ((t - 1.0) / t_new) * (beta_new - beta)
            beta = beta_new
            t = t_new

        return best_beta

    def evaluate(self, c: npt.NDArray[np.float64]) -> float:
        r"""Return :math:`g(c;\, \beta^*(c))`."""
        return float(self._model.evaluate(c, self.minimizer(c)))

    def grad_c(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Return :math:`\nabla_c g(c;\, \beta^*(c))` by the envelope theorem."""
        return self._model.grad_c(c, self.minimizer(c))


class DefaultPrimalObjective(PrimalObjective):
    r"""General primal objective via inner APG on :math:`c`.

    For any :class:`~maximin.outcome_models.OutcomeModel` and
    :class:`~maximin.decision_spaces.DecisionSpace`, approximates

    .. math::

        h(\beta) = \max_{c \in C} g(c;\, \beta)

    by running FISTA (Beck & Teboulle, 2009) projected gradient **ascent**
    on :math:`c \mapsto g(c;\, \beta)` over :math:`C`.  The outer
    gradient is recovered by the envelope theorem:
    :math:`\nabla_\beta h(\beta) = \nabla_\beta g(c^*(\beta);\, \beta)`.

    When a closed-form primal is available, prefer it for speed and
    precision.  This class is useful when no analytic formula exists.

    Parameters
    ----------
    model : OutcomeModel
        Outcome function providing ``evaluate``, ``grad_c``, and
        ``grad_beta``.
    space : DecisionSpace
        Feasible set :math:`C` for ``c`` with a ``project`` method.
    max_iter : int
        Maximum FISTA iterations for each inner maximization call.
    tol : float
        Inner convergence tolerance on the iterate-change norm.
    step_size : float
        FISTA step size ``alpha``.  Must satisfy ``alpha <= 1/L`` where
        ``L`` is the Lipschitz constant of
        :math:`\nabla_c g(\cdot;\, \beta)`.  For
        :class:`~maximin.outcome_models.MatrixGame` the gradient is
        constant in ``c`` (Lipschitz constant zero), so any positive
        value works.
    """

    def __init__(
        self,
        model: OutcomeModel,
        space: DecisionSpace,
        max_iter: int = 500,
        tol: float = 1e-8,
        step_size: float = 1e-2,
    ) -> None:
        if model.dim_c != space.dim:
            raise ValueError(
                f"model.dim_c ({model.dim_c}) must equal "
                f"space.dim ({space.dim})"
            )
        self._model = model
        self._space = space
        self._max_iter = max_iter
        self._tol = tol
        self._step_size = step_size

    def maximizer(self, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Find :math:`c^*(\beta)` via FISTA ascent on :math:`g(\cdot;\,\beta)` over :math:`C`.

        The FISTA iterates are

        .. math::

            c_{k+1} &= \Pi_C\!\bigl(y_k + \alpha\,\nabla_c g(y_k;\,\beta)\bigr), \\
            t_{k+1} &= \tfrac{1 + \sqrt{1 + 4t_k^2}}{2}, \\
            y_{k+1} &= c_{k+1}
                       + \tfrac{t_k-1}{t_{k+1}}(c_{k+1} - c_k).

        The best iterate (highest :math:`g` value seen) is returned.
        """
        alpha = self._step_size
        c = self._space.project(np.zeros(self._space.dim))
        y = c.copy()
        t = 1.0
        best_c = c.copy()
        best_obj = self._model.evaluate(c, beta)

        for _ in range(self._max_iter):
            grad = self._model.grad_c(y, beta)
            c_new = self._space.project(y + alpha * grad)

            obj = self._model.evaluate(c_new, beta)
            if obj > best_obj:
                best_obj = obj
                best_c = c_new.copy()

            if float(np.linalg.norm(c_new - c)) < self._tol:
                break

            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            y = c_new + ((t - 1.0) / t_new) * (c_new - c)
            c = c_new
            t = t_new

        return best_c

    def evaluate(self, beta: npt.NDArray[np.float64]) -> float:
        r"""Return :math:`g(c^*(\beta);\, \beta)`."""
        return float(self._model.evaluate(self.maximizer(beta), beta))

    def grad_beta(self, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Return :math:`\nabla_\beta g(c^*(\beta);\, \beta)` by the envelope theorem."""
        return self._model.grad_beta(self.maximizer(beta), beta)
