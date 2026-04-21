# pyre-strict
"""Confidence regions S for the unknown parameter vector beta."""

import math
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import scipy.optimize


class ConfidenceRegion(ABC):
    r"""Abstract base for uncertainty sets :math:`S` over parameters.

    A ConfidenceRegion describes the set of plausible values for
    ``beta`` and provides Euclidean projection, enabling proximal
    gradient methods on the primal objective.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the parameter space."""

    @abstractmethod
    def project(
        self,
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Euclidean projection of ``beta`` onto :math:`S`.

        Parameters
        ----------
        beta : npt.NDArray[np.float64]
            Point to project, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Projected point, shape ``(n,)``.
        """

    @abstractmethod
    def contains(
        self,
        beta: npt.NDArray[np.float64],
        atol: float = 1e-9,
    ) -> bool:
        r"""Return True if ``beta`` lies in :math:`S`."""


class Hypercube(ConfidenceRegion):
    r"""Hypercube (box) confidence region.

    .. math::

        S = \bigl\{ \beta \in \mathbb{R}^n :
            \ell_i \le \beta_i \le u_i,\ i = 1,\dots,n \bigr\}

    Parameters
    ----------
    lo : npt.NDArray[np.float64]
        Lower bounds, shape ``(n,)``.
    hi : npt.NDArray[np.float64]
        Upper bounds, shape ``(n,)``.  Must satisfy ``lo[i] <= hi[i]``
        for all ``i``.
    """

    def __init__(
        self,
        lo: npt.NDArray[np.float64],
        hi: npt.NDArray[np.float64],
    ) -> None:
        if lo.ndim != 1:
            raise ValueError(f"lo must be 1-dimensional, got shape {lo.shape}")
        if hi.shape != lo.shape:
            raise ValueError(
                f"hi must have the same shape as lo ({lo.shape}), got {hi.shape}"
            )
        if np.any(lo > hi):
            raise ValueError("lo must be <= hi for every component")
        self._lo = lo.copy()
        self._hi = hi.copy()

    @property
    def dim(self) -> int:
        """Dimension of the parameter space."""
        return len(self._lo)

    @property
    def lo(self) -> npt.NDArray[np.float64]:
        """Lower bounds."""
        return self._lo.copy()

    @property
    def hi(self) -> npt.NDArray[np.float64]:
        """Upper bounds."""
        return self._hi.copy()

    def contains(
        self,
        beta: npt.NDArray[np.float64],
        atol: float = 1e-9,
    ) -> bool:
        r"""Return True if ``beta`` lies in :math:`S`."""
        return bool(np.all(beta >= self._lo - atol) and np.all(beta <= self._hi + atol))

    def project(
        self,
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Project ``beta`` onto :math:`S` by component-wise clamping.

        Parameters
        ----------
        beta : npt.NDArray[np.float64]
            Point to project, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Projected point, shape ``(n,)``.
        """
        return np.clip(beta, self._lo, self._hi)


class Ellipsoid(ConfidenceRegion):
    r"""Ellipsoidal confidence region.

    .. math::

        S = \bigl\{ \beta \in \mathbb{R}^n :
            (\beta - \hat\beta)^\top \Sigma^{-1}
            (\beta - \hat\beta) \le 1 \bigr\}

    Parameters
    ----------
    beta_hat : npt.NDArray[np.float64]
        Center of the ellipsoid, shape ``(n,)``.
    Sigma : npt.NDArray[np.float64]
        Positive definite shape matrix, shape ``(n, n)``. The axes of
        the ellipsoid align with the eigenvectors of :math:`\Sigma`,
        with half-lengths equal to the square roots of the eigenvalues.

    Notes
    -----
    The eigendecomposition :math:`\Sigma = Q D Q^\top` (with ``D``
    diagonal, eigenvalues sorted ascending) is computed once at
    initialization and reused for both :meth:`contains` and
    :meth:`project`.

    Projection onto the ellipsoid when a point lies outside requires
    finding :math:`\nu > 0` solving the secular equation

    .. math::

        \phi(\nu) \;=\; \sum_i \frac{d_i v_i^2}{(d_i + \nu)^2} = 1,

    where :math:`d_i` are the eigenvalues of :math:`\Sigma` and
    :math:`v = Q^\top(\beta - \hat\beta)`. The function :math:`\phi`
    is strictly decreasing from :math:`\phi(0) > 1` (since
    :math:`\beta` is outside) to :math:`0`, so the root is bracketed
    and found by bisection via :func:`scipy.optimize.brentq`.
    """

    def __init__(
        self,
        beta_hat: npt.NDArray[np.float64],
        Sigma: npt.NDArray[np.float64],
    ) -> None:
        if beta_hat.ndim != 1:
            raise ValueError(
                f"beta_hat must be 1-dimensional, got shape {beta_hat.shape}"
            )
        n = len(beta_hat)
        if Sigma.shape != (n, n):
            raise ValueError(f"Sigma must have shape ({n}, {n}), got {Sigma.shape}")
        d, Q = np.linalg.eigh(Sigma)
        if np.any(d <= 0.0):
            raise ValueError("Sigma must be positive definite")
        self._beta_hat = beta_hat.copy()
        self._Sigma = Sigma.copy()
        self._n = n
        self._d = d  # eigenvalues, shape (n,), ascending
        self._Q = Q  # eigenvectors, shape (n, n)

    @property
    def dim(self) -> int:
        """Dimension of the parameter space."""
        return self._n

    @property
    def beta_hat(self) -> npt.NDArray[np.float64]:
        """Center of the ellipsoid."""
        return self._beta_hat.copy()

    @property
    def Sigma(self) -> npt.NDArray[np.float64]:
        """Shape matrix."""
        return self._Sigma.copy()

    def _mahalanobis_sq(self, beta: npt.NDArray[np.float64]) -> float:
        r"""Return :math:`(\beta-\hat\beta)^\top \Sigma^{-1} (\beta-\hat\beta)`."""
        v = self._Q.T @ (beta - self._beta_hat)  # shape (n,)
        return float(np.dot(v, v / self._d))

    def contains(
        self,
        beta: npt.NDArray[np.float64],
        atol: float = 1e-9,
    ) -> bool:
        r"""Return True if ``beta`` lies in :math:`S`."""
        return self._mahalanobis_sq(beta) <= 1.0 + atol

    def project(
        self,
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Project ``beta`` onto the ellipsoid :math:`S`.

        Parameters
        ----------
        beta : npt.NDArray[np.float64]
            Point to project, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Projected point on or inside :math:`S`, shape ``(n,)``.

        Notes
        -----
        If ``beta`` is already inside the ellipsoid it is returned
        unchanged. Otherwise the KKT conditions give

        .. math::

            \beta^* = \hat\beta + Q\, \mathrm{diag}
                      \!\left(\frac{d_i}{d_i + \nu}\right) v,

        where :math:`\nu` solves the secular equation
        :math:`\phi(\nu) = 1`. The upper bracket for bisection follows
        from the bound :math:`\phi(\nu) \le \|d \odot v\|_2^2 / \nu^2`,
        which falls below 1 when :math:`\nu > \|d \odot v\|_2`.
        """
        d, Q = self._d, self._Q
        v = Q.T @ (beta - self._beta_hat)  # shape (n,)

        if self._mahalanobis_sq(beta) <= 1.0:
            return beta.copy()

        # Secular equation phi(nu) = sum_i d_i v_i^2 / (d_i + nu)^2 - 1.
        dv2 = d * v**2  # shape (n,)

        def phi(nu: float) -> float:
            return float(np.sum(dv2 / (d + nu) ** 2)) - 1.0

        # Upper bound: sum dv2 / nu^2 < 1 when nu > sqrt(sum dv2).
        nu_upper = 2.0 * math.sqrt(float(np.sum(dv2))) + 1.0
        nu_star: float = scipy.optimize.brentq(phi, 0.0, nu_upper)

        w = d * v / (d + nu_star)  # shape (n,)
        return self._beta_hat + Q @ w


class LogConcaveLikelihoodRegion(ConfidenceRegion, ABC):
    r"""Confidence region from inverting a log-concave likelihood ratio test.

    The region is:

    .. math::

        S = \bigl\{ \beta :
            2\bigl(\ell(\hat\beta) - \ell(\beta)\bigr) \le t \bigr\}

    where :math:`\ell` is the joint log-likelihood, :math:`\hat\beta` is the
    MLE, and :math:`t` is the LRT threshold.  Because the log-likelihood is
    concave, :math:`S` is convex.

    Parameters
    ----------
    threshold : float
        LRT threshold.  For a joint confidence region at level ``confidence``
        on ``m`` parameters use ``scipy.stats.chi2.ppf(confidence, df=m)``.

    Notes
    -----
    Subclasses must implement :attr:`beta_hat`, :meth:`log_likelihood`, and
    :attr:`dim`.  Overriding :meth:`grad_log_likelihood` with an analytic
    expression improves projection speed; the default uses forward finite
    differences.

    :meth:`project` solves

    .. math::

        \min_x \tfrac{1}{2}\|x - \beta\|^2 \quad\text{s.t.}\quad
        2(\ell(\hat\beta) - \ell(x)) \le t

    via SLSQP, starting from :math:`\hat\beta` (always feasible).
    """

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold
        self._cached_ll_mle: float | None = None

    @property
    @abstractmethod
    def beta_hat(self) -> npt.NDArray[np.float64]:
        """MLE / center of the confidence region."""

    @abstractmethod
    def log_likelihood(self, beta: npt.NDArray[np.float64]) -> float:
        """Joint log-likelihood at ``beta``; constant terms may be dropped."""

    def grad_log_likelihood(
        self, beta: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Gradient of the log-likelihood (forward finite differences fallback)."""
        eps = 1e-7
        f0 = self.log_likelihood(beta)
        grad = np.empty_like(beta)
        for i in range(len(beta)):
            beta_p = beta.copy()
            beta_p[i] += eps
            grad[i] = (self.log_likelihood(beta_p) - f0) / eps
        return grad

    @property
    def _opt_bounds(
        self,
    ) -> list[tuple[float | None, float | None]] | None:
        """Parameter-space bounds for the SLSQP optimizer. ``None`` = unbounded."""
        return None

    def _ll_at_mle(self) -> float:
        if self._cached_ll_mle is None:
            self._cached_ll_mle = self.log_likelihood(self.beta_hat)
        return self._cached_ll_mle

    def _lrt(self, beta: npt.NDArray[np.float64]) -> float:
        """LRT statistic: 2 * (ll(beta_hat) - ll(beta))."""
        return 2.0 * (self._ll_at_mle() - self.log_likelihood(beta))

    def contains(
        self,
        beta: npt.NDArray[np.float64],
        atol: float = 1e-9,
    ) -> bool:
        r"""Return True if ``beta`` lies in :math:`S`."""
        return bool(self._lrt(beta) <= self._threshold + atol)

    def project(
        self,
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Project ``beta`` onto :math:`S` via SLSQP.

        Parameters
        ----------
        beta : npt.NDArray[np.float64]
            Point to project, shape ``(m,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Point in :math:`S` closest to ``beta`` in Euclidean distance,
            shape ``(m,)``.
        """
        if self.contains(beta):
            return beta.copy()
        bounds = self._opt_bounds
        beta_hat = self.beta_hat
        lo: npt.NDArray[np.float64]
        hi: npt.NDArray[np.float64]
        if bounds is not None:
            lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
            hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
            beta_hat = np.clip(beta_hat, lo, hi)

        # Avoid starting from the exact MLE: grad_log_likelihood(MLE) = 0 makes
        # the constraint Jacobian degenerate, causing SLSQP to see spurious KKT
        # conditions and terminate early.  Binary-search for x0 strictly inside S
        # but closer to beta, so the constraint Jacobian is non-zero.
        t_lo, t_hi = 0.0, 1.0
        for _ in range(30):
            t_mid = (t_lo + t_hi) / 2.0
            x_mid = beta_hat + t_mid * (beta - beta_hat)
            if bounds is not None:
                x_mid = np.clip(x_mid, lo, hi)
            if self.contains(x_mid):
                t_lo = t_mid
            else:
                t_hi = t_mid
        x0 = beta_hat + t_lo * (beta - beta_hat)
        if bounds is not None:
            x0 = np.clip(x0, lo, hi)

        threshold = self._threshold

        def _constraint_val(x: npt.NDArray[np.float64]) -> float:
            return float(threshold - self._lrt(x))

        def _constraint_jac(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return 2.0 * self.grad_log_likelihood(x)

        result = scipy.optimize.minimize(
            fun=lambda x: 0.5 * float(np.dot(x - beta, x - beta)),
            jac=lambda x: x - beta,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": _constraint_val, "jac": _constraint_jac}
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        result_x = result.x

        # SLSQP's constraint satisfaction is ~1e-8; if the result is just
        # outside S, binary-search toward beta_hat (always feasible) to
        # recover a point that satisfies contains() with its default atol.
        if not self.contains(result_x):
            t_lo, t_hi = 0.0, 1.0
            for _ in range(50):
                t_mid = (t_lo + t_hi) / 2.0
                x_mid = (1.0 - t_mid) * result_x + t_mid * beta_hat
                if self.contains(x_mid):
                    t_hi = t_mid
                else:
                    t_lo = t_mid
            result_x = (1.0 - t_hi) * result_x + t_hi * beta_hat

        return result_x


class BinomialRegion(LogConcaveLikelihoodRegion):
    r"""Joint LRT confidence region for m independent Binomial proportions.

    Each component :math:`\beta_i \in (0,1)` is the success probability for
    distribution :math:`i`, with :math:`n_i` trials and :math:`k_i` observed
    successes.  The joint log-likelihood is

    .. math::

        \ell(\beta)
        = \sum_{i=1}^m \bigl[k_i\log\beta_i + (n_i-k_i)\log(1-\beta_i)\bigr]

    and the MLE is :math:`\hat\beta_i = k_i / n_i`.

    Parameters
    ----------
    n : npt.NDArray[np.float64]
        Trial counts, shape ``(m,)``. Must be positive.
    k : npt.NDArray[np.float64]
        Success counts, shape ``(m,)``. Must satisfy ``0 <= k[i] <= n[i]``.
    threshold : float
        LRT threshold.  Use ``scipy.stats.chi2.ppf(confidence, df=m)`` for a
        joint region at level ``confidence``.
    """

    def __init__(
        self,
        n: npt.NDArray[np.float64],
        k: npt.NDArray[np.float64],
        threshold: float,
    ) -> None:
        if n.ndim != 1:
            raise ValueError(f"n must be 1-dimensional, got shape {n.shape}")
        if k.shape != n.shape:
            raise ValueError(
                f"k must have the same shape as n ({n.shape}), got {k.shape}"
            )
        if np.any(n <= 0):
            raise ValueError("Trial counts n must be positive")
        if np.any(k < 0) or np.any(k > n):
            raise ValueError("Success counts k must satisfy 0 <= k[i] <= n[i]")
        super().__init__(threshold)
        self._n = n.copy()
        self._k = k.copy()
        self._m = len(n)
        self._bhat = k / n

    @property
    def dim(self) -> int:
        """Dimension of the parameter space."""
        return self._m

    @property
    def beta_hat(self) -> npt.NDArray[np.float64]:
        """MLE: component-wise success rates k / n."""
        return self._bhat.copy()

    def log_likelihood(self, beta: npt.NDArray[np.float64]) -> float:
        r"""Binomial joint log-likelihood (combinatorial constants omitted)."""
        # np.where avoids log(0) when k_i = 0 or n_i - k_i = 0.
        terms = np.where(self._k > 0, self._k * np.log(beta), 0.0)
        terms += np.where(
            self._n - self._k > 0, (self._n - self._k) * np.log1p(-beta), 0.0
        )
        return float(np.sum(terms))

    def grad_log_likelihood(
        self, beta: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Analytic gradient of the Binomial log-likelihood."""
        grad = np.where(self._k > 0, self._k / beta, 0.0)
        grad -= np.where(self._n - self._k > 0, (self._n - self._k) / (1.0 - beta), 0.0)
        return grad

    @property
    def _opt_bounds(self) -> list[tuple[float | None, float | None]]:
        eps = 1e-10
        return [(eps, 1.0 - eps)] * self._m


class PoissonRegion(LogConcaveLikelihoodRegion):
    r"""Joint LRT confidence region for m independent Poisson rates.

    Each component :math:`\beta_i > 0` is the rate for distribution :math:`i`,
    with :math:`n_i` observations summing to :math:`s_i`.  The joint
    log-likelihood is

    .. math::

        \ell(\beta) = \sum_{i=1}^m \bigl[s_i\log\beta_i - n_i\beta_i\bigr]

    and the MLE is :math:`\hat\beta_i = s_i / n_i`.

    Parameters
    ----------
    n : npt.NDArray[np.float64]
        Observation counts, shape ``(m,)``. Must be positive.
    x_sum : npt.NDArray[np.float64]
        Sum of observed counts per distribution, shape ``(m,)``.
        Must be non-negative.
    threshold : float
        LRT threshold.  Use ``scipy.stats.chi2.ppf(confidence, df=m)`` for a
        joint region at level ``confidence``.
    """

    def __init__(
        self,
        n: npt.NDArray[np.float64],
        x_sum: npt.NDArray[np.float64],
        threshold: float,
    ) -> None:
        if n.ndim != 1:
            raise ValueError(f"n must be 1-dimensional, got shape {n.shape}")
        if x_sum.shape != n.shape:
            raise ValueError(
                f"x_sum must have the same shape as n ({n.shape}), got {x_sum.shape}"
            )
        if np.any(n <= 0):
            raise ValueError("Observation counts n must be positive")
        if np.any(x_sum < 0):
            raise ValueError("x_sum must be non-negative")
        super().__init__(threshold)
        self._n = n.copy()
        self._x_sum = x_sum.copy()
        self._m = len(n)
        self._bhat = x_sum / n

    @property
    def dim(self) -> int:
        """Dimension of the parameter space."""
        return self._m

    @property
    def beta_hat(self) -> npt.NDArray[np.float64]:
        """MLE: component-wise sample means x_sum / n."""
        return self._bhat.copy()

    def log_likelihood(self, beta: npt.NDArray[np.float64]) -> float:
        r"""Poisson joint log-likelihood (factorial constants omitted)."""
        terms = np.where(self._x_sum > 0, self._x_sum * np.log(beta), 0.0)
        terms -= self._n * beta
        return float(np.sum(terms))

    def grad_log_likelihood(
        self, beta: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Analytic gradient of the Poisson log-likelihood."""
        grad = np.where(self._x_sum > 0, self._x_sum / beta, 0.0)
        grad -= self._n
        return grad

    @property
    def _opt_bounds(self) -> list[tuple[float | None, float | None]]:
        eps = 1e-10
        return [(eps, None)] * self._m


class GammaRegion(LogConcaveLikelihoodRegion):
    r"""Joint LRT confidence region for m independent Gamma rates.

    Each component :math:`\beta_i > 0` is the rate parameter for distribution
    :math:`i`, with known shape :math:`\alpha_i`, :math:`n_i` observations
    summing to :math:`s_i`.  The joint log-likelihood (dropping terms
    independent of :math:`\beta`) is

    .. math::

        \ell(\beta)
        = \sum_{i=1}^m \bigl[n_i\alpha_i\log\beta_i - \beta_i s_i\bigr]

    and the MLE is :math:`\hat\beta_i = n_i\alpha_i / s_i`.

    Parameters
    ----------
    alpha : npt.NDArray[np.float64]
        Known shape parameters, shape ``(m,)``. Must be positive.
    n : npt.NDArray[np.float64]
        Observation counts, shape ``(m,)``. Must be positive.
    x_sum : npt.NDArray[np.float64]
        Sum of observations per distribution, shape ``(m,)``. Must be positive.
    threshold : float
        LRT threshold.  Use ``scipy.stats.chi2.ppf(confidence, df=m)`` for a
        joint region at level ``confidence``.
    """

    def __init__(
        self,
        alpha: npt.NDArray[np.float64],
        n: npt.NDArray[np.float64],
        x_sum: npt.NDArray[np.float64],
        threshold: float,
    ) -> None:
        if alpha.ndim != 1:
            raise ValueError(f"alpha must be 1-dimensional, got shape {alpha.shape}")
        if n.shape != alpha.shape:
            raise ValueError(
                f"n must have the same shape as alpha ({alpha.shape}), got {n.shape}"
            )
        if x_sum.shape != alpha.shape:
            raise ValueError(
                f"x_sum must have the same shape as alpha ({alpha.shape}), "
                f"got {x_sum.shape}"
            )
        if np.any(alpha <= 0):
            raise ValueError("Shape parameters alpha must be positive")
        if np.any(n <= 0):
            raise ValueError("Observation counts n must be positive")
        if np.any(x_sum <= 0):
            raise ValueError("x_sum must be positive")
        super().__init__(threshold)
        self._alpha = alpha.copy()
        self._n = n.copy()
        self._x_sum = x_sum.copy()
        self._m = len(alpha)
        self._bhat = n * alpha / x_sum

    @property
    def dim(self) -> int:
        """Dimension of the parameter space."""
        return self._m

    @property
    def beta_hat(self) -> npt.NDArray[np.float64]:
        """MLE: component-wise rates n * alpha / x_sum."""
        return self._bhat.copy()

    def log_likelihood(self, beta: npt.NDArray[np.float64]) -> float:
        r"""Gamma joint log-likelihood (terms independent of :math:`\beta` omitted)."""
        return float(
            np.sum(self._n * self._alpha * np.log(beta) - beta * self._x_sum)
        )

    def grad_log_likelihood(
        self, beta: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Analytic gradient of the Gamma log-likelihood."""
        return self._n * self._alpha / beta - self._x_sum

    @property
    def _opt_bounds(self) -> list[tuple[float | None, float | None]]:
        eps = 1e-10
        return [(eps, None)] * self._m
