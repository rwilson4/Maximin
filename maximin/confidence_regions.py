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

    @abstractmethod
    def generalized_project(
        self,
        A: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Generalized projection onto :math:`S`.

        Solves

        .. math::

            \min_{\beta \in S} \|A\beta - v\|_2^2

        where ``A`` has shape ``(n, self.dim)`` and ``v`` has shape ``(n,)``.
        When ``A = np.eye(self.dim)``, this reduces exactly to
        :meth:`project` called with ``v``.

        Parameters
        ----------
        A : npt.NDArray[np.float64]
            Linear operator, shape ``(n, self.dim)``.
        v : npt.NDArray[np.float64]
            Target vector, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Optimal point in :math:`S`, shape ``(self.dim,)``.
        """


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

    def generalized_project(
        self,
        A: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Generalized projection onto the hypercube via L-BFGS-B.

        Parameters
        ----------
        A : npt.NDArray[np.float64]
            Linear operator, shape ``(n, self.dim)``.
        v : npt.NDArray[np.float64]
            Target vector, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Minimizer of :math:`\|A\beta - v\|^2` over :math:`S`,
            shape ``(self.dim,)``.

        Notes
        -----
        The objective :math:`\|A\beta - v\|^2` is convex quadratic; the
        box constraints are simple bounds.  L-BFGS-B maintains feasibility
        at every iterate, so no post-solve feasibility recovery is needed.
        The warm start is the unconstrained least-squares solution clipped
        to the box.
        """
        beta_ls = np.linalg.lstsq(A, v, rcond=None)[0]
        x0 = np.clip(beta_ls, self._lo, self._hi)

        def obj(x: npt.NDArray[np.float64]) -> float:
            r = A @ x - v
            return float(np.dot(r, r))

        def jac(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return 2.0 * (A.T @ (A @ x - v))

        result = scipy.optimize.minimize(
            fun=obj,
            jac=jac,
            x0=x0,
            method="L-BFGS-B",
            bounds=scipy.optimize.Bounds(self._lo, self._hi),
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000},
        )
        return np.asarray(result.x, dtype=np.float64)


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
        self._sqrt_d = np.sqrt(d)  # shape (n,); reused by generalized_project

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

    def generalized_project(
        self,
        A: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Generalized projection onto the ellipsoid via a trust-region secular
        equation.

        Parameters
        ----------
        A : npt.NDArray[np.float64]
            Linear operator, shape ``(n, self.dim)``.
        v : npt.NDArray[np.float64]
            Target vector, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Minimizer of :math:`\|A\beta - v\|^2` over :math:`S`,
            shape ``(self.dim,)``.

        Notes
        -----
        Let :math:`y = Q^\top(\beta - \hat\beta)` (eigenbasis coordinates).
        The constraint becomes :math:`\sum_i y_i^2 / d_i \le 1`.
        Substituting :math:`z_i = y_i / \sqrt{d_i}` maps the constraint to
        :math:`\|z\| \le 1` and the objective to :math:`\|Cz - w\|^2` where

        .. math::

            C = (A Q) \odot \sqrt{d}^\top, \quad w = v - A\hat\beta.

        This is a standard trust-region subproblem.  With thin SVD
        :math:`C = U S V^\top` and :math:`\phi = U^\top w`, the KKT
        conditions give :math:`z^*(\nu) = V \operatorname{diag}(s_i/(s_i^2 +
        \nu)) \phi` where :math:`\nu \ge 0` solves

        .. math::

            \psi(\nu) = \sum_i \frac{s_i^2 \phi_i^2}{(s_i^2 + \nu)^2} = 1.

        Recovery: :math:`\beta^* = \hat\beta + Q ((\sqrt{d}) \odot z^*)`.

        When :math:`A = I`, the secular equation reduces to the standard
        ellipsoid-projection secular equation and the result matches
        :meth:`project` exactly.
        """
        Q, sqrt_d = self._Q, self._sqrt_d
        beta_hat = self._beta_hat
        w = v - A @ beta_hat  # shape (n,)
        C = (A @ Q) * sqrt_d  # shape (n, dim); column j scaled by sqrt(d_j)

        U_C, s, Vt_C = np.linalg.svd(C, full_matrices=False)
        phi = U_C.T @ w  # shape (r,), r = min(n, dim)

        # Unconstrained least-squares solution in z-space.
        s_max = s[0] if len(s) > 0 else 0.0
        nonzero = s > 1e-12 * s_max
        xi_unc = np.where(nonzero, phi / np.where(nonzero, s, 1.0), 0.0)
        z_unc = Vt_C.T @ xi_unc  # shape (dim,)

        if float(np.dot(z_unc, z_unc)) <= 1.0:
            y = sqrt_d * z_unc
            return np.asarray(beta_hat + Q @ y, dtype=np.float64)

        # Solve secular equation psi(nu) = sum_i (s_i phi_i)^2 / (s_i^2 + nu)^2 = 1.
        s_phi = s * phi  # shape (r,)

        def psi(nu: float) -> float:
            return float(np.sum((s_phi / (s**2 + nu)) ** 2)) - 1.0

        # Upper bound: psi(nu) <= ||s_phi||^2 / nu^2 < 1 when nu > ||s_phi||.
        nu_upper = 2.0 * math.sqrt(float(np.dot(s_phi, s_phi))) + 1.0
        nu_star: float = scipy.optimize.brentq(psi, 0.0, nu_upper)

        xi = s_phi / (s**2 + nu_star)
        z = Vt_C.T @ xi
        y = sqrt_d * z
        return np.asarray(beta_hat + Q @ y, dtype=np.float64)


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

    def generalized_project(
        self,
        A: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Generalized projection onto :math:`S` via SLSQP.

        Parameters
        ----------
        A : npt.NDArray[np.float64]
            Linear operator, shape ``(n, self.dim)``.
        v : npt.NDArray[np.float64]
            Target vector, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Minimizer of :math:`\|A\beta - v\|^2` over :math:`S`,
            shape ``(self.dim,)``.
        """
        bounds = self._opt_bounds
        beta_hat = self.beta_hat
        lo: npt.NDArray[np.float64]
        hi: npt.NDArray[np.float64]
        if bounds is not None:
            lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
            hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
            beta_hat = np.clip(beta_hat, lo, hi)

        # Unconstrained least-squares warm start, clipped to parameter domain.
        x_target: npt.NDArray[np.float64] = np.asarray(
            np.linalg.lstsq(A, v, rcond=None)[0], dtype=np.float64
        )
        if bounds is not None:
            x_target = np.clip(x_target, lo, hi)

        if self.contains(x_target):
            return x_target

        # Binary-search for a starting point strictly inside S.
        t_lo, t_hi = 0.0, 1.0
        for _ in range(30):
            t_mid = (t_lo + t_hi) / 2.0
            x_mid = beta_hat + t_mid * (x_target - beta_hat)
            if bounds is not None:
                x_mid = np.clip(x_mid, lo, hi)
            if self.contains(x_mid):
                t_lo = t_mid
            else:
                t_hi = t_mid
        x0 = beta_hat + t_lo * (x_target - beta_hat)
        if bounds is not None:
            x0 = np.clip(x0, lo, hi)

        threshold = self._threshold

        def _constraint_val(x: npt.NDArray[np.float64]) -> float:
            return float(threshold - self._lrt(x))

        def _constraint_jac(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return 2.0 * self.grad_log_likelihood(x)

        result = scipy.optimize.minimize(
            fun=lambda x: float(np.dot(A @ x - v, A @ x - v)),
            jac=lambda x: 2.0 * (A.T @ (A @ x - v)),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": _constraint_val, "jac": _constraint_jac}
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        result_x: npt.NDArray[np.float64] = result.x

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
        return float(np.sum(self._n * self._alpha * np.log(beta) - beta * self._x_sum))

    def grad_log_likelihood(
        self, beta: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Analytic gradient of the Gamma log-likelihood."""
        return self._n * self._alpha / beta - self._x_sum

    @property
    def _opt_bounds(self) -> list[tuple[float | None, float | None]]:
        eps = 1e-10
        return [(eps, None)] * self._m


class CriterionRegion(ConfidenceRegion, ABC):
    r"""Confidence region defined by a sublevel set of a convex criterion loss.

    The region is:

    .. math::

        S = \bigl\{ \beta :
            L(\beta) \le L(\hat\beta) + t \bigr\}

    where :math:`L` is a convex loss, :math:`\hat\beta` is the loss minimizer,
    and :math:`t` is the threshold.  Because :math:`L` is convex, :math:`S`
    is convex.

    Parameters
    ----------
    threshold : float
        Excess-loss threshold ``t >= 0``.

    Notes
    -----
    Subclasses must implement :attr:`beta_hat`, :meth:`loss`, and :attr:`dim`.
    Overriding :meth:`grad_loss` with an analytic expression improves projection
    speed; the default uses forward finite differences.

    :meth:`project` solves

    .. math::

        \min_x \tfrac{1}{2}\|x - \beta\|^2 \quad\text{s.t.}\quad
        L(x) \le L(\hat\beta) + t

    via SLSQP, starting from a point strictly inside :math:`S` found by
    bisection along the segment from :math:`\hat\beta` to ``beta``.
    """

    def __init__(self, threshold: float) -> None:
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        self._threshold = threshold
        self._cached_loss_at_hat: float | None = None

    @property
    @abstractmethod
    def beta_hat(self) -> npt.NDArray[np.float64]:
        """Loss minimizer / center of the confidence region."""

    @abstractmethod
    def loss(self, beta: npt.NDArray[np.float64]) -> float:
        """Convex criterion loss at ``beta``."""

    def grad_loss(self, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gradient of the loss (forward finite differences fallback)."""
        eps = 1e-7
        f0 = self.loss(beta)
        grad = np.empty_like(beta)
        for i in range(len(beta)):
            beta_p = beta.copy()
            beta_p[i] += eps
            grad[i] = (self.loss(beta_p) - f0) / eps
        return grad

    @property
    def _opt_bounds(
        self,
    ) -> list[tuple[float | None, float | None]] | None:
        """Parameter-space bounds for the SLSQP optimizer. ``None`` = unbounded."""
        return None

    def _loss_at_hat(self) -> float:
        if self._cached_loss_at_hat is None:
            self._cached_loss_at_hat = self.loss(self.beta_hat)
        return self._cached_loss_at_hat

    def _excess(self, beta: npt.NDArray[np.float64]) -> float:
        """Excess loss L(beta) - L(beta_hat).  Non-negative by convexity."""
        return self.loss(beta) - self._loss_at_hat()

    def contains(
        self,
        beta: npt.NDArray[np.float64],
        atol: float = 1e-9,
    ) -> bool:
        r"""Return True if ``beta`` lies in :math:`S`."""
        return bool(self._excess(beta) <= self._threshold + atol)

    def project(
        self,
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Project ``beta`` onto :math:`S` via SLSQP.

        Parameters
        ----------
        beta : npt.NDArray[np.float64]
            Point to project, shape ``(p,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Point in :math:`S` closest to ``beta`` in Euclidean distance,
            shape ``(p,)``.
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

        # Avoid starting from the exact minimizer: grad_loss(beta_hat) = 0 makes
        # the constraint Jacobian degenerate, causing SLSQP to terminate early.
        # Binary-search for x0 strictly inside S but closer to beta.
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
            return float(threshold - self._excess(x))

        def _constraint_jac(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return -self.grad_loss(x)

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

    def generalized_project(
        self,
        A: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Generalized projection onto :math:`S` via SLSQP.

        Parameters
        ----------
        A : npt.NDArray[np.float64]
            Linear operator, shape ``(n, self.dim)``.
        v : npt.NDArray[np.float64]
            Target vector, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Minimizer of :math:`\|A\beta - v\|^2` over :math:`S`,
            shape ``(self.dim,)``.
        """
        bounds = self._opt_bounds
        beta_hat = self.beta_hat
        lo: npt.NDArray[np.float64]
        hi: npt.NDArray[np.float64]
        if bounds is not None:
            lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
            hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
            beta_hat = np.clip(beta_hat, lo, hi)

        x_target: npt.NDArray[np.float64] = np.asarray(
            np.linalg.lstsq(A, v, rcond=None)[0], dtype=np.float64
        )
        if bounds is not None:
            x_target = np.clip(x_target, lo, hi)

        if self.contains(x_target):
            return x_target

        t_lo, t_hi = 0.0, 1.0
        for _ in range(30):
            t_mid = (t_lo + t_hi) / 2.0
            x_mid = beta_hat + t_mid * (x_target - beta_hat)
            if bounds is not None:
                x_mid = np.clip(x_mid, lo, hi)
            if self.contains(x_mid):
                t_lo = t_mid
            else:
                t_hi = t_mid
        x0 = beta_hat + t_lo * (x_target - beta_hat)
        if bounds is not None:
            x0 = np.clip(x0, lo, hi)

        threshold = self._threshold

        def _constraint_val(x: npt.NDArray[np.float64]) -> float:
            return float(threshold - self._excess(x))

        def _constraint_jac(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return -self.grad_loss(x)

        result = scipy.optimize.minimize(
            fun=lambda x: float(np.dot(A @ x - v, A @ x - v)),
            jac=lambda x: 2.0 * (A.T @ (A @ x - v)),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": _constraint_val, "jac": _constraint_jac}
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        result_x: npt.NDArray[np.float64] = result.x

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


class HuberCriterionRegion(CriterionRegion):
    r"""Confidence region defined by the sublevel set of the Huber loss.

    The region is:

    .. math::

        S = \bigl\{ \beta :
            L(\beta) \le L(\hat\beta) + t \bigr\}

    where the Huber loss is

    .. math::

        L(\beta) = \sum_{i=1}^n \rho_\delta(y_i - x_i^\top \beta),
        \quad
        \rho_\delta(r) = \begin{cases}
            r^2/2 & |r| \le \delta, \\
            \delta|r| - \delta^2/2 & |r| > \delta,
        \end{cases}

    :math:`\hat\beta` is the Huber M-estimator (minimizer of :math:`L`),
    computed once via L-BFGS-B and cached, and :math:`t` is the
    excess-loss threshold.

    Parameters
    ----------
    X : npt.NDArray[np.float64]
        Design matrix, shape ``(n, p)``.
    y : npt.NDArray[np.float64]
        Response vector, shape ``(n,)``.
    delta : float
        Huber robustness parameter ``delta > 0``.  Residuals larger than
        ``delta`` are penalized linearly rather than quadratically.
    threshold : float
        Excess-loss threshold ``t >= 0``.
    """

    def __init__(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        delta: float,
        threshold: float,
    ) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows, "
                f"got {X.shape[0]} and {y.shape[0]}"
            )
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        super().__init__(threshold)
        self._X = X.copy()
        self._y = y.copy()
        self._delta = delta
        self._p = X.shape[1]
        self._cached_beta_hat: npt.NDArray[np.float64] | None = None

    @property
    def dim(self) -> int:
        """Dimension of the parameter space (number of predictors)."""
        return self._p

    @property
    def beta_hat(self) -> npt.NDArray[np.float64]:
        """Huber M-estimator: minimizer of the Huber loss, computed lazily."""
        if self._cached_beta_hat is None:
            self._cached_beta_hat = self._compute_beta_hat()
        return self._cached_beta_hat.copy()

    def _compute_beta_hat(self) -> npt.NDArray[np.float64]:
        result = scipy.optimize.minimize(
            fun=self.loss,
            jac=self.grad_loss,
            x0=np.zeros(self._p),
            method="L-BFGS-B",
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000},
        )
        return result.x

    def loss(self, beta: npt.NDArray[np.float64]) -> float:
        r"""Huber loss :math:`\sum_i \rho_\delta(y_i - x_i^\top\beta)`."""
        r = self._y - self._X @ beta
        return float(
            np.sum(
                np.where(
                    np.abs(r) <= self._delta,
                    r**2 / 2.0,
                    self._delta * np.abs(r) - self._delta**2 / 2.0,
                )
            )
        )

    def grad_loss(self, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Gradient of the Huber loss: :math:`-X^\top \psi_\delta(y - X\beta)`.

        The influence function is
        :math:`\psi_\delta(r) = r` if :math:`|r| \le \delta`, else
        :math:`\delta \operatorname{sign}(r)`.
        """
        r = self._y - self._X @ beta
        psi = np.where(np.abs(r) <= self._delta, r, self._delta * np.sign(r))
        return -self._X.T @ psi


class ModelDrift(ConfidenceRegion, ABC):
    r"""Abstract base for drift-augmented confidence regions.

    Wraps an existing confidence region :math:`S` and adds a model-drift
    constraint: the true future parameter :math:`\gamma` must satisfy
    :math:`D(\beta, \gamma) \le \varepsilon` for *some*
    :math:`\beta \in S`.  The resulting region is

    .. math::

        T = \bigl\{ \gamma :
            \min_{\beta \in S} D(\beta, \gamma) \le \varepsilon \bigr\}

    which is the :math:`\varepsilon`-expansion of :math:`S` under metric
    :math:`D`.  Because :math:`T` satisfies the :class:`ConfidenceRegion`
    interface it composes transparently with all existing solvers.

    Parameters
    ----------
    region : ConfidenceRegion
        The base confidence region :math:`S` for the historical parameter
        :math:`\beta`.
    epsilon : float
        Drift tolerance :math:`\varepsilon \ge 0`.
    """

    def __init__(self, region: ConfidenceRegion, epsilon: float) -> None:
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        self._region = region
        self._epsilon = epsilon

    @property
    def dim(self) -> int:
        """Dimension of the parameter space."""
        return self._region.dim

    @property
    def epsilon(self) -> float:
        """Drift tolerance."""
        return self._epsilon

    @abstractmethod
    def _min_distance_and_grad(
        self,
        gamma: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]:
        r"""Minimum distance from :math:`\gamma` to :math:`S` and its gradient.

        Returns
        -------
        tuple[float, npt.NDArray[np.float64]]
            ``(d, g)`` where ``d = min_{beta in S} D(beta, gamma)`` and
            ``g = grad_gamma d`` (via the envelope theorem).
        """

    def contains(
        self,
        gamma: npt.NDArray[np.float64],
        atol: float = 1e-9,
    ) -> bool:
        r"""Return True if ``gamma`` lies in :math:`T`."""
        dist, _ = self._min_distance_and_grad(gamma)
        return bool(dist <= self._epsilon + atol)

    def project(
        self,
        gamma: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Euclidean projection of ``gamma`` onto :math:`T` via SLSQP.

        Any point in :math:`S` is feasible for :math:`T` (since
        :math:`D(x, x) = 0 \le \varepsilon`), so ``region.project(gamma)``
        always provides a valid warm start.

        Parameters
        ----------
        gamma : npt.NDArray[np.float64]
            Point to project, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Projected point in :math:`T`, shape ``(n,)``.
        """
        if self.contains(gamma):
            return gamma.copy()

        x0 = self._region.project(gamma)
        epsilon = self._epsilon

        def _constraint_val(x: npt.NDArray[np.float64]) -> float:
            return float(epsilon - self._min_distance_and_grad(x)[0])

        def _constraint_jac(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return -self._min_distance_and_grad(x)[1]

        result = scipy.optimize.minimize(
            fun=lambda x: 0.5 * float(np.dot(x - gamma, x - gamma)),
            jac=lambda x: x - gamma,
            x0=x0,
            method="SLSQP",
            constraints=[
                {"type": "ineq", "fun": _constraint_val, "jac": _constraint_jac}
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        result_x: npt.NDArray[np.float64] = result.x

        # Post-SLSQP feasibility recovery: bisect toward x0 (always feasible),
        # mirroring the pattern in LogConcaveLikelihoodRegion.project.
        if not self.contains(result_x):
            t_lo, t_hi = 0.0, 1.0
            for _ in range(50):
                t_mid = (t_lo + t_hi) / 2.0
                x_mid = (1.0 - t_mid) * result_x + t_mid * x0
                if self.contains(x_mid):
                    t_hi = t_mid
                else:
                    t_lo = t_mid
            result_x = (1.0 - t_hi) * result_x + t_hi * x0

        return result_x

    def generalized_project(
        self,
        A: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Generalized projection onto :math:`T` via SLSQP.

        Parameters
        ----------
        A : npt.NDArray[np.float64]
            Linear operator, shape ``(n, self.dim)``.
        v : npt.NDArray[np.float64]
            Target vector, shape ``(n,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Minimizer of :math:`\|A\gamma - v\|^2` over :math:`T`,
            shape ``(self.dim,)``.

        Notes
        -----
        The warm start is ``region.project(lstsq_solution)``, which lies
        in :math:`S` and hence has zero distance to :math:`S`, so it is
        always feasible for :math:`T`.
        """
        x_target: npt.NDArray[np.float64] = np.asarray(
            np.linalg.lstsq(A, v, rcond=None)[0], dtype=np.float64
        )
        if self.contains(x_target):
            return x_target

        x0 = self._region.project(x_target)
        epsilon = self._epsilon

        def _constraint_val(x: npt.NDArray[np.float64]) -> float:
            return float(epsilon - self._min_distance_and_grad(x)[0])

        def _constraint_jac(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return -self._min_distance_and_grad(x)[1]

        result = scipy.optimize.minimize(
            fun=lambda x: float(np.dot(A @ x - v, A @ x - v)),
            jac=lambda x: 2.0 * (A.T @ (A @ x - v)),
            x0=x0,
            method="SLSQP",
            constraints=[
                {"type": "ineq", "fun": _constraint_val, "jac": _constraint_jac}
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        result_x: npt.NDArray[np.float64] = result.x

        if not self.contains(result_x):
            t_lo, t_hi = 0.0, 1.0
            for _ in range(50):
                t_mid = (t_lo + t_hi) / 2.0
                x_mid = (1.0 - t_mid) * result_x + t_mid * x0
                if self.contains(x_mid):
                    t_hi = t_mid
                else:
                    t_lo = t_mid
            result_x = (1.0 - t_hi) * result_x + t_hi * x0

        return result_x


class EuclideanModelDrift(ModelDrift):
    r"""Drift region with Euclidean distance.

    .. math::

        T = \bigl\{ \gamma :
            \operatorname{dist}(\gamma, S) \le \varepsilon \bigr\}
          = S \oplus B(0, \varepsilon)

    where :math:`\oplus` is the Minkowski sum and
    :math:`\operatorname{dist}(\gamma, S) = \|\gamma - \pi_S(\gamma)\|_2`.

    For convex :math:`S`, both :meth:`contains` and :meth:`project` have
    efficient closed forms via a single call to ``region.project()``.

    Parameters
    ----------
    region : ConfidenceRegion
        The base confidence region :math:`S`.
    epsilon : float
        Drift radius :math:`\varepsilon \ge 0`.
    """

    def _min_distance_and_grad(
        self,
        gamma: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]:
        beta_star = self._region.project(gamma)
        diff = gamma - beta_star
        dist = float(np.linalg.norm(diff))
        if dist < 1e-12:
            return 0.0, np.zeros_like(gamma)
        return dist, diff / dist

    def contains(
        self,
        gamma: npt.NDArray[np.float64],
        atol: float = 1e-9,
    ) -> bool:
        r"""Return True if ``dist(gamma, S) <= epsilon``."""
        beta_star = self._region.project(gamma)
        return bool(np.linalg.norm(gamma - beta_star) <= self._epsilon + atol)

    def project(
        self,
        gamma: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Project ``gamma`` onto :math:`S \oplus B(0, \varepsilon)`.

        For exterior points the formula is
        :math:`\pi_S(\gamma) + \varepsilon \,
        (\gamma - \pi_S(\gamma)) / \|\gamma - \pi_S(\gamma)\|`,
        which follows from minimizing
        :math:`(\|\gamma - \beta\| - \varepsilon)^2` over :math:`\beta \in S`.
        """
        beta_star = self._region.project(gamma)
        diff = gamma - beta_star
        dist = float(np.linalg.norm(diff))
        if dist <= self._epsilon:
            return gamma.copy()
        return np.asarray(beta_star + self._epsilon * diff / dist, dtype=np.float64)
