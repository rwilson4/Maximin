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
            raise ValueError(
                f"Sigma must have shape ({n}, {n}), got {Sigma.shape}"
            )
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
