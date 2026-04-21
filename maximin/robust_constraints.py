# pyre-strict
"""Robust constraints: enforce r(c; gamma) >= 0 for all gamma in an uncertainty set."""

import math
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from maximin.confidence_regions import Ellipsoid


class RobustConstraint(ABC):
    r"""Abstract base for a robust constraint :math:`q(c) \ge 0`.

    The constraint requires

    .. math::

        r(c;\, \gamma) \ge 0 \quad \text{for all } \gamma \in T,

    which is equivalent to

    .. math::

        q(c) = \inf_{\gamma \in T} r(c;\, \gamma) \ge 0.

    When :math:`r` is concave in :math:`c` for each :math:`\gamma`, and
    convex in :math:`\gamma` for each :math:`c`, then :math:`q` is concave
    in :math:`c`, so :math:`q(c) \ge 0` is a convex constraint.
    """

    @property
    @abstractmethod
    def dim_c(self) -> int:
        """Dimension of the decision variable ``c``."""

    @abstractmethod
    def infimum(self, c: npt.NDArray[np.float64]) -> float:
        r"""Compute :math:`q(c) = \inf_{\gamma \in T} r(c;\, \gamma)`.

        Parameters
        ----------
        c : npt.NDArray[np.float64]
            Decision variable, shape ``(m,)``.

        Returns
        -------
        float
            Worst-case constraint value.
        """

    @abstractmethod
    def worst_case_gamma(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Return :math:`\gamma^*(c) = \arg\min_{\gamma \in T} r(c;\, \gamma)`.

        Parameters
        ----------
        c : npt.NDArray[np.float64]
            Decision variable, shape ``(m,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Worst-case parameter, shape ``(p,)``.
        """

    def is_satisfied(self, c: npt.NDArray[np.float64], atol: float = 0.0) -> bool:
        """Return ``True`` if :math:`q(c) \\ge -\\text{atol}`.

        Parameters
        ----------
        c : npt.NDArray[np.float64]
            Decision variable, shape ``(m,)``.
        atol : float
            Non-negative feasibility tolerance.
        """
        return self.infimum(c) >= -atol


class MatrixGameEllipsoidRobustConstraint(RobustConstraint):
    r"""Robust constraint for a matrix game with ellipsoidal uncertainty.

    Requires :math:`r(c;\, \gamma) = c^\top B \gamma \ge 0` for all
    :math:`\gamma` in the ellipsoid

    .. math::

        T = \bigl\{ \gamma :
            (\gamma - \hat\gamma)^\top \Sigma_T^{-1}
            (\gamma - \hat\gamma) \le 1 \bigr\}.

    The infimum has the closed form

    .. math::

        q(c) = c^\top B \hat\gamma
               - \bigl\| \Sigma_T^{1/2} B^\top c \bigr\|_2,

    and the worst-case parameter is

    .. math::

        \gamma^*(c) = \hat\gamma
                    - \frac{\Sigma_T B^\top c}
                      {\bigl\| \Sigma_T^{1/2} B^\top c \bigr\|_2}.

    The constraint :math:`q(c) \ge 0` is the second-order cone condition

    .. math::

        \bigl(c^\top B \hat\gamma,\; L_T^\top B^\top c\bigr)
        \in \mathrm{SOC}_{p+1},

    where :math:`\Sigma_T = L_T L_T^\top` is the Cholesky factorization.
    This form is returned by :meth:`socp_block` for direct use with Clarabel.

    Parameters
    ----------
    B : npt.NDArray[np.float64]
        Constraint payoff matrix, shape ``(m, p)``.
    region : Ellipsoid
        Ellipsoidal uncertainty set for :math:`\gamma`, with ``region.dim == p``.
    """

    def __init__(
        self,
        B: npt.NDArray[np.float64],
        region: Ellipsoid,
    ) -> None:
        if B.ndim != 2:
            raise ValueError(f"B must be a 2-D array, got ndim={B.ndim}")
        m, p = B.shape
        if p != region.dim:
            raise ValueError(f"B must have shape (m, {region.dim}), got {B.shape}")
        self._B = B.copy()
        self._region = region
        self._m = m
        self._p = p
        self._L_T = np.linalg.cholesky(region.Sigma)
        self._LT_T_BT = self._L_T.T @ B.T  # (p, m)
        self._B_gamma_hat = B @ region.beta_hat  # (m,)

    @property
    def dim_c(self) -> int:
        """Dimension of the decision variable ``c``."""
        return self._m

    def _at_c_quantities(
        self, c: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        r"""Compute :math:`B^\top c`, :math:`\Sigma_T B^\top c`, and
        :math:`\|\Sigma_T^{1/2} B^\top c\|_2`.

        Returns
        -------
        tuple
            ``(BT_c, Sigma_T_BT_c, norm)``.
        """
        BT_c = self._B.T @ c
        Sigma_T_BT_c = self._region.Sigma @ BT_c
        norm = math.sqrt(max(float(np.dot(BT_c, Sigma_T_BT_c)), 0.0))
        return BT_c, Sigma_T_BT_c, norm

    def infimum(self, c: npt.NDArray[np.float64]) -> float:
        r"""Return :math:`q(c) = c^\top B \hat\gamma - \|\Sigma_T^{1/2} B^\top c\|`."""
        _, _, norm = self._at_c_quantities(c)
        return float(np.dot(c, self._B_gamma_hat)) - norm

    def worst_case_gamma(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Return :math:`\gamma^*(c) = \hat\gamma - \Sigma_T B^\top c \,/\, \|\Sigma_T^{1/2} B^\top c\|`."""
        gamma_hat = self._region.beta_hat
        _, Sigma_T_BT_c, norm = self._at_c_quantities(c)
        if norm == 0.0:
            return gamma_hat.copy()
        return gamma_hat - Sigma_T_BT_c / norm

    def socp_block(self, m: int) -> tuple[sp.csc_matrix, int]:
        r"""Return the Clarabel constraint block enforcing :math:`q(c) \ge 0`.

        For the variable layout :math:`x = [c \;;\; t] \in \mathbb{R}^{m+1}`,
        returns a sparse matrix ``A_block`` of shape ``(p+1, m+1)`` such that

        .. code-block:: text

            s = b_block - A_block @ x  in  SOC_{p+1}

        with ``b_block = zeros(p+1)`` enforces
        :math:`c^\top B \hat\gamma \ge \|L_T^\top B^\top c\|_2`, i.e.
        :math:`q(c) \ge 0`.

        Parameters
        ----------
        m : int
            Number of decision variables.  Must equal ``self.dim_c``.

        Returns
        -------
        A_block : sp.csc_matrix
            Constraint matrix, shape ``(p+1, m+1)``.
        cone_size : int
            Size of the second-order cone, ``p+1``.
        """
        if m != self._m:
            raise ValueError(f"m={m} does not match constraint dim_c={self._m}")
        p = self._p

        # Row 0: scalar SOC component.
        # s[0] = 0 - (-(B gamma_hat)^T c + 0*t) = c^T B gamma_hat
        A_row0 = sp.hstack(
            [
                sp.csc_matrix(-self._B_gamma_hat[np.newaxis, :]),
                sp.csc_matrix((1, 1)),
            ],
            format="csc",
        )  # (1, m+1)

        # Rows 1..p: vector SOC component.
        # s[1:p+1] = 0 - (-(L_T^T B^T) c + 0*t) = L_T^T B^T c
        A_body = sp.hstack(
            [
                sp.csc_matrix(-self._LT_T_BT),
                sp.csc_matrix((p, 1)),
            ],
            format="csc",
        )  # (p, m+1)

        A_block = sp.vstack([A_row0, A_body], format="csc")  # (p+1, m+1)
        return A_block, p + 1
