# pyre-strict
"""Decision spaces C for the decision variable c."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class DecisionSpace(ABC):
    r"""Abstract base for feasible sets :math:`C` of the decision variable.

    A DecisionSpace describes the constraint set for ``c`` and provides
    Euclidean projection, enabling proximal gradient methods.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the decision space."""

    @abstractmethod
    def project(
        self,
        c: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Euclidean projection of ``c`` onto :math:`C`.

        Parameters
        ----------
        c : npt.NDArray[np.float64]
            Point to project, shape ``(m,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Projected point, shape ``(m,)``.
        """

    @abstractmethod
    def contains(
        self,
        c: npt.NDArray[np.float64],
        atol: float = 1e-9,
    ) -> bool:
        r"""Return True if ``c`` lies in :math:`C`."""


def _project_onto_probability_simplex(
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Project ``y`` onto :math:`\Delta = \{c \ge 0 : \sum_i c_i = 1\}`.

    Parameters
    ----------
    y : npt.NDArray[np.float64]
        Input vector, shape ``(m,)``. Need not be non-negative.

    Returns
    -------
    npt.NDArray[np.float64]
        Projected vector on the probability simplex, shape ``(m,)``.

    Notes
    -----
    Uses the O(m log m) algorithm from Duchi et al. (2008). Sort
    ``y`` descending to find the threshold ``theta`` such that
    ``(y - theta)_+`` sums to one.

    References
    ----------
    Duchi, J., Shalev-Shwartz, S., Singer, Y., and Chandra, T. (2008).
    Efficient projections onto the L1-ball for learning in high
    dimensions. In *Proceedings of ICML*.
    """
    m = len(y)
    u = np.sort(y)[::-1]
    cumsum = np.cumsum(u)
    j = np.arange(1, m + 1, dtype=np.float64)
    rho = int(np.where(u > (cumsum - 1.0) / j)[0][-1])
    theta = (cumsum[rho] - 1.0) / float(rho + 1)
    return np.maximum(y - theta, 0.0)


class AllocationDecision(DecisionSpace):
    r"""Feasible set for budget-constrained allocation decisions.

    .. math::

        C = \bigl\{ c \in \mathbb{R}^m : c \ge 0,\;
            \textstyle\sum_i c_i \le 1 \bigr\}

    This is the probability simplex together with its interior: a
    decision-maker may allocate up to a unit budget across ``m``
    options, but need not spend it all.

    Parameters
    ----------
    m : int
        Number of allocation options.
    """

    def __init__(self, m: int) -> None:
        if m < 1:
            raise ValueError(f"Dimension m must be at least 1, got {m}")
        self._m = m

    @property
    def dim(self) -> int:
        """Number of allocation options."""
        return self._m

    def project(
        self,
        c: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Project ``c`` onto :math:`C`.

        Parameters
        ----------
        c : npt.NDArray[np.float64]
            Point to project, shape ``(m,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Projected point in :math:`C`, shape ``(m,)``.

        Notes
        -----
        Negative components are clipped to zero first. If the clipped
        vector satisfies the budget constraint it is returned directly;
        otherwise it is projected onto the probability simplex
        :math:`\{ c \ge 0 : \sum_i c_i = 1 \}` via the Duchi et al.
        (2008) algorithm.
        """
        c_plus = np.maximum(c, 0.0)
        if float(np.sum(c_plus)) <= 1.0:
            return c_plus
        return _project_onto_probability_simplex(c_plus)

    def contains(
        self,
        c: npt.NDArray[np.float64],
        atol: float = 1e-9,
    ) -> bool:
        r"""Return True if ``c`` lies in :math:`C`."""
        if np.any(c < -atol):
            return False
        if float(np.sum(c)) > 1.0 + atol:
            return False
        return True
