# pyre-strict
"""Robust optimization: maximize worst-case objective over an uncertainty set."""

from maximin.confidence_regions import ConfidenceRegion, Ellipsoid
from maximin.decision_spaces import AllocationDecision, DecisionSpace
from maximin.outcome_models import MatrixGame, OutcomeModel
from maximin.problem_objectives import (
    DualObjective,
    MatrixGameEllipsoidDualObjective,
    PrimalObjective,
)
from maximin.solvers import (
    DualSolver,
    PrimalSolver,
    ProximalSubgradientDualSolver,
    ProximalSubgradientPrimalSolver,
    SolverResult,
)

__all__ = [
    "AllocationDecision",
    "ConfidenceRegion",
    "DecisionSpace",
    "DualObjective",
    "DualSolver",
    "Ellipsoid",
    "MatrixGame",
    "MatrixGameEllipsoidDualObjective",
    "OutcomeModel",
    "PrimalObjective",
    "PrimalSolver",
    "ProximalSubgradientDualSolver",
    "ProximalSubgradientPrimalSolver",
    "SolverResult",
]
