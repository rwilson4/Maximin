# pyre-strict
"""Robust optimization: maximize worst-case objective over an uncertainty set."""

from maximin.confidence_regions import ConfidenceRegion, Ellipsoid, Hypercube
from maximin.decision_spaces import AllocationDecision, DecisionSpace
from maximin.outcome_models import MatrixGame, OutcomeModel
from maximin.problem_objectives import (
    DualObjective,
    MatrixGameEllipsoidDualObjective,
    PrimalObjective,
)
from maximin.solvers import (
    AcceleratedProximalGradientDualSolver,
    DualSolver,
    MarkowitzSolver,
    MaximinLinearSolver,
    PrimalSolver,
    ProximalSubgradientDualSolver,
    ProximalSubgradientPrimalSolver,
    SolverResult,
)

__all__ = [
    "AcceleratedProximalGradientDualSolver",
    "AllocationDecision",
    "ConfidenceRegion",
    "DecisionSpace",
    "DualObjective",
    "DualSolver",
    "Ellipsoid",
    "Hypercube",
    "MarkowitzSolver",
    "MatrixGame",
    "MatrixGameEllipsoidDualObjective",
    "MaximinLinearSolver",
    "OutcomeModel",
    "PrimalObjective",
    "PrimalSolver",
    "ProximalSubgradientDualSolver",
    "ProximalSubgradientPrimalSolver",
    "SolverResult",
]
