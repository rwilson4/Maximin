# pyre-strict
"""Robust optimization: maximize worst-case objective over an uncertainty set."""

from maximin.confidence_regions import (
    BinomialRegion,
    ConfidenceRegion,
    CriterionRegion,
    Ellipsoid,
    GammaRegion,
    HuberCriterionRegion,
    Hypercube,
    LogConcaveLikelihoodRegion,
    PoissonRegion,
)
from maximin.decision_spaces import AllocationDecision, DecisionSpace
from maximin.outcome_models import CobbDouglas, MatrixGame, OutcomeModel
from maximin.problem_objectives import (
    CobbDouglasEllipsoidDualObjective,
    DefaultDualObjective,
    DefaultPrimalObjective,
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
    "BinomialRegion",
    "CobbDouglas",
    "CobbDouglasEllipsoidDualObjective",
    "ConfidenceRegion",
    "CriterionRegion",
    "DecisionSpace",
    "DefaultDualObjective",
    "DefaultPrimalObjective",
    "DualObjective",
    "DualSolver",
    "Ellipsoid",
    "GammaRegion",
    "HuberCriterionRegion",
    "Hypercube",
    "LogConcaveLikelihoodRegion",
    "MarkowitzSolver",
    "MatrixGame",
    "MatrixGameEllipsoidDualObjective",
    "MaximinLinearSolver",
    "OutcomeModel",
    "PoissonRegion",
    "PrimalObjective",
    "PrimalSolver",
    "ProximalSubgradientDualSolver",
    "ProximalSubgradientPrimalSolver",
    "SolverResult",
]
