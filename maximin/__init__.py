# pyre-strict
"""Robust optimization: maximize worst-case objective over an uncertainty set."""

from maximin.confidence_regions import (
    BinomialRegion,
    ConfidenceRegion,
    CriterionRegion,
    Ellipsoid,
    EuclideanModelDrift,
    GammaRegion,
    HuberCriterionRegion,
    Hypercube,
    LogConcaveLikelihoodRegion,
    ModelDrift,
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
from maximin.robust_constraints import (
    MatrixGameEllipsoidRobustConstraint,
    RobustConstraint,
)
from maximin.solvers import (
    AcceleratedProximalGradientDualSolver,
    ConstrainedMarkowitzSolver,
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
    "ConstrainedMarkowitzSolver",
    "CriterionRegion",
    "DecisionSpace",
    "DefaultDualObjective",
    "DefaultPrimalObjective",
    "DualObjective",
    "DualSolver",
    "Ellipsoid",
    "EuclideanModelDrift",
    "GammaRegion",
    "HuberCriterionRegion",
    "Hypercube",
    "LogConcaveLikelihoodRegion",
    "MarkowitzSolver",
    "MatrixGame",
    "MatrixGameEllipsoidDualObjective",
    "MatrixGameEllipsoidRobustConstraint",
    "MaximinLinearSolver",
    "ModelDrift",
    "OutcomeModel",
    "PoissonRegion",
    "PrimalObjective",
    "PrimalSolver",
    "ProximalSubgradientDualSolver",
    "ProximalSubgradientPrimalSolver",
    "RobustConstraint",
    "SolverResult",
]
