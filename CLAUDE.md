# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run pytest test/                        # run all tests
uv run pytest test/test_solvers.py         # run one test file
uv run pytest test/test_solvers.py::TestProximalSubgradientDualSolver::test_known_optimum  # run one test
uv run ruff check maximin/                 # lint
uv run mypy maximin/                       # type-check
uv run black maximin/ test/               # format
```

## Architecture

The library solves maximin problems of the form:

```
maximize_c  minimize_beta  g(c; beta)
subject to  beta in S,  c in C
```

The five modules mirror the five roles in this structure:

- **`outcome_models.py`** — `OutcomeModel` ABC + `MatrixGame` (`g = c^T A beta`). Exposes `evaluate`, `grad_c`, `grad_beta`.
- **`decision_spaces.py`** — `DecisionSpace` ABC + `AllocationDecision` (`{c ≥ 0, Σcᵢ ≤ 1}`). Exposes `project` (Euclidean projection) and `contains`.
- **`confidence_regions.py`** — `ConfidenceRegion` ABC + `Ellipsoid` (`{beta: (beta-b̂)ᵀ Σ⁻¹ (beta-b̂) ≤ 1}`). Same interface as DecisionSpace. Projection uses bisection on a secular equation derived from the eigendecomposition of Σ.
- **`problem_objectives.py`** — `DualObjective` ABC (`h(c) = min_{beta∈S} g`) and `PrimalObjective` ABC (`f(beta) = max_{c∈C} g`). The concrete class `MatrixGameEllipsoidDualObjective` evaluates `h` analytically via `h(c) = c^T A b̂ − ‖Σ^{1/2} Aᵀc‖`. Both ABCs expose `evaluate`, a gradient method, and an optimizer method (`minimizer` / `maximizer`).
- **`solvers.py`** — `DualSolver` ABC (maximize `h` over C) and `PrimalSolver` ABC (minimize `f` over S), plus `ProximalSubgradientDualSolver` and `ProximalSubgradientPrimalSolver` using projected subgradient with diminishing step sizes `α₀/√t`. Results returned as frozen `SolverResult` dataclasses.

### Saddle point principle

By the minimax theorem, `max_c h(c) = min_beta f(beta)`, so either solver reaches the same maximin value. The dual path (maximize `h`) tends to be preferable when `h` has a closed form, as in the `MatrixGame` + `Ellipsoid` case.

### Adding new concrete classes

To add a new outcome model, decision space, or confidence region: subclass the corresponding ABC and implement `evaluate`/`project`/`contains`. To add a new analytic objective, subclass `DualObjective` or `PrimalObjective` and wire it to specific `OutcomeModel` and `ConfidenceRegion`/`DecisionSpace` types. Export new public names from `maximin/__init__.py`.

### Planned extensions (not yet implemented)

- Cvxium-backed `DualObjective` and `PrimalObjective` for general convex cases
- `AcceleratedProximalDualSolver` / `AcceleratedProximalPrimalSolver`
- SOCP `DualSolver` for the `MatrixGame` + `Ellipsoid` + `AllocationDecision` special case
