# mkm-solver

Mean-field microkinetic model solver for heterogeneous catalysis on metal surfaces.

Developed for studying C-N coupling reaction mechanisms on mono- and bimetallic surfaces (Cu, Zn, Cu-Zn alloys) using DFT-derived energetics.

**Author:** Osman Mamun (mamun.che06@gmail.com)

## Installation

```bash
# Base install (scipy solvers)
pip install -e .

# With SUNDIALS support
conda install -c conda-forge sundials
pip install -e ".[sundials]"

# With Assimulo support (conda only)
conda install -c conda-forge assimulo
pip install -e .
```

## Usage

```bash
# Run with default settings (scipy-bdf)
mkm-solver experiments/Cu/cu.json -o experiments/Cu/results

# Override solver method
mkm-solver experiments/Cu/cu.json -o results --method assimulo-radau5

# With lateral interactions and plots
mkm-solver experiments/Cu/cu.json -o results --lateral lateral.json --plot
```

## Input Format

Main input JSON:

```json
{
  "model_name": "CN_coupling",
  "temperature": 298.15,
  "site_types": {"A": 1.812e19},
  "gas_species": {
    "H2_g": {"pressure": {"value": 1.0, "unit": "bar"}, "mass": 2.016}
  },
  "species_n_sites": {"LA*_A": 2},
  "reactions": [
    {
      "type": "adsorption",
      "reactants": {"H2_g": 1},
      "products": {"H*_A": 2},
      "Ea": 0.0,
      "DeltaE": -0.14
    },
    {
      "type": "surface",
      "reactants": {"N*_A": 1, "CO*_A": 1},
      "products": {"NCO*_A": 1},
      "Ea": 0.37,
      "DeltaE": -2.00
    }
  ],
  "solver": {
    "method": "scipy-bdf",
    "rtol": 1e-8,
    "atol": 1e-10,
    "t_span": [0, 1e6],
    "steady_state_tol": 1e-6
  }
}
```

- Free sites are auto-balanced from species site occupancy (not written in reactions).
- Surface species naming convention: `Name*_SiteType` (e.g., `NH3*_A`, `H*_B`).
- Site type inferred from species names; `species_n_sites` overrides default occupancy of 1.

Lateral interactions (separate JSON):

```json
{
  "H*_A": {"theta0": 0.12, "epsilon": 0.15}
}
```

## Rate Constants

- **Surface reactions (hTST):** k_f = (k_B T / h) exp(-E_a / k_B T), k_r = k_f / K_eq
- **Adsorption (collision theory):** k_f = 1 / (N_0 sqrt(2 pi m k_B T)), k_r = k_f / K_eq
- **Lateral interactions** modify K_eq only: E_eff = E_0 + epsilon * max(0, theta - theta_0)

## Solver Methods

| Method | Backend | Type | Install |
|---|---|---|---|
| `scipy-bdf` | scipy | ODE (BDF) | included |
| `scipy-lsoda` | scipy | ODE (auto stiff/nonstiff) | included |
| `scipy-radau` | scipy | ODE (implicit RK) | included |
| `sundials-cvode-bdf` | scikits.odes | ODE (SUNDIALS CVODE) | `pip install -e ".[sundials]"` |
| `assimulo-cvode-bdf` | Assimulo | ODE (SUNDIALS CVODE) | `conda install assimulo` |
| `assimulo-radau5` | Assimulo | ODE (Hairer Radau5) | `conda install assimulo` |
| `assimulo-ida-dae` | Assimulo | DAE (SUNDIALS IDA) | `conda install assimulo` |

The ODE formulation includes free-site coverages as state variables with dtheta_*/dt = -sum(n_i * dtheta_i/dt), matching MATLAB ode15s. The DAE formulation (IDA) enforces the site balance as a true algebraic constraint.

## Output

Results are written to a directory containing:

| File | Content |
|---|---|
| `model_equations.txt` | Reactions, rate expressions, site balances, ODEs |
| `rate_constants.json` | Base and lateral-corrected k_f, k_r, K_eq |
| `coverages.json` | Steady-state coverages and free sites |
| `rates.json` | Forward, reverse, net rates per reaction |
| `summary.txt` | Human-readable overview |
| `sensitivity.json` | DRC/DTRC (when enabled) |
| `*.png` | Plots (with `--plot`) |

## Project Structure

```
mkm_solver/
    schema.py        # Data model, JSON parser, validation
    validation.py    # Input validation rules
    kinetics.py      # Rate constant formulas (hTST, collision theory)
    build_mkm.py     # ODE system assembly, stoichiometry, site balance
    solver.py        # Integration/rootfinding with multiple backends
    output.py        # Result file writers
    plotting.py      # Coverage, energy diagram, sensitivity plots
    logger.py        # Rich console logging
    __main__.py      # CLI entry point
examples/
    cn_coupling.json # Example C-N coupling mechanism
experiments/
    Cu/              # Cu(111) C-N coupling
    Zn/              # Zn C-N coupling
    Cu3Zn/           # Cu3Zn alloy
    Zn3Cu/           # Zn3Cu alloy
```

