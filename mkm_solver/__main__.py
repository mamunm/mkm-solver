"""CLI entry point: python -m mkm_solver input.json -o mkm_results"""

import argparse
from pathlib import Path

from mkm_solver.logger import log
from mkm_solver.schema import load_model, load_lateral_interactions
from mkm_solver.solver import solve_steady_state
from mkm_solver.sensitivity import compute_sensitivity
from mkm_solver.output import write_all
from mkm_solver.plotting import plot_coverage_vs_time, plot_energy_diagram, plot_sensitivity


def main():
    parser = argparse.ArgumentParser(
        prog="mkm_solver",
        description="Microkinetic model solver for heterogeneous catalysis",
    )
    parser.add_argument("input", type=str, help="Path to JSON input file")
    parser.add_argument(
        "-o", "--output", type=str, default="mkm_results",
        help="Output directory (default: mkm_results)",
    )
    parser.add_argument(
        "--lateral", type=str, default=None,
        help="Path to lateral interactions JSON file",
    )
    parser.add_argument(
        "--method", type=str,
        choices=[
            "scipy-bdf", "scipy-lsoda", "scipy-radau",
            "sundials-cvode-bdf",
            "assimulo-cvode-bdf", "assimulo-radau5", "assimulo-ida-dae",
        ],
        help="Override ODE solver method",
    )
    parser.add_argument(
        "--ss-method", type=str, choices=["integration", "rootfinding"],
        help="Override steady-state method",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate plots (coverage vs time, energy diagram, sensitivity)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()
    out_dir = Path(args.output)

    log.info("Loading model...")
    model = load_model(args.input)

    if args.lateral:
        log.info(f"Loading lateral interactions from {args.lateral}")
        model.lateral_interactions = load_lateral_interactions(args.lateral)
        log.info(f"Lateral interactions: {list(model.lateral_interactions.keys())}")

    if args.method:
        model.solver.method = args.method
    if args.ss_method:
        model.solver.steady_state_method = args.ss_method

    log.info(
        f"Model '{model.model_name}': "
        f"{len(model.surface_species)} surface species, "
        f"{len(model.reactions)} reactions, "
        f"T = {model.temperature} K"
    )

    log.info(f"Solving (method={model.solver.method}, ss_method={model.solver.steady_state_method})")
    ss_result = solve_steady_state(model)

    if ss_result["converged"]:
        log.success("Steady state found")
    else:
        log.warning("Steady state NOT converged")

    sens_result = {}
    if model.sensitivity.enabled:
        log.info("Running sensitivity analysis...")
        sens_result = compute_sensitivity(model, ss_result)
        log.success("Sensitivity analysis complete")

    write_all(model, ss_result, sens_result, args.output)
    log.success(f"Results written to {out_dir}/")

    if args.plot:
        out_dir.mkdir(parents=True, exist_ok=True)

        sol = ss_result.get("sol")
        if hasattr(sol, "t") and hasattr(sol, "y"):
            cov_path = str(out_dir / "coverage_vs_time.png")
            plot_coverage_vs_time(sol, model, cov_path)
            log.info(f"Plot: {cov_path}")

        energy_path = str(out_dir / "energy_diagram.png")
        plot_energy_diagram(model, energy_path)
        log.info(f"Plot: {energy_path}")

        if sens_result:
            sens_path = str(out_dir / "sensitivity.png")
            plot_sensitivity(sens_result, model, sens_path)
            log.info(f"Plot: {sens_path}")

        log.success("All plots generated")


if __name__ == "__main__":
    main()
