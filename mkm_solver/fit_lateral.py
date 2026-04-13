"""Fit Grabow's two-parameter lateral interaction model.

Grabow, L.; Hvolbaek, B.; Norskov, J. Top. Catal. 2010, 53, 298-310.

    E_integral(theta) = E0 * theta + epsilon * (theta - theta0) ** 2

where E_integral is the DFT integral energy, E0 is fixed to the adsorption
(average) energy at the lowest coverage in the data, and {theta0, epsilon}
are obtained by unweighted least-squares minimization.

Input file: two whitespace- or comma-separated columns
    col 1 = ML coverage (theta)
    col 2 = adsorption energy E_avg(theta) in eV

Internally the integral energy is reconstructed as E_int = E_avg * theta
before fitting.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from mkm_solver.logger import log


def _grabow_integral(theta, E0, theta0, epsilon):
    theta = np.asarray(theta)
    return E0 * theta + epsilon * (theta - theta0) ** 2


def load_coverage_energy(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"Expected a two-column file (theta, energy); got shape {data.shape}"
        )
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def fit_grabow(theta, E_avg):
    theta = np.asarray(theta, dtype=float)
    E_avg = np.asarray(E_avg, dtype=float)
    if theta.size < 3:
        raise ValueError("Need at least 3 points to fit theta0 and epsilon")

    # E0: adsorption energy at the lowest coverage (theta sorted ascending).
    E0 = float(E_avg[0])

    # Grabow fits the integral energy: E_int = E_avg * theta.
    E_int = E_avg * theta

    # Initial guess: the deviation d(theta) = E_int - E0*theta should equal
    # epsilon*(theta - theta0)^2. A quadratic polyfit on d gives excellent
    # starting values for (theta0, epsilon).
    residual = E_int - E0 * theta
    a, b, _ = np.polyfit(theta, residual, 2)
    if abs(a) < 1e-12:
        theta0_0 = float(np.median(theta))
        epsilon_0 = 0.0
    else:
        epsilon_0 = float(a)
        theta0_0 = float(-b / (2.0 * a))

    def model(th, theta0, epsilon):
        return _grabow_integral(th, E0, theta0, epsilon)

    try:
        popt, pcov = curve_fit(
            model, theta, E_int, p0=[theta0_0, epsilon_0], maxfev=20000
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as exc:
        log.warning(f"curve_fit failed ({exc}); returning initial guess")
        popt = np.array([theta0_0, epsilon_0])
        perr = np.array([np.nan, np.nan])

    theta0_fit, epsilon_fit = float(popt[0]), float(popt[1])
    rmse = float(np.sqrt(np.mean((E_int - model(theta, *popt)) ** 2)))

    return {
        "E0": E0,
        "theta0": theta0_fit,
        "epsilon": epsilon_fit,
        "theta0_std": float(perr[0]),
        "epsilon_std": float(perr[1]),
        "rmse": rmse,
        "theta_ref": float(theta[0]),
        "n_points": int(theta.size),
    }


def main():
    parser = argparse.ArgumentParser(
        prog="mkm-fit-lateral",
        description=(
            "Fit Grabow's two-parameter lateral interaction model "
            "E_int(theta) = E0*theta + epsilon*(theta-theta0)^2 to a "
            "two-column (coverage, adsorption-energy) txt file. "
            "E0 is fixed to the adsorption energy at the lowest coverage."
        ),
    )
    parser.add_argument(
        "data_file", type=str,
        help="Two-column txt file: ML coverage and adsorption energy (eV)",
    )
    parser.add_argument(
        "--save-json", nargs="?", const="lat_out.json", default=None, metavar="PATH",
        help="Save fitted {theta0, epsilon} to JSON (default: lat_out.json)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print additional fit diagnostics (std errors, RMSE)",
    )
    args = parser.parse_args()

    path = Path(args.data_file)
    if not path.exists():
        log.error(f"Data file not found: {path}")
        raise SystemExit(1)

    log.info(f"Loading coverage/energy data from {path}")
    theta, E_avg = load_coverage_energy(path)
    log.info(f"Loaded {theta.size} points, theta in [{theta[0]:.4f}, {theta[-1]:.4f}]")

    fit = fit_grabow(theta, E_avg)

    log.success("Grabow two-parameter fit:")
    log.info(f"  E0      = {fit['E0']:+.4f} eV   (at theta = {fit['theta_ref']:.4f})")
    log.info(f"  theta0  = {fit['theta0']:.4f}")
    log.info(f"  epsilon = {fit['epsilon']:+.4f} eV")
    if args.verbose:
        log.info(f"  theta0_std  = {fit['theta0_std']:.4f}")
        log.info(f"  epsilon_std = {fit['epsilon_std']:.4f} eV")
        log.info(f"  RMSE        = {fit['rmse']:.4e} eV")

    if args.save_json is not None:
        out_path = Path(args.save_json)
        payload = {"theta0": fit["theta0"], "epsilon": fit["epsilon"]}
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        log.success(f"Saved parameters to {out_path}")


if __name__ == "__main__":
    main()
