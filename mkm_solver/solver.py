"""ODE integration and steady-state solving.

Steady-state methods:
    integration: Integrate dθ/dt forward in time with solve_ivp until steady state.
        method controls the ODE solver: scipy-bdf, scipy-lsoda, scipy-radau.
    rootfinding: Solve dθ/dt=0 directly with fsolve. method is ignored.

With fsolve_fallback=true, if integration doesn't converge, rootfinding is
tried automatically using the integration result as initial guess.
"""

import threading
import time
from typing import Dict

import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from mkm_solver.schema import MKMModel, VALID_METHODS
from mkm_solver.build_mkm import (
    build_rhs, build_initial_conditions, compute_reaction_rates,
)
from mkm_solver.logger import log


SCIPY_METHOD_MAP = {
    "scipy-bdf": "BDF",
    "scipy-lsoda": "LSODA",
    "scipy-radau": "Radau",
}


def solve_steady_state(model: MKMModel) -> Dict:
    """Find steady-state coverages, then compute rates and TOF.

    Returns a dict with keys: coverages, rates, tof, converged,
    method_used, and sol (raw solver output).
    """
    settings = model.solver

    if settings.method not in VALID_METHODS:
        raise ValueError(
            f"Unknown solver method '{settings.method}'. "
            f"Valid methods: {', '.join(VALID_METHODS)}"
        )

    rhs_func = build_rhs(model)
    theta0 = build_initial_conditions(model)

    if settings.steady_state_method == "integration":
        result = _solve_by_integration(rhs_func, theta0, settings, model)
    elif settings.steady_state_method == "rootfinding":
        result = _solve_by_rootfinding(rhs_func, theta0, model)
    else:
        raise ValueError(
            f"Unknown steady_state_method: {settings.steady_state_method}"
        )

    if not result["converged"] and settings.fsolve_fallback:
        integration_residual = result.get("max_deriv", float("inf"))
        if integration_residual > 10 * settings.steady_state_tol:
            log.warning("Integration did not converge, falling back to fsolve")
            # Rebuild full state vector from species coverages for fsolve
            y_guess = build_initial_conditions(model)
            n_sp = len(model.surface_species)
            y_guess[:n_sp] = result["coverages"]
            # Recompute free sites from site balance
            for k, st_name in enumerate(model.site_types):
                occupied = sum(
                    sp.n_sites * y_guess[model.species_index[sp.name]]
                    for sp in model.surface_species.values()
                    if sp.site_type == st_name
                )
                y_guess[n_sp + k] = max(0.0, 1.0 - occupied)

            result_fsolve = _solve_by_rootfinding(rhs_func, y_guess, model)
            if result_fsolve["converged"]:
                result = result_fsolve
            else:
                fsolve_residual = np.max(np.abs(
                    rhs_func(0, y_guess)
                ))
                if fsolve_residual < integration_residual:
                    result = result_fsolve
                else:
                    log.warning("fsolve did not improve, keeping integration result")
        else:
            log.info("Nearly converged, accepting integration result")

    # Always compute rates and TOF
    result["rates"] = compute_reaction_rates(model, result["coverages"])
    result["tof"] = {
        rxn_id: r["net"] for rxn_id, r in result["rates"].items()
    }

    return result


# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------

def _run_in_thread_with_progress(target, progress_desc, progress_state=None,
                                 t_start=None, t_end=None):
    """Run target() in a thread with a Rich progress indicator."""
    result_holder = {"result": None, "error": None}

    def wrapper():
        try:
            result_holder["result"] = target()
        except Exception as e:
            result_holder["error"] = e

    thread = threading.Thread(target=wrapper)

    has_time_range = (t_start is not None and t_end is not None
                      and t_end > t_start and progress_state is not None)

    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ]
    if has_time_range:
        columns += [
            TextColumn("t={task.fields[t_val]:.2e}/{task.fields[t_end]:.2e}"),
            TimeElapsedColumn(),
            TextColumn("ETA {task.fields[eta]}"),
        ]
    else:
        columns.append(TimeElapsedColumn())

    with Progress(*columns, transient=True) as progress:
        task_kwargs = {"total": None}
        if has_time_range:
            task_kwargs["t_val"] = t_start
            task_kwargs["t_end"] = t_end
            task_kwargs["eta"] = "..."
        task = progress.add_task(progress_desc, **task_kwargs)

        wall_start = time.time()
        thread.start()
        while thread.is_alive():
            if has_time_range:
                t_cur = progress_state.get("t_current", t_start)
                elapsed_wall = time.time() - wall_start
                frac = (t_cur - t_start) / (t_end - t_start) if t_cur > t_start else 0.0
                frac = max(0.0, min(1.0, frac))
                if frac > 1e-10 and elapsed_wall > 0.5:
                    eta_sec = elapsed_wall / frac * (1.0 - frac)
                    if eta_sec < 60:
                        eta_str = f"{eta_sec:.0f}s"
                    elif eta_sec < 3600:
                        eta_str = f"{eta_sec/60:.1f}m"
                    else:
                        eta_str = f"{eta_sec/3600:.1f}h"
                else:
                    eta_str = "..."
                progress.update(task, t_val=t_cur, eta=eta_str)
            time.sleep(0.1)
        thread.join()

        if has_time_range:
            progress.update(task, t_val=t_end, eta="done")

    if result_holder["error"] is not None:
        raise result_holder["error"]

    return result_holder["result"]


# ---------------------------------------------------------------------------
# Integration (time-march to steady state) — dispatches by method
# ---------------------------------------------------------------------------

def _solve_by_integration(rhs_func, theta0, settings, model) -> Dict:
    """Dispatch to the appropriate integration backend."""
    method = settings.method
    if method in SCIPY_METHOD_MAP:
        return _integrate_scipy(rhs_func, theta0, settings, model)
    elif method == "sundials-cvode-bdf":
        return _integrate_sundials_cvode(rhs_func, theta0, settings, model)
    elif method == "assimulo-cvode-bdf":
        return _integrate_assimulo_cvode(rhs_func, theta0, settings, model)
    elif method == "assimulo-radau5":
        return _integrate_assimulo_radau5(rhs_func, theta0, settings, model)
    elif method == "assimulo-ida-dae":
        return _integrate_assimulo_ida_dae(rhs_func, theta0, settings, model)
    else:
        raise ValueError(f"No integration backend for method '{method}'")


def _postprocess_integration(rhs_func, y_final, settings, model) -> Dict:
    """Common post-processing after integration: extract coverages, check convergence."""
    n_sp = len(model.surface_species)
    theta_final = y_final[:n_sp]

    deriv_final = rhs_func(0, y_final)
    max_deriv = np.max(np.abs(deriv_final[:n_sp]))
    converged = max_deriv < settings.steady_state_tol
    log.info(f"max |dθ/dt| = {max_deriv:.4e} (tol={settings.steady_state_tol:.1e})")

    for k, st_name in enumerate(model.site_types):
        site_balance = y_final[n_sp + k]
        for sp_name, sp in model.surface_species.items():
            if sp.site_type == st_name:
                site_balance += sp.n_sites * y_final[model.species_index[sp_name]]
        log.info(f"Site balance ({st_name}): {site_balance:.10f}")

    return {
        "coverages": theta_final,
        "converged": converged,
        "max_deriv": max_deriv,
        "method_used": settings.method,
    }


def _integrate_scipy(rhs_func, theta0, settings, model) -> Dict:
    """Integrate with scipy solve_ivp (BDF/LSODA/Radau)."""
    t_start, t_end = settings.t_span
    progress_state = {"t_current": t_start}

    def rhs_with_progress(t, y):
        progress_state["t_current"] = t
        return rhs_func(t, y)

    scipy_method = SCIPY_METHOD_MAP[settings.method]
    sol = _run_in_thread_with_progress(
        target=lambda: solve_ivp(
            rhs_with_progress,
            settings.t_span,
            theta0,
            method=scipy_method,
            rtol=settings.rtol,
            atol=settings.atol,
            dense_output=True,
        ),
        progress_desc=f"Integrating ({settings.method})",
        progress_state=progress_state,
        t_start=t_start,
        t_end=t_end,
    )

    result = _postprocess_integration(rhs_func, sol.y[:, -1], settings, model)
    result["sol"] = sol
    return result


def _integrate_sundials_cvode(rhs_func, theta0, settings, model) -> Dict:
    """Integrate with SUNDIALS CVODE via scikits.odes.

    Matches debug/solve_ivp/sundials/mkm_cvode_bdf.py.
    CVODE RHS signature: rhs(t, y, ydot) -> 0
    """
    try:
        from scikits_odes_sundials.cvode import CVODE
    except ImportError:
        raise ImportError(
            "sundials-cvode-bdf requires scikits-odes-sundials. Install with:\n"
            "  conda install -c conda-forge sundials\n"
            "  pip install scikits-odes-sundials"
        )

    t_start, t_end = settings.t_span
    progress_state = {"t_current": t_start}

    # CVODE expects rhs(t, y, ydot) -> 0, filling ydot in-place
    def cvode_rhs(t, y, ydot):
        progress_state["t_current"] = t
        ydot[:] = rhs_func(t, y)
        return 0

    t_eval = np.concatenate([
        np.array([0.0]),
        np.logspace(-10, np.log10(t_end), 500),
    ])

    def run_cvode():
        solver = CVODE(cvode_rhs)
        solver.set_options(
            old_api=False,
            lmm_type='BDF',
            atol=settings.atol,
            rtol=settings.rtol,
            max_steps=500000,
        )
        return solver.solve(t_eval, theta0)

    sol = _run_in_thread_with_progress(
        target=run_cvode,
        progress_desc=f"Integrating ({settings.method})",
        progress_state=progress_state,
        t_start=t_start,
        t_end=t_end,
    )

    if sol.flag != 0:
        log.warning(f"CVODE flag={sol.flag}: {sol.message}")

    # sol.values.y shape is (npoints, n_vars) — take last row
    y_final = sol.values.y[-1, :]

    result = _postprocess_integration(rhs_func, y_final, settings, model)
    result["sol"] = sol
    return result


# ---------------------------------------------------------------------------
# Assimulo CVode BDF
# ---------------------------------------------------------------------------

def _integrate_assimulo_cvode(rhs_func, theta0, settings, model) -> Dict:
    """Integrate with Assimulo CVode (SUNDIALS BDF via Assimulo wrapper).

    Matches debug/solve_ivp/assimulo/mkm_cvode.py.
    Assimulo Explicit_Problem RHS: rhs(t, y) -> dydt (same as scipy).
    """
    try:
        from assimulo.solvers import CVode
        from assimulo.problem import Explicit_Problem
    except ImportError:
        raise ImportError(
            "assimulo-cvode-bdf requires Assimulo. Install with:\n"
            "  conda install -c conda-forge assimulo"
        )

    t_start, t_end = settings.t_span
    progress_state = {"t_current": t_start}

    def rhs_with_progress(t, y):
        progress_state["t_current"] = t
        return rhs_func(t, y)

    n_vars = len(theta0)

    def run_solver():
        problem = Explicit_Problem(rhs_with_progress, theta0, t_start)
        problem.name = model.model_name

        solver = CVode(problem)
        solver.discr = 'BDF'
        solver.iter = 'Newton'
        solver.linear_solver = 'DENSE'
        solver.atol = np.ones(n_vars) * settings.atol
        solver.rtol = settings.rtol
        solver.maxsteps = 500000
        solver.verbosity = 50

        t, y = solver.simulate(t_end)
        return np.array(t), np.array(y)

    t, y = _run_in_thread_with_progress(
        target=run_solver,
        progress_desc=f"Integrating ({settings.method})",
        progress_state=progress_state,
        t_start=t_start,
        t_end=t_end,
    )

    # y shape: (npoints, n_vars) — take last row
    y_final = y[-1, :]
    result = _postprocess_integration(rhs_func, y_final, settings, model)
    result["sol"] = (t, y)
    return result


# ---------------------------------------------------------------------------
# Assimulo Radau5ODE
# ---------------------------------------------------------------------------

def _integrate_assimulo_radau5(rhs_func, theta0, settings, model) -> Dict:
    """Integrate with Assimulo Radau5ODE (Hairer implicit Runge-Kutta).

    Matches debug/solve_ivp/assimulo/mkm_radau5.py.
    """
    try:
        from assimulo.solvers import Radau5ODE
        from assimulo.problem import Explicit_Problem
    except ImportError:
        raise ImportError(
            "assimulo-radau5 requires Assimulo. Install with:\n"
            "  conda install -c conda-forge assimulo"
        )

    t_start, t_end = settings.t_span
    progress_state = {"t_current": t_start}

    def rhs_with_progress(t, y):
        progress_state["t_current"] = t
        return rhs_func(t, y)

    def run_solver():
        problem = Explicit_Problem(rhs_with_progress, theta0, t_start)
        problem.name = model.model_name

        solver = Radau5ODE(problem)
        solver.atol = settings.atol
        solver.rtol = settings.rtol
        solver.inith = 1e-10
        solver.maxsteps = 500000

        t, y = solver.simulate(t_end)
        return np.array(t), np.array(y)

    t, y = _run_in_thread_with_progress(
        target=run_solver,
        progress_desc=f"Integrating ({settings.method})",
        progress_state=progress_state,
        t_start=t_start,
        t_end=t_end,
    )

    y_final = y[-1, :]
    result = _postprocess_integration(rhs_func, y_final, settings, model)
    result["sol"] = (t, y)
    return result


# ---------------------------------------------------------------------------
# Assimulo IDA DAE
# ---------------------------------------------------------------------------

def _integrate_assimulo_ida_dae(rhs_func, theta0, settings, model) -> Dict:
    """Integrate with Assimulo IDA (SUNDIALS DAE solver).

    Matches debug/solve_ivp/assimulo/mkm_ida_dae.py.
    Uses true algebraic constraint for site balance instead of ODE.

    For differential variables: F[i] = yd[i] - rhs_i(t, y) = 0
    For algebraic variables (free sites): F[i] = 1.0 - Σ(n_j * y[j]) = 0
    """
    try:
        from assimulo.solvers import IDA
        from assimulo.problem import Implicit_Problem
    except ImportError:
        raise ImportError(
            "assimulo-ida-dae requires Assimulo. Install with:\n"
            "  conda install -c conda-forge assimulo"
        )

    t_start, t_end = settings.t_span
    progress_state = {"t_current": t_start}
    n_sp = len(model.surface_species)
    n_vars = len(theta0)
    site_type_list = list(model.site_types.keys())

    # Build site occupancy array and free-site indices
    n_occ = np.zeros(n_vars)
    for sp_name, sp in model.surface_species.items():
        n_occ[model.species_index[sp_name]] = sp.n_sites
    free_site_indices = []
    for k, st_name in enumerate(site_type_list):
        fi = n_sp + k
        n_occ[fi] = 1.0  # free site occupies 1 site in the balance
        free_site_indices.append(fi)

    # Build mapping: which species belong to which site type's balance
    site_balance_members = {}  # free_site_idx -> list of (species_idx, n_occ)
    for k, st_name in enumerate(site_type_list):
        fi = n_sp + k
        members = [(fi, 1.0)]  # free site itself
        for sp_name, sp in model.surface_species.items():
            if sp.site_type == st_name:
                members.append((model.species_index[sp_name], float(sp.n_sites)))
        site_balance_members[fi] = members

    def dae_residual(t, y, yd):
        progress_state["t_current"] = t
        rhs_vals = rhs_func(t, y)
        F = yd - rhs_vals
        # Override free-site equations with algebraic site balance constraint
        for fi in free_site_indices:
            constraint = 1.0
            for idx, n in site_balance_members[fi]:
                constraint -= n * y[idx]
            F[fi] = constraint
        return F

    # Consistent initial derivatives
    yd0 = rhs_func(t_start, theta0)
    for fi in free_site_indices:
        yd0[fi] = 0.0  # algebraic variable has no time derivative

    # Mark algebraic vs differential variables
    algvar = np.ones(n_vars)
    for fi in free_site_indices:
        algvar[fi] = 0.0

    def run_solver():
        problem = Implicit_Problem(dae_residual, theta0, yd0, t_start)
        problem.name = model.model_name
        problem.algvar = algvar

        solver = IDA(problem)
        solver.atol = np.ones(n_vars) * settings.atol
        solver.rtol = settings.rtol
        solver.maxsteps = 500000
        solver.suppress_alg = True
        solver.make_consistent('IDA_YA_YDP_INIT')

        t, y, yd = solver.simulate(t_end)
        return np.array(t), np.array(y), np.array(yd)

    t, y, yd = _run_in_thread_with_progress(
        target=run_solver,
        progress_desc=f"Integrating ({settings.method})",
        progress_state=progress_state,
        t_start=t_start,
        t_end=t_end,
    )

    y_final = y[-1, :]
    result = _postprocess_integration(rhs_func, y_final, settings, model)
    result["sol"] = (t, y, yd)
    return result


# ---------------------------------------------------------------------------
# Root-finding (solve dθ/dt = 0 directly)
# ---------------------------------------------------------------------------

def _solve_by_rootfinding(rhs_func, y0, model) -> Dict:
    """Solve f(y) = 0 using scipy fsolve with a spinner."""
    def f(y):
        return rhs_func(0, np.clip(y, 0.0, None))

    result = _run_in_thread_with_progress(
        target=lambda: fsolve(f, y0, full_output=True),
        progress_desc="Root-finding (fsolve)",
    )

    y_ss, info, ier, _ = result
    n_sp = len(model.surface_species)

    return {
        "coverages": y_ss[:n_sp],
        "converged": ier == 1,
        "sol": info,
        "method_used": "fsolve",
    }
