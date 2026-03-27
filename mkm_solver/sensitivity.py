"""Sensitivity analysis: DRC, DTRC, and coverage sensitivity via finite difference."""

import copy
from typing import Dict

import numpy as np

from mkm_solver.schema import MKMModel
from mkm_solver.logger import log
from mkm_solver.utils.constants import KB


def compute_sensitivity(model: MKMModel, ss_result: Dict) -> Dict:
    """Run all enabled sensitivity analyses."""
    results = {}
    sens = model.sensitivity

    if not sens.enabled:
        return results

    target_rxn_id = sens.target_rate_reaction_id
    delta = sens.delta
    r0 = ss_result["tof"][target_rxn_id]

    if sens.compute_drc:
        log.info(f"Computing degree of rate control (delta={delta} eV)")
        results["drc"] = _degree_of_rate_control(model, r0, target_rxn_id, delta)

    if sens.compute_dtrc:
        log.info(f"Computing degree of thermodynamic rate control (delta={delta} eV)")
        results["dtrc"] = _degree_of_thermodynamic_rate_control(
            model, r0, target_rxn_id, delta
        )

    log.info("Computing coverage sensitivity")
    results["coverage_sensitivity"] = _coverage_sensitivity(
        model, ss_result["coverages"], delta
    )

    return results


def _degree_of_rate_control(
    model: MKMModel, r0: float, target_rxn_id: str, delta: float
) -> Dict[str, float]:
    """Campbell's DRC: X_RC,i = d(ln r) / d(ln k_i).

    Perturbs Ea by -delta (lowers both barriers equally, preserves K_eq).
    """
    from mkm_solver.solver import solve_steady_state

    T = model.temperature
    drc = {}

    for i, rxn in enumerate(model.reactions):
        model_pert = _perturb_reaction(model, i, -delta, 0.0)
        result_pert = solve_steady_state(model_pert)

        if result_pert["converged"] and abs(r0) > 1e-30:
            r_pert = result_pert["tof"][target_rxn_id]
            drc[rxn.id] = (
                (np.log(abs(r_pert)) - np.log(abs(r0))) / (delta / (KB * T))
            )
        else:
            drc[rxn.id] = float("nan") if not result_pert["converged"] else 0.0

    return drc


def _degree_of_thermodynamic_rate_control(
    model: MKMModel, r0: float, target_rxn_id: str, delta: float
) -> Dict[str, float]:
    """DTRC: X_TRC,i = d(ln r) / d(-dG_i / (kB*T)).

    Perturbs DeltaE by -delta (more exothermic), keeps Ea fixed.
    """
    from mkm_solver.solver import solve_steady_state

    T = model.temperature
    dtrc = {}

    for i, rxn in enumerate(model.reactions):
        model_pert = _perturb_reaction(model, i, 0.0, -delta)
        result_pert = solve_steady_state(model_pert)

        if result_pert["converged"] and abs(r0) > 1e-30:
            r_pert = result_pert["tof"][target_rxn_id]
            dtrc[rxn.id] = (
                (np.log(abs(r_pert)) - np.log(abs(r0))) / (delta / (KB * T))
            )
        else:
            dtrc[rxn.id] = float("nan") if not result_pert["converged"] else 0.0

    return dtrc


def _coverage_sensitivity(
    model: MKMModel, theta0: np.ndarray, delta: float
) -> Dict[str, Dict[str, float]]:
    """Sensitivity of steady-state coverages to each forward barrier:
    d(theta_i) / d(Ea_j)   [1/eV].
    """
    from mkm_solver.solver import solve_steady_state

    sens = {}
    species_names = list(model.surface_species.keys())

    for j, rxn in enumerate(model.reactions):
        model_pert = _perturb_reaction(model, j, delta, 0.0)
        result_pert = solve_steady_state(model_pert)

        if result_pert["converged"]:
            dtheta = result_pert["coverages"] - theta0
            sens[rxn.id] = {
                species_names[i]: dtheta[i] / delta
                for i in range(len(species_names))
            }
        else:
            sens[rxn.id] = {sp: float("nan") for sp in species_names}

    return sens


def _perturb_reaction(
    model: MKMModel, rxn_index: int, delta_Ea: float, delta_DeltaE: float
) -> MKMModel:
    """Return a deep copy of model with Ea and/or DeltaE perturbed."""
    model_pert = copy.deepcopy(model)
    model_pert.reactions[rxn_index].Ea += delta_Ea
    model_pert.reactions[rxn_index].DeltaE += delta_DeltaE
    return model_pert
