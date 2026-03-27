"""Write all MKM results to an output directory as separate files."""

import json
from pathlib import Path
from typing import Dict

import numpy as np

from mkm_solver.schema import MKMModel
from mkm_solver.build_mkm import (
    compute_free_site_coverages, _compute_free_site_powers,
    _build_stoich_matrix, _get_adsorption_params,
)
from mkm_solver.kinetics import rate_constants_surface, rate_constants_adsorption
from mkm_solver.utils.constants import KB


def _reaction_equation(rxn, model, free_site_powers_j):
    """Build a human-readable reaction equation string with auto-balanced free sites."""
    lhs_parts = []
    for sp, stoich in rxn.reactants.items():
        if sp in model.gas_species:
            name = sp.replace("_g", "(g)")
        else:
            name = sp
        lhs_parts.append(name if stoich == 1 else f"{stoich}{name}")
    for st, (n_fwd, _) in free_site_powers_j.items():
        if n_fwd > 0:
            site = f"*_{st}"
            lhs_parts.append(site if n_fwd == 1 else f"{n_fwd}{site}")

    rhs_parts = []
    for sp, stoich in rxn.products.items():
        if sp in model.gas_species:
            name = sp.replace("_g", "(g)")
        else:
            name = sp
        rhs_parts.append(name if stoich == 1 else f"{stoich}{name}")
    for st, (_, n_rev) in free_site_powers_j.items():
        if n_rev > 0:
            site = f"*_{st}"
            rhs_parts.append(site if n_rev == 1 else f"{n_rev}{site}")

    return " + ".join(lhs_parts) + " <-> " + " + ".join(rhs_parts)


def write_all(
    model: MKMModel,
    ss_result: Dict,
    sensitivity_result: Dict,
    output_dir: str,
) -> None:
    """Write all results to the output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_model_equations(model, out / "model_equations.txt")
    write_rate_constants(model, ss_result, out / "rate_constants.json")
    write_coverages(model, ss_result, out / "coverages.json")
    write_rates(model, ss_result, out / "rates.json")
    write_summary(model, ss_result, out / "summary.txt")

    if sensitivity_result:
        write_sensitivity(model, sensitivity_result, out / "sensitivity.json")


# ---------------------------------------------------------------------------
# model_equations.txt
# ---------------------------------------------------------------------------

def write_model_equations(model: MKMModel, path: Path) -> None:
    """Write rate expressions, site balances, and ODEs to a text file."""
    free_site_powers = _compute_free_site_powers(model)
    nu = _build_stoich_matrix(model)
    species_names = list(model.surface_species.keys())
    lines = []

    lines.append(f"Model: {model.model_name}")
    lines.append(f"Temperature: {model.temperature} K")
    lines.append(f"Site types: {list(model.site_types.keys())}")
    lines.append(f"Surface species: {len(model.surface_species)}")
    lines.append(f"Reactions: {len(model.reactions)}")
    lines.append("")

    # Reaction equations (with auto-balanced free sites)
    lines.append("--- Reactions ---")
    for j, rxn in enumerate(model.reactions):
        eq = _reaction_equation(rxn, model, free_site_powers[j])
        lines.append(f"  {rxn.id}: {eq}")

    lines.append("")

    # Rate expressions
    lines.append("--- Rate Expressions ---")
    for j, rxn in enumerate(model.reactions):
        powers = free_site_powers[j]
        fwd_terms = []
        for sp, stoich in rxn.reactants.items():
            if sp in model.gas_species:
                fwd_terms.append(f"P_{sp}" if stoich == 1 else f"P_{sp}^{stoich}")
            else:
                fwd_terms.append(f"θ_{sp}" if stoich == 1 else f"θ_{sp}^{stoich}")
        for st, (n_fwd, _) in powers.items():
            if n_fwd > 0:
                fwd_terms.append(f"θ_*{st}" if n_fwd == 1 else f"θ_*{st}^{n_fwd}")

        rev_terms = []
        for sp, stoich in rxn.products.items():
            if sp in model.gas_species:
                rev_terms.append(f"P_{sp}" if stoich == 1 else f"P_{sp}^{stoich}")
            else:
                rev_terms.append(f"θ_{sp}" if stoich == 1 else f"θ_{sp}^{stoich}")
        for st, (_, n_rev) in powers.items():
            if n_rev > 0:
                rev_terms.append(f"θ_*{st}" if n_rev == 1 else f"θ_*{st}^{n_rev}")

        fwd_str = " * ".join([f"k{j+1}f"] + fwd_terms)
        rev_str = " * ".join([f"k{j+1}r"] + rev_terms)
        lines.append(f"  {rxn.id} ({rxn.reaction_type}): r{j+1} = {fwd_str} - {rev_str}")

    lines.append("")

    # Site balances
    lines.append("--- Site Balances ---")
    for st_name in model.site_types:
        terms = []
        for sp_name, sp in model.surface_species.items():
            if sp.site_type == st_name:
                if sp.n_sites == 1:
                    terms.append(f"θ_{sp_name}")
                else:
                    terms.append(f"{sp.n_sites}*θ_{sp_name}")
        balance = " + ".join(terms + [f"θ_*{st_name}"])
        lines.append(f"  Site {st_name}: {balance} = 1.0")

    lines.append("")

    # ODEs
    lines.append("--- ODEs ---")
    for i, sp_name in enumerate(species_names):
        sp = model.surface_species[sp_name]
        terms = []
        for j in range(len(model.reactions)):
            coeff = nu[i, j]
            if coeff == 0:
                continue
            rxn_id = model.reactions[j].id
            if coeff == 1:
                terms.append(f"+{rxn_id}")
            elif coeff == -1:
                terms.append(f"-{rxn_id}")
            elif coeff > 0:
                terms.append(f"+{coeff:.0f}*{rxn_id}")
            else:
                terms.append(f"{coeff:.0f}*{rxn_id}")
        rhs_str = " ".join(terms) if terms else "0"
        lines.append(f"  dθ_{sp_name}/dt (site {sp.site_type}) = {rhs_str}")

    if model.lateral_interactions:
        lines.append("")
        lines.append("--- Lateral Interactions ---")
        for sp_name, li in model.lateral_interactions.items():
            lines.append(
                f"  {sp_name}: E_eff = E_0 + {li.epsilon:.4f} * max(0, θ - {li.theta0:.4f})"
            )

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# rate_constants.json
# ---------------------------------------------------------------------------

def write_rate_constants(model: MKMModel, ss_result: Dict, path: Path) -> None:
    """Write base and final (lateral-corrected) rate constants."""
    T = model.temperature
    theta = ss_result.get("coverages")
    free_site_powers = _compute_free_site_powers(model)
    data = {}

    for j, rxn in enumerate(model.reactions):
        # Base rate constants (no lateral interactions)
        if rxn.reaction_type == "surface":
            kf_base, kr_base = rate_constants_surface(rxn.Ea, rxn.DeltaE, T)
            method = "hTST"
        elif rxn.reaction_type == "adsorption":
            mass, n_sites_density = _get_adsorption_params(rxn, model)
            kf_base, kr_base = rate_constants_adsorption(rxn.DeltaE, T, mass, n_sites_density)
            method = "collision_theory"

        entry = {
            "reaction": _reaction_equation(rxn, model, free_site_powers[j]),
            "type": rxn.reaction_type,
            "method": method,
            "Ea": rxn.Ea,
            "DeltaE": rxn.DeltaE,
            "kf_base": kf_base,
            "kr_base": kr_base,
            "K_eq_base": np.exp(-rxn.DeltaE / (KB * T)),
        }

        # Final rate constants (with lateral interaction correction on K_eq)
        kf_final = kf_base
        kr_final = kr_base
        DeltaE_eff = rxn.DeltaE
        lateral_applied = False

        if theta is not None:
            for sp in rxn.products:
                if sp in model.lateral_interactions:
                    li = model.lateral_interactions[sp]
                    idx = model.species_index[sp]
                    delta = li.epsilon * max(0.0, theta[idx] - li.theta0)
                    DeltaE_eff += delta
                    lateral_applied = True

        if lateral_applied:
            K_eq_eff = np.exp(-DeltaE_eff / (KB * T))
            kr_final = kf_final / K_eq_eff
            entry["DeltaE_eff"] = DeltaE_eff
            entry["K_eq_eff"] = K_eq_eff

        entry["kf_final"] = kf_final
        entry["kr_final"] = kr_final
        entry["lateral_applied"] = lateral_applied

        data[rxn.id] = entry

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# coverages.json
# ---------------------------------------------------------------------------

def write_coverages(model: MKMModel, ss_result: Dict, path: Path) -> None:
    """Write steady-state coverages and free site coverages."""
    theta = ss_result["coverages"]
    free = compute_free_site_coverages(theta, model)

    data = {
        "surface_species": {},
        "free_sites": {},
    }
    for name, sp in model.surface_species.items():
        idx = model.species_index[name]
        data["surface_species"][name] = {
            "coverage": float(theta[idx]),
            "site_type": sp.site_type,
            "n_sites": sp.n_sites,
        }
    for site_name, cov in free.items():
        data["free_sites"][site_name] = float(cov)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# rates.json
# ---------------------------------------------------------------------------

def write_rates(model: MKMModel, ss_result: Dict, path: Path) -> None:
    """Write forward, reverse, and net rates + TOF for each reaction."""
    free_site_powers = _compute_free_site_powers(model)
    data = {}
    if "rates" in ss_result:
        for j, (rxn_id, rates) in enumerate(ss_result["rates"].items()):
            rxn = model.reactions[j]
            data[rxn_id] = {
                "reaction": _reaction_equation(rxn, model, free_site_powers[j]),
                "forward": rates["forward"],
                "reverse": rates["reverse"],
                "net": rates["net"],
            }
    if "tof" in ss_result:
        for rxn_id in data:
            data[rxn_id]["tof"] = ss_result["tof"].get(rxn_id, None)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# summary.txt
# ---------------------------------------------------------------------------

def write_summary(model: MKMModel, ss_result: Dict, path: Path) -> None:
    """Write a concise summary of the run."""
    lines = []
    lines.append(f"Model: {model.model_name}")
    lines.append(f"Temperature: {model.temperature} K")
    lines.append(f"Solver: {ss_result.get('method_used', 'N/A')}")
    lines.append(f"Converged: {ss_result['converged']}")
    lines.append("")

    theta = ss_result["coverages"]
    lines.append("Steady-State Coverages:")
    for name in model.surface_species:
        idx = model.species_index[name]
        lines.append(f"  {name:20s}  {theta[idx]:.6e}")
    free = compute_free_site_coverages(theta, model)
    for site_name, cov in free.items():
        lines.append(f"  {site_name:20s}  {cov:.6e}  (free)")
    lines.append("")

    if "rates" in ss_result:
        free_site_powers = _compute_free_site_powers(model)
        lines.append("Reaction Rates (s^-1):")
        for j, (rxn_id, rates) in enumerate(ss_result["rates"].items()):
            rxn = model.reactions[j]
            eq = _reaction_equation(rxn, model, free_site_powers[j])
            lines.append(f"  {rxn_id}: {eq}")
            lines.append(
                f"    forward={rates['forward']:.4e}"
                f"  reverse={rates['reverse']:.4e}"
                f"  net={rates['net']:.4e}"
            )

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# sensitivity.json
# ---------------------------------------------------------------------------

def write_sensitivity(model: MKMModel, sensitivity_result: Dict, path: Path) -> None:
    """Write sensitivity analysis results."""
    data = {}
    if "drc" in sensitivity_result:
        data["drc"] = {
            "target_reaction": model.sensitivity.target_rate_reaction_id,
            "values": {k: (v if not np.isnan(v) else None)
                       for k, v in sensitivity_result["drc"].items()},
        }
    if "dtrc" in sensitivity_result:
        data["dtrc"] = {
            "values": {k: (v if not np.isnan(v) else None)
                       for k, v in sensitivity_result["dtrc"].items()},
        }
    if "coverage_sensitivity" in sensitivity_result:
        data["coverage_sensitivity"] = sensitivity_result["coverage_sensitivity"]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
