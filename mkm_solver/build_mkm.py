"""Assemble the microkinetic ODE system.

ODE formulation: θ_* (free sites) are state variables with
    dθ_*/dt = -Σ(n_i * dθ_i/dt)  for species i on that site type.
This matches the MATLAB ode15s implementation and the tested debug files.

State vector: y = [θ_sp1, ..., θ_spN, θ_*A, θ_*B, ...]
Initial conditions: species = 0, free sites = 1.0

Site balance per site type:
    Σ(n_sites_i * θ_i) + θ_free = 1.0
    Maintained by the integrator through the site-balance ODE.

Lateral interactions modify K_eq (and thus k_reverse) only:
    E_eff = E_0 + ε * max(0, θ - θ_0)
    K_eq_eff = exp(-E_eff / (kB*T))
    k_reverse = k_forward / K_eq_eff
"""

import numpy as np
from typing import Callable, Dict, List, Tuple

from mkm_solver.schema import MKMModel, Reaction
from mkm_solver.kinetics import rate_constants_surface, rate_constants_adsorption
from mkm_solver.utils.constants import KB


# ---------------------------------------------------------------------------
# Free site power computation (per site type, per reaction)
# ---------------------------------------------------------------------------

def _compute_free_site_powers(
    model: MKMModel,
) -> List[Dict[str, Tuple[int, int]]]:
    """For each reaction, compute the free site power on each side per site type.

    Returns a list (one per reaction) of dicts:
        {site_type: (n_free_fwd, n_free_rev)}
    where n_free_fwd = power of θ_*X in forward rate,
          n_free_rev = power of θ_*X in reverse rate.
    """
    result = []
    for rxn in model.reactions:
        site_powers = {}
        for st_name in model.site_types:
            sites_lhs = 0
            for sp, stoich in rxn.reactants.items():
                if sp in model.surface_species:
                    ss = model.surface_species[sp]
                    if ss.site_type == st_name:
                        sites_lhs += ss.n_sites * stoich

            sites_rhs = 0
            for sp, stoich in rxn.products.items():
                if sp in model.surface_species:
                    ss = model.surface_species[sp]
                    if ss.site_type == st_name:
                        sites_rhs += ss.n_sites * stoich

            n_free_fwd = max(0, sites_rhs - sites_lhs)
            n_free_rev = max(0, sites_lhs - sites_rhs)

            if n_free_fwd > 0 or n_free_rev > 0:
                site_powers[st_name] = (n_free_fwd, n_free_rev)

        result.append(site_powers)
    return result


def compute_free_site_coverages(
    theta: np.ndarray, model: MKMModel
) -> Dict[str, float]:
    """Compute free-site coverages from species coverages via site balance.

    Works with both species-only arrays (length n_species) and full state
    vectors (length n_species + n_site_types).
    """
    n_sp = len(model.surface_species)
    site_type_list = list(model.site_types.keys())

    # If full state vector, read free sites directly
    if len(theta) > n_sp:
        free = {}
        for k, st_name in enumerate(site_type_list):
            free[f"*_{st_name}"] = theta[n_sp + k]
        return free

    # Otherwise compute from site balance
    free = {}
    for st_name in site_type_list:
        occupied = sum(
            sp.n_sites * theta[model.species_index[sp.name]]
            for sp in model.surface_species.values()
            if sp.site_type == st_name
        )
        free[f"*_{st_name}"] = 1.0 - occupied
    return free


# ---------------------------------------------------------------------------
# Stoichiometric matrix
# ---------------------------------------------------------------------------

def _build_stoich_matrix(model: MKMModel) -> np.ndarray:
    """Build stoichiometric matrix nu of shape (n_species, n_reactions).

    Only for surface species (not free sites).
    nu[i, j] = net stoichiometric coefficient of surface species i
    in reaction j. Gas species excluded.
    """
    n_sp = len(model.surface_species)
    n_rxn = len(model.reactions)
    nu = np.zeros((n_sp, n_rxn))
    for j, rxn in enumerate(model.reactions):
        for sp, coeff in rxn.reactants.items():
            if sp in model.species_index:
                nu[model.species_index[sp], j] -= coeff
        for sp, coeff in rxn.products.items():
            if sp in model.species_index:
                nu[model.species_index[sp], j] += coeff
    return nu


# ---------------------------------------------------------------------------
# Adsorption params helper
# ---------------------------------------------------------------------------

def _get_adsorption_params(
    rxn: Reaction, model: MKMModel
) -> Tuple[float, float]:
    """For adsorption reactions, find gas species mass (amu) and site density (m^-2)."""
    gas_name = None
    for sp in rxn.reactants:
        if sp in model.gas_species:
            gas_name = sp
            break

    site_type_name = None
    for sp in rxn.products:
        if sp in model.surface_species:
            site_type_name = model.surface_species[sp].site_type
            break

    mass = model.gas_species[gas_name].mass
    n_sites = model.site_types[site_type_name].n_sites
    return mass, n_sites


# ---------------------------------------------------------------------------
# Build initial conditions
# ---------------------------------------------------------------------------

def build_initial_conditions(model: MKMModel) -> np.ndarray:
    """Build initial state vector: species = 0, free sites = 1.0.

    State vector layout: [θ_sp1, ..., θ_spN, θ_*A, θ_*B, ...]
    """
    n_sp = len(model.surface_species)
    n_sites = len(model.site_types)
    y0 = np.zeros(n_sp + n_sites)
    # Free sites start at 1.0 (clean surface)
    for k in range(n_sites):
        y0[n_sp + k] = 1.0
    return y0


# ---------------------------------------------------------------------------
# Build RHS closure (ODE formulation)
# ---------------------------------------------------------------------------

def build_rhs(model: MKMModel) -> Callable:
    """Build and return the ODE right-hand side function.

    State vector: y = [θ_sp1, ..., θ_spN, θ_*A, θ_*B, ...]

    Species ODEs: dθ_i/dt = Σ(nu_ij * r_j)
    Free site ODEs: dθ_*/dt = -Σ(n_i * dθ_i/dt)  for species on that site

    This matches the MATLAB/debug file implementation where the site balance
    is maintained by the integrator through the free-site ODE.
    """
    T = model.temperature
    nu = _build_stoich_matrix(model)
    free_site_powers = _compute_free_site_powers(model)
    species_names = list(model.surface_species.keys())
    n_sp = len(species_names)
    n_rxn = len(model.reactions)
    site_type_list = list(model.site_types.keys())
    n_site_types = len(site_type_list)
    n_total = n_sp + n_site_types

    # Map site type name -> index in the free-site portion of state vector
    free_site_idx = {}  # site_type -> index in y
    for k, st_name in enumerate(site_type_list):
        free_site_idx[st_name] = n_sp + k

    # Precompute base rate constants
    base_kf = np.zeros(n_rxn)
    base_kr = np.zeros(n_rxn)
    for j, rxn in enumerate(model.reactions):
        if rxn.reaction_type == "surface":
            kf, kr = rate_constants_surface(rxn.Ea, rxn.DeltaE, T)
        elif rxn.reaction_type == "adsorption":
            mass, n_sites_density = _get_adsorption_params(rxn, model)
            kf, kr = rate_constants_adsorption(rxn.DeltaE, T, mass, n_sites_density)
        base_kf[j] = kf
        base_kr[j] = kr

    # Identify reactions with lateral-affected products
    lateral_rxn_indices = []
    for j, rxn in enumerate(model.reactions):
        for sp in rxn.products:
            if sp in model.lateral_interactions:
                lateral_rxn_indices.append(j)
                break

    # Site-type to species mapping for the site-balance ODE
    # site_species_for_balance[st_name] = list of (species_index, n_sites_occupied)
    site_species_for_balance = {}
    for sp_name, sp in model.surface_species.items():
        idx = model.species_index[sp_name]
        site_species_for_balance.setdefault(sp.site_type, []).append((idx, sp.n_sites))

    # Build reaction info for rate computation
    rxn_info = []
    for j, rxn in enumerate(model.reactions):
        fwd_surface = []
        fwd_gas = []
        for sp, stoich in rxn.reactants.items():
            if sp in model.gas_species:
                fwd_gas.append((model.gas_species[sp].pressure, stoich))
            elif sp in model.species_index:
                fwd_surface.append((model.species_index[sp], stoich))

        rev_surface = []
        for sp, stoich in rxn.products.items():
            if sp in model.species_index:
                rev_surface.append((model.species_index[sp], stoich))

        rxn_info.append((fwd_surface, fwd_gas, rev_surface))

    def rhs(t, y):
        F = np.zeros(n_total)

        # Read θ values directly from state vector (no clipping — let
        # the integrator handle it, matching MATLAB ode15s behavior)
        theta = y

        # Free site coverages are in the state vector
        theta_free = {}
        for st_name in site_type_list:
            theta_free[st_name] = theta[free_site_idx[st_name]]

        # Copy base rate constants
        kf = base_kf.copy()
        kr = base_kr.copy()

        # Recompute k_r for lateral-affected reactions
        kBT = KB * T
        for j in lateral_rxn_indices:
            rxn = model.reactions[j]
            DeltaE_eff = rxn.DeltaE
            for sp in rxn.products:
                if sp in model.lateral_interactions:
                    li = model.lateral_interactions[sp]
                    idx = model.species_index[sp]
                    delta = li.epsilon * max(0.0, theta[idx] - li.theta0)
                    DeltaE_eff += delta
            K_eq_eff = np.exp(-DeltaE_eff / kBT)
            kr[j] = kf[j] / K_eq_eff

        # Compute net rates
        r_net = np.zeros(n_rxn)
        for j in range(n_rxn):
            fwd_surface, fwd_gas, rev_surface = rxn_info[j]
            powers = free_site_powers[j]

            rf = kf[j]
            for pressure, stoich in fwd_gas:
                rf *= pressure ** stoich
            for idx, stoich in fwd_surface:
                rf *= theta[idx] ** stoich
            for st, (n_fwd, _) in powers.items():
                if n_fwd > 0:
                    rf *= theta_free[st] ** n_fwd

            rr = kr[j]
            for idx, stoich in rev_surface:
                rr *= theta[idx] ** stoich
            for st, (_, n_rev) in powers.items():
                if n_rev > 0:
                    rr *= theta_free[st] ** n_rev

            r_net[j] = rf - rr

        # Species ODEs: dθ_i/dt = Σ(nu_ij * r_j)
        F[:n_sp] = nu @ r_net

        # Free site ODEs: dθ_*/dt = -Σ(n_i * dθ_i/dt) for species on that site
        for st_name in site_type_list:
            fi = free_site_idx[st_name]
            F[fi] = 0.0
            for sp_idx, n_occ in site_species_for_balance[st_name]:
                F[fi] -= n_occ * F[sp_idx]

        return F

    return rhs


def compute_reaction_rates(
    model: MKMModel, theta_species: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Compute forward, reverse, and net rates at given species coverages.

    Args:
        model: The MKM model.
        theta_species: Array of species coverages (length n_species, no free sites).

    Returns:
        Dict mapping reaction id -> {"forward": float, "reverse": float, "net": float}
    """
    T = model.temperature
    free_site_powers = _compute_free_site_powers(model)
    theta = np.clip(theta_species, 0.0, None)

    # Compute free site coverages from site balance
    theta_free = {}
    for st_name in model.site_types:
        occupied = sum(
            sp.n_sites * theta[model.species_index[sp.name]]
            for sp in model.surface_species.values()
            if sp.site_type == st_name
        )
        theta_free[st_name] = max(0.0, 1.0 - occupied)

    rates = {}
    for j, rxn in enumerate(model.reactions):
        # Base rate constants
        if rxn.reaction_type == "surface":
            kf, kr = rate_constants_surface(rxn.Ea, rxn.DeltaE, T)
        elif rxn.reaction_type == "adsorption":
            mass, n_sites_density = _get_adsorption_params(rxn, model)
            kf, kr = rate_constants_adsorption(rxn.DeltaE, T, mass, n_sites_density)

        # Lateral interaction correction on kr
        DeltaE_eff = rxn.DeltaE
        for sp in rxn.products:
            if sp in model.lateral_interactions:
                li = model.lateral_interactions[sp]
                idx = model.species_index[sp]
                delta = li.epsilon * max(0.0, theta[idx] - li.theta0)
                DeltaE_eff += delta
        if DeltaE_eff != rxn.DeltaE:
            K_eq_eff = np.exp(-DeltaE_eff / (KB * T))
            kr = kf / K_eq_eff

        powers = free_site_powers[j]

        # Forward rate
        rf = kf
        for sp, stoich in rxn.reactants.items():
            if sp in model.gas_species:
                rf *= model.gas_species[sp].pressure ** stoich
            elif sp in model.species_index:
                rf *= theta[model.species_index[sp]] ** stoich
        for st, (n_fwd, _) in powers.items():
            if n_fwd > 0:
                rf *= theta_free[st] ** n_fwd

        # Reverse rate
        rr = kr
        for sp, stoich in rxn.products.items():
            if sp in model.species_index:
                rr *= theta[model.species_index[sp]] ** stoich
        for st, (_, n_rev) in powers.items():
            if n_rev > 0:
                rr *= theta_free[st] ** n_rev

        rates[rxn.id] = {"forward": rf, "reverse": rr, "net": rf - rr}

    return rates


def build_jacobian(model: MKMModel) -> Callable:
    """Build and return a finite-difference Jacobian function."""
    rhs_func = build_rhs(model)

    def jac(t, y, eps=1e-8):
        n = len(y)
        f0 = rhs_func(t, y)
        J = np.zeros((n, n))
        for j in range(n):
            y_pert = y.copy()
            y_pert[j] += eps
            J[:, j] = (rhs_func(t, y_pert) - f0) / eps
        return J

    return jac
