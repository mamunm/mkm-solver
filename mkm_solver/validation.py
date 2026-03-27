"""Input validation for microkinetic model JSON files.

Checks structural and physical constraints before the model is built.
Called from load_model() after JSON parsing but before model construction.
"""

from typing import Dict, List, Set


VALID_REACTION_TYPES = {"adsorption", "surface"}


def validate_model_data(
    data: dict,
    gas_names: Set[str],
    site_type_names: Set[str],
) -> List[str]:
    """Validate parsed JSON data and return a list of error messages.

    Returns an empty list if validation passes.
    """
    errors = []

    errors.extend(_validate_reactions_structure(data.get("reactions", [])))
    errors.extend(_validate_reaction_types(data.get("reactions", [])))
    errors.extend(_validate_gas_species_placement(data.get("reactions", []), gas_names))
    errors.extend(
        _validate_adsorption_reactions(data.get("reactions", []), gas_names)
    )
    errors.extend(
        _validate_surface_reactions(data.get("reactions", []), gas_names)
    )
    errors.extend(
        _validate_species_naming(data.get("reactions", []), gas_names, site_type_names)
    )
    errors.extend(_validate_barriers(data.get("reactions", [])))

    return errors


def _validate_reactions_structure(reactions: List[dict]) -> List[str]:
    """Check that each reaction has all required fields."""
    errors = []
    required = {"type", "reactants", "products", "Ea", "DeltaE"}
    for i, rxn in enumerate(reactions):
        missing = required - set(rxn.keys())
        if missing:
            errors.append(
                f"R{i+1}: missing required fields: {', '.join(sorted(missing))}"
            )
    return errors


def _validate_reaction_types(reactions: List[dict]) -> List[str]:
    """Check that reaction types are valid (adsorption or surface only)."""
    errors = []
    for i, rxn in enumerate(reactions):
        rtype = rxn.get("type", "")
        if rtype not in VALID_REACTION_TYPES:
            errors.append(
                f"R{i+1}: invalid reaction type '{rtype}'. "
                f"Allowed: {', '.join(sorted(VALID_REACTION_TYPES))}"
            )
    return errors


def _validate_gas_species_placement(
    reactions: List[dict], gas_names: Set[str]
) -> List[str]:
    """Gas species must only appear in reactants, never in products."""
    errors = []
    for i, rxn in enumerate(reactions):
        products = rxn.get("products", {})
        for sp in products:
            if sp in gas_names:
                errors.append(
                    f"R{i+1}: gas species '{sp}' found in products. "
                    f"Gas species must only appear in reactants. "
                    f"Rewrite the reaction in the adsorption direction."
                )
    return errors


def _validate_adsorption_reactions(
    reactions: List[dict], gas_names: Set[str]
) -> List[str]:
    """Adsorption reactions must have exactly one gas species in reactants."""
    errors = []
    for i, rxn in enumerate(reactions):
        if rxn.get("type") != "adsorption":
            continue
        reactants = rxn.get("reactants", {})
        gas_in_reactants = [sp for sp in reactants if sp in gas_names]
        if len(gas_in_reactants) == 0:
            errors.append(
                f"R{i+1}: adsorption reaction has no gas species in reactants"
            )
        elif len(gas_in_reactants) > 1:
            errors.append(
                f"R{i+1}: adsorption reaction has multiple gas species "
                f"in reactants: {gas_in_reactants}"
            )
    return errors


def _validate_surface_reactions(
    reactions: List[dict], gas_names: Set[str]
) -> List[str]:
    """Surface reactions must not contain any gas species."""
    errors = []
    for i, rxn in enumerate(reactions):
        if rxn.get("type") != "surface":
            continue
        all_species = (
            list(rxn.get("reactants", {}).keys())
            + list(rxn.get("products", {}).keys())
        )
        gas_found = [sp for sp in all_species if sp in gas_names]
        if gas_found:
            errors.append(
                f"R{i+1}: surface reaction contains gas species: {gas_found}. "
                f"Use type 'adsorption' for reactions involving gas species."
            )
    return errors


def _validate_species_naming(
    reactions: List[dict],
    gas_names: Set[str],
    site_type_names: Set[str],
) -> List[str]:
    """Validate species naming conventions.

    Reactions must contain only gas species and surface species.
    Free sites (*_X) must NOT appear — they are auto-balanced by the code.
    Surface species must follow Name*_SiteType convention (e.g., NH3*_A).
    """
    errors = []
    for i, rxn in enumerate(reactions):
        all_species = (
            list(rxn.get("reactants", {}).keys())
            + list(rxn.get("products", {}).keys())
        )
        for sp in all_species:
            if sp in gas_names:
                continue
            if sp.startswith("*_") or sp == "*":
                errors.append(
                    f"R{i+1}: free site '{sp}' must not appear in reactions. "
                    f"Free sites are auto-balanced by the code."
                )
                continue
            if "*_" not in sp:
                errors.append(
                    f"R{i+1}: species '{sp}' does not match naming convention "
                    f"'Name*_SiteType' (e.g., NH3*_A)"
                )
                continue
            site_type = sp.split("*_")[1]
            if site_type not in site_type_names:
                errors.append(
                    f"R{i+1}: species '{sp}' references unknown "
                    f"site type '{site_type}'"
                )
    return errors


def _validate_barriers(reactions: List[dict]) -> List[str]:
    """Validate activation energy constraints.

    For surface reactions: Ea >= 0 and reverse barrier (Ea - DeltaE) >= 0.
    For adsorption: Ea is not used in k_f (collision theory), so only check
    that it is non-negative. Reverse barrier check is skipped since k_r
    comes from K_eq directly.
    """
    errors = []
    for i, rxn in enumerate(reactions):
        Ea = rxn.get("Ea")
        DeltaE = rxn.get("DeltaE")
        if Ea is None or DeltaE is None:
            continue

        if Ea < -1e-10:
            errors.append(f"R{i+1}: Ea must be non-negative, got {Ea}")

        if rxn.get("type") == "surface":
            reverse_barrier = Ea - DeltaE
            if reverse_barrier < -1e-10:
                errors.append(
                    f"R{i+1}: reverse barrier (Ea - DeltaE = {reverse_barrier:.4f}) "
                    f"is negative"
                )
    return errors
