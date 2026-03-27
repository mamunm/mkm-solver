"""Data model and JSON input parser for the microkinetic model solver."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from mkm_solver.validation import validate_model_data


VALID_METHODS = (
    "scipy-bdf", "scipy-lsoda", "scipy-radau",
    "sundials-cvode-bdf",
    "assimulo-cvode-bdf", "assimulo-radau5", "assimulo-ida-dae",
)


PRESSURE_UNITS = {
    "bar": 1e5,
    "atm": 101325.0,
    "Pa": 1.0,
    "pa": 1.0,
}


@dataclass
class SiteType:
    name: str
    n_sites: float  # surface site density in m^-2


@dataclass
class GasSpecies:
    name: str
    pressure: float  # stored internally in Pa
    mass: float      # molecular mass in amu


@dataclass
class SurfaceSpecies:
    name: str
    site_type: str
    n_sites: int = 1  # number of adsorption sites occupied (default 1)


@dataclass
class LateralInteraction:
    species: str
    theta0: float    # threshold coverage
    epsilon: float   # interaction energy in eV


@dataclass
class Reaction:
    id: str
    reaction_type: str          # "adsorption" or "surface"
    reactants: Dict[str, int]
    products: Dict[str, int]
    Ea: float                   # forward activation energy (eV)
    DeltaE: float               # reaction energy, products - reactants (eV)


@dataclass
class SolverSettings:
    method: str = "scipy-bdf"
    rtol: float = 1e-8
    atol: float = 1e-10
    t_span: List[float] = field(default_factory=lambda: [0, 1e6])
    max_step: float = 1e4
    steady_state_method: str = "integration"
    steady_state_tol: float = 1e-12
    fsolve_fallback: bool = True


@dataclass
class SensitivitySettings:
    enabled: bool = False
    delta: float = 0.01
    target_rate_reaction_id: str = ""
    compute_drc: bool = True
    compute_dtrc: bool = True


@dataclass
class MKMModel:
    model_name: str
    temperature: float
    site_types: Dict[str, SiteType]
    gas_species: Dict[str, GasSpecies]
    surface_species: Dict[str, SurfaceSpecies]
    reactions: List[Reaction]
    solver: SolverSettings
    sensitivity: SensitivitySettings
    lateral_interactions: Dict[str, LateralInteraction] = field(default_factory=dict)
    species_index: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.species_index = {
            name: i for i, name in enumerate(self.surface_species.keys())
        }


def _parse_pressure(pressure_data) -> float:
    """Parse pressure with units and convert to Pa."""
    if not isinstance(pressure_data, dict):
        raise ValueError(
            f"Pressure must be {{\"value\": ..., \"unit\": ...}}, got: {pressure_data}"
        )
    value = pressure_data.get("value")
    unit = pressure_data.get("unit")
    if value is None or unit is None:
        raise ValueError(
            f"Pressure must have 'value' and 'unit' keys, got: {pressure_data}"
        )
    if unit not in PRESSURE_UNITS:
        raise ValueError(
            f"Unknown pressure unit '{unit}'. Supported: {list(PRESSURE_UNITS.keys())}"
        )
    return value * PRESSURE_UNITS[unit]


def _infer_surface_species(
    reactions: List[dict],
    species_n_sites: Dict[str, int],
    gas_names: set,
    site_type_names: set,
) -> Dict[str, SurfaceSpecies]:
    """Infer surface species and their site types from reaction participants.

    Reactions contain only gas species and surface species (no free sites).
    Naming convention: species name contains '*_X' where X is the site type.
    E.g., NH3*_A -> site type 'A', H*_A -> site type 'A'.
    """
    surface_species = {}
    for rxn in reactions:
        all_sp = list(rxn["reactants"].keys()) + list(rxn["products"].keys())
        for sp in all_sp:
            if sp in gas_names:
                continue
            if sp in surface_species:
                continue
            if "*_" not in sp:
                raise ValueError(
                    f"Species '{sp}' is not a gas species and does not match "
                    f"surface species naming convention 'Name*_SiteType' (e.g., NH3*_A)"
                )
            site_type = sp.split("*_")[1]
            if site_type not in site_type_names:
                raise ValueError(
                    f"Surface species '{sp}' references unknown site type '{site_type}'"
                )
            n = species_n_sites.get(sp, 1)
            surface_species[sp] = SurfaceSpecies(
                name=sp, site_type=site_type, n_sites=n,
            )
    return surface_species


def load_model(json_path: str) -> MKMModel:
    """Parse a JSON input file and return a validated MKMModel."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {json_path}")

    with open(path) as f:
        data = json.load(f)

    for key in ("model_name", "temperature", "site_types", "gas_species", "reactions"):
        if key not in data:
            raise ValueError(f"Missing required field: '{key}'")

    # Parse site types: {"A": 1e19, "B": 1e19}
    site_types = {}
    for name, n_sites in data["site_types"].items():
        site_types[name] = SiteType(name=name, n_sites=float(n_sites))

    # Parse gas species with pressure (value+unit) and mass
    gas_species = {}
    for name, gs_data in data["gas_species"].items():
        if "pressure" not in gs_data:
            raise ValueError(f"Gas species '{name}' missing 'pressure'")
        if "mass" not in gs_data:
            raise ValueError(f"Gas species '{name}' missing 'mass' (amu)")
        pressure_pa = _parse_pressure(gs_data["pressure"])
        gas_species[name] = GasSpecies(
            name=name,
            pressure=pressure_pa,
            mass=float(gs_data["mass"]),
        )

    # Run validation before building model objects
    gas_names = set(gas_species.keys())
    site_type_names = set(site_types.keys())
    validation_errors = validate_model_data(data, gas_names, site_type_names)
    if validation_errors:
        msg = "Input validation failed:\n" + "\n".join(f"  - {e}" for e in validation_errors)
        raise ValueError(msg)

    # Parse species_n_sites: optional mapping of species -> number of sites occupied
    # e.g., {"NH3*_A": 2, "CN*_B": 2}  (default is 1 if not specified)
    species_n_sites = {}
    for sp_name, n in data.get("species_n_sites", {}).items():
        species_n_sites[sp_name] = int(n)

    # Parse reactions
    reactions = []
    for i, rxn_data in enumerate(data["reactions"]):
        reactions.append(Reaction(
            id=f"R{i+1}",
            reaction_type=rxn_data["type"],
            reactants=rxn_data["reactants"],
            products=rxn_data["products"],
            Ea=float(rxn_data["Ea"]),
            DeltaE=float(rxn_data["DeltaE"]),
        ))

    # Infer surface species from reactions
    surface_species = _infer_surface_species(
        data["reactions"], species_n_sites, gas_names, site_type_names,
    )

    # Parse solver settings
    solver_data = data.get("solver", {})
    method = solver_data.get("method", "scipy-bdf")
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown solver method '{method}'. "
            f"Valid methods: {', '.join(VALID_METHODS)}"
        )
    solver = SolverSettings(
        method=method,
        rtol=solver_data.get("rtol", 1e-8),
        atol=solver_data.get("atol", 1e-10),
        t_span=solver_data.get("t_span", [0, 1e6]),
        max_step=solver_data.get("max_step", 1e4),
        steady_state_method=solver_data.get("steady_state_method", "integration"),
        steady_state_tol=solver_data.get("steady_state_tol", 1e-12),
        fsolve_fallback=solver_data.get("fsolve_fallback", True),
    )

    # Parse sensitivity settings
    sens_data = data.get("sensitivity", {})
    sensitivity = SensitivitySettings(
        enabled=sens_data.get("enabled", False),
        delta=sens_data.get("delta", 0.01),
        target_rate_reaction_id=sens_data.get("target_rate_reaction_id", ""),
        compute_drc=sens_data.get("compute_drc", True),
        compute_dtrc=sens_data.get("compute_dtrc", True),
    )

    if sensitivity.enabled and not sensitivity.target_rate_reaction_id:
        raise ValueError(
            "Sensitivity analysis enabled but 'target_rate_reaction_id' not set"
        )

    return MKMModel(
        model_name=data["model_name"],
        temperature=data["temperature"],
        site_types=site_types,
        gas_species=gas_species,
        surface_species=surface_species,
        reactions=reactions,
        solver=solver,
        sensitivity=sensitivity,
    )


def load_lateral_interactions(json_path: str) -> Dict[str, LateralInteraction]:
    """Load lateral interaction parameters from a separate JSON file.

    Expected format: {"H*_A": {"theta0": 0.12, "epsilon": 0.15}, ...}
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Lateral interactions file not found: {json_path}"
        )

    with open(path) as f:
        data = json.load(f)

    interactions = {}
    for species, params in data.items():
        if "theta0" not in params or "epsilon" not in params:
            raise ValueError(
                f"Lateral interaction for '{species}' must have 'theta0' and 'epsilon'"
            )
        interactions[species] = LateralInteraction(
            species=species,
            theta0=float(params["theta0"]),
            epsilon=float(params["epsilon"]),
        )
    return interactions
