"""Visualization utilities: coverage transients, energy diagrams, sensitivity charts."""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from mkm_solver.schema import MKMModel


def plot_coverage_vs_time(
    sol, model: MKMModel, output_path: str
) -> None:
    """Plot transient coverage evolution from the solve_ivp solution object."""
    fig, ax = plt.subplots(figsize=(10, 6))

    species_names = list(model.surface_species.keys())
    for i, name in enumerate(species_names):
        ax.plot(sol.t, sol.y[i, :], label=name, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Coverage")
    ax.set_title(f"Coverage vs Time — {model.model_name}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_energy_diagram(model: MKMModel, output_path: str) -> None:
    """Plot reaction energy profile along the pathway.

    Each reaction is shown as: reactant level -> TS peak -> product level.
    The first reactant state is set to 0 eV and subsequent states are
    computed from the reaction energies (DeltaE).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n_rxn = len(model.reactions)
    energy_levels = [0.0]
    for rxn in model.reactions:
        energy_levels.append(energy_levels[-1] + rxn.DeltaE)

    x_positions = []
    y_positions = []
    labels = []

    for i, rxn in enumerate(model.reactions):
        x_start = i * 3
        x_ts = x_start + 1.5
        x_end = x_start + 3

        E_start = energy_levels[i]
        E_ts = E_start + rxn.Ea
        E_end = energy_levels[i + 1]

        ax.plot([x_start, x_start + 0.8], [E_start, E_start], "b-", linewidth=2)
        ax.plot([x_end - 0.8, x_end], [E_end, E_end], "b-", linewidth=2)

        ax.plot(
            [x_start + 0.8, x_ts, x_end - 0.8],
            [E_start, E_ts, E_end],
            "r--", linewidth=1, alpha=0.7,
        )

        ax.annotate(
            f"{rxn.Ea:.2f}",
            xy=(x_ts, E_ts),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="red",
        )

        ax.text(
            x_start + 0.4, E_start - 0.08, rxn.id,
            ha="center", fontsize=7, color="blue",
        )

    ax.set_xlabel("Reaction Coordinate")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"Energy Diagram — {model.model_name}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(
    sensitivity_result: Dict, model: MKMModel, output_path: str
) -> None:
    """Plot horizontal bar charts for DRC and DTRC values."""
    has_drc = "drc" in sensitivity_result
    has_dtrc = "dtrc" in sensitivity_result
    n_plots = has_drc + has_dtrc

    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, max(4, len(model.reactions) * 0.4)))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    if has_drc:
        ax = axes[plot_idx]
        drc = sensitivity_result["drc"]
        rxn_ids = list(drc.keys())
        values = [drc[r] for r in rxn_ids]
        colors = ["#d32f2f" if v > 0 else "#1565c0" for v in values]

        y_pos = np.arange(len(rxn_ids))
        ax.barh(y_pos, values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(rxn_ids, fontsize=8)
        ax.set_xlabel("X_RC")
        ax.set_title("Degree of Rate Control")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(True, axis="x", alpha=0.3)
        plot_idx += 1

    if has_dtrc:
        ax = axes[plot_idx]
        dtrc = sensitivity_result["dtrc"]
        rxn_ids = list(dtrc.keys())
        values = [dtrc[r] for r in rxn_ids]
        colors = ["#d32f2f" if v > 0 else "#1565c0" for v in values]

        y_pos = np.arange(len(rxn_ids))
        ax.barh(y_pos, values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(rxn_ids, fontsize=8)
        ax.set_xlabel("X_TRC")
        ax.set_title("Degree of Thermodynamic Rate Control")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Sensitivity Analysis — {model.model_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
