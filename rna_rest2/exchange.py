"""Metropolis MC exchange logic for conformation swaps and replica exchange."""
from __future__ import annotations

import math
from typing import List

import numpy as np
from openmm import unit, app
from openmm.app import Simulation

KB = 8.314462618e-3  # kJ/mol/K


def metropolis_accept(delta_e_kj: float, temperature_K: float, rng: np.random.Generator) -> bool:
    """
    Standard Metropolis criterion.
    Accepts if delta_E <= 0 or with probability exp(-delta_E / kT).
    """
    if delta_e_kj <= 0.0:
        return True
    kT = KB * temperature_K
    prob = math.exp(-delta_e_kj / kT)
    return rng.random() < prob


def compute_potential_energy(simulation: Simulation) -> float:
    """Get potential energy in kJ/mol."""
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def attempt_conformation_swap(
    simulation: Simulation,
    topology: app.Topology,
    conformation_pool: List[np.ndarray],  # list of [n_atoms, 3] arrays in nm
    temperature: unit.Quantity,
    rng: np.random.Generator,
) -> bool:
    """
    Attempt a Metropolis MC swap of the current coordinates with a randomly
    chosen conformation from `conformation_pool`.

    Steps:
    1. Compute E_current
    2. Pick a random conformation j from pool
    3. Temporarily set positions to conf j
    4. Compute E_proposed
    5. Accept with Metropolis criterion
    6. If rejected, restore original positions

    Returns True if swap was accepted.
    """
    T_K = temperature.value_in_unit(unit.kelvin)

    # Save current state
    state_current = simulation.context.getState(
        getPositions=True, getVelocities=True, enforcePeriodicBox=True
    )
    e_current = compute_potential_energy(simulation)

    # Pick a random conformation (exclude current if identifiable, just pick randomly)
    j = rng.integers(0, len(conformation_pool))
    proposed_pos = conformation_pool[j]  # [n_atoms, 3] nm

    # Set proposed positions, keep box
    simulation.context.setPositions(
        unit.Quantity(proposed_pos, unit.nanometer)
    )
    # Randomize velocities for fair comparison
    simulation.context.setVelocitiesToTemperature(temperature)

    e_proposed = compute_potential_energy(simulation)

    delta_e = e_proposed - e_current
    accepted = metropolis_accept(delta_e, T_K, rng)

    if not accepted:
        # Restore previous state
        simulation.context.setState(state_current)
    else:
        # Update the pool entry with the previous current conformation
        old_pos = state_current.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        conformation_pool[j] = old_pos

    return accepted


def attempt_replica_exchange(
    e_i: float,
    e_j: float,
    T_i: float,
    T_j: float,
    rng: np.random.Generator,
) -> bool:
    """
    Attempt temperature-based replica exchange between replica i (T_i) and j (T_j).
    Uses the standard REMD Metropolis criterion:
      delta = (1/kT_i - 1/kT_j) * (E_j - E_i)
    """
    beta_i = 1.0 / (KB * T_i)
    beta_j = 1.0 / (KB * T_j)
    delta = (beta_i - beta_j) * (e_j - e_i)
    if delta <= 0.0:
        return True
    return rng.random() < math.exp(-delta)
