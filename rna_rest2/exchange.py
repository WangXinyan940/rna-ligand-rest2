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
    conformation_pool: list,  # list of (pos_nm, vel_nm_ps, box_nm) tuples
    temperature: unit.Quantity,
    rng: np.random.Generator,
) -> bool:
    """
    Attempt a Metropolis MC swap of the current coordinates with a randomly
    chosen conformation from `conformation_pool`.

    Each pool entry is a tuple (positions_nm, velocities_nm_ps, box_vectors_nm)
    so that positions, velocities, and box vectors are swapped together.

    Steps:
    1. Compute E_current
    2. Pick a random conformation j from pool
    3. Temporarily set box vectors and positions to conf j
    4. Compute E_proposed
    5. Accept with Metropolis criterion
    6. If rejected, restore original state

    Returns True if swap was accepted.
    """
    T_K = temperature.value_in_unit(unit.kelvin)

    # Save current state
    state_current = simulation.context.getState(
        getPositions=True, getVelocities=True, enforcePeriodicBox=True
    )
    e_current = compute_potential_energy(simulation)

    # Pick a random conformation
    j = rng.integers(0, len(conformation_pool))
    proposed_pos, proposed_vel, proposed_box = conformation_pool[j]

    # Set proposed box vectors first, then positions
    simulation.context.setPeriodicBoxVectors(
        *[unit.Quantity(v, unit.nanometer) for v in proposed_box]
    )
    simulation.context.setPositions(
        unit.Quantity(proposed_pos, unit.nanometer)
    )
    # Randomize velocities for fair energy comparison (potential energy only)
    simulation.context.setVelocitiesToTemperature(temperature)

    e_proposed = compute_potential_energy(simulation)

    delta_e = e_proposed - e_current
    accepted = metropolis_accept(delta_e, T_K, rng)

    if not accepted:
        # Restore previous state (positions, velocities, box)
        simulation.context.setState(state_current)
    else:
        # Update the pool entry with the previous current conformation
        old_pos = state_current.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        old_vel = state_current.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
        old_box = state_current.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        conformation_pool[j] = (old_pos, old_vel, old_box)

        # Apply the pool conformer's velocities to the simulation
        simulation.context.setVelocities(
            unit.Quantity(proposed_vel, unit.nanometer / unit.picosecond)
        )

    return accepted


def attempt_replica_exchange(
    e_ii: float,
    e_jj: float,
    e_ij: float,
    e_ji: float,
    T_ref: float,
    rng: np.random.Generator,
) -> bool:
    """
    REST2 Hamiltonian replica exchange (HREX) Metropolis criterion.

    All replicas run at the same integrator temperature T_ref (= T_low).
    Effective temperatures are achieved via Hamiltonian scaling only.

    Parameters
    ----------
    e_ii : energy of replica i's coords under replica i's Hamiltonian, H_i(X_i)
    e_jj : energy of replica j's coords under replica j's Hamiltonian, H_j(X_j)
    e_ij : energy of replica j's coords under replica i's Hamiltonian, H_i(X_j)
    e_ji : energy of replica i's coords under replica j's Hamiltonian, H_j(X_i)
    T_ref : reference temperature T_low (K), same for all replicas

    Acceptance criterion:
      Δ = β_0 * [H_i(X_j) + H_j(X_i) - H_i(X_i) - H_j(X_j)]
      accept with min(1, exp(-Δ))
    """
    beta_0 = 1.0 / (KB * T_ref)
    delta = beta_0 * (e_ij + e_ji - e_ii - e_jj)
    if delta <= 0.0:
        return True
    return rng.random() < math.exp(-delta)
