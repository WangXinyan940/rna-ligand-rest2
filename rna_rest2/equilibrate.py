"""EM → NVT → NPT equilibration with heavy-atom positional restraints."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from openmm import (
    CustomExternalForce,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
    XmlSerializer,
    app,
    unit,
)
from openmm.app import (
    DCDReporter,
    ForceField,
    MinimizationReporter,
    PDBFile,
    Simulation,
    StateDataReporter,
)

HEAVY_ATOMS = {"C", "N", "O", "P", "S", "MG", "ZN", "FE"}


def _add_restraints(
    system,
    topology: app.Topology,
    positions: unit.Quantity,
    k: float = 1000.0,  # kJ/mol/nm^2
) -> int:
    """
    Add flat-bottom harmonic positional restraints on heavy atoms.
    Returns the force index in the system.
    """
    restraint = CustomExternalForce(
        "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    restraint.addGlobalParameter("k", k * unit.kilojoules_per_mole / unit.nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    pos_nm = positions.value_in_unit(unit.nanometer)
    for atom in topology.atoms():
        if atom.element is not None and atom.element.symbol in HEAVY_ATOMS:
            # Only restrain solute (non-water, non-ion)
            res_name = atom.residue.name.upper()
            if res_name not in ("HOH", "WAT", "TIP3", "TIP3P", "NA", "CL", "K"):
                x0, y0, z0 = pos_nm[atom.index]
                restraint.addParticle(atom.index, [x0, y0, z0])

    force_idx = system.addForce(restraint)
    return force_idx


def run_minimization(
    simulation: Simulation,
    max_iter: int = 5000,
    tolerance: float = 10.0,
) -> None:
    """Energy minimization."""
    simulation.minimizeEnergy(
        maxIterations=max_iter,
        tolerance=tolerance * unit.kilojoule_per_mole / unit.nanometer,
    )


def run_nvt(
    simulation: Simulation,
    n_steps: int = 50000,
    temperature: unit.Quantity = 300 * unit.kelvin,
    report_interval: int = 5000,
    out_prefix: str = "nvt",
) -> None:
    """NVT equilibration with heavy-atom restraints (caller sets integrator temp)."""
    simulation.reporters.clear()
    simulation.reporters.append(
        StateDataReporter(
            f"{out_prefix}_state.csv",
            report_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
        )
    )
    simulation.step(n_steps)


def run_npt(
    simulation: Simulation,
    system,
    n_steps: int = 100000,
    temperature: unit.Quantity = 300 * unit.kelvin,
    pressure: unit.Quantity = 1.0 * unit.bar,
    report_interval: int = 5000,
    out_prefix: str = "npt",
) -> None:
    """NPT equilibration. Adds a MonteCarloBarostat to the system."""
    barostat = MonteCarloBarostat(pressure, temperature)
    baro_idx = system.addForce(barostat)

    simulation.context.reinitialize(preserveState=True)
    simulation.reporters.clear()
    simulation.reporters.append(
        StateDataReporter(
            f"{out_prefix}_state.csv",
            report_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            volume=True,
        )
    )
    simulation.reporters.append(
        DCDReporter(f"{out_prefix}_traj.dcd", report_interval)
    )
    simulation.step(n_steps)

    # Remove barostat after NPT (REST2 production runs in NVT-like ensemble per replica)
    system.removeForce(baro_idx)
    simulation.context.reinitialize(preserveState=True)


def equilibrate_conformation(
    topology: app.Topology,
    positions: unit.Quantity,
    ff: ForceField,
    temperature: unit.Quantity = 300 * unit.kelvin,
    em_max_iter: int = 5000,
    nvt_steps: int = 50_000,
    npt_steps: int = 100_000,
    restraint_k: float = 1000.0,
    dt: float = 0.002,  # ps
    out_prefix: str = "equil",
    platform_name: str = "CUDA",
    platform_properties: Optional[dict] = None,
) -> tuple:
    """
    Full EM -> NVT -> NPT equilibration for a single conformation.

    Returns
    -------
    system : openmm.System
        The equilibrated system (with restraints removed).
    state_positions : unit.Quantity
        Final positions after NPT.
    box_vectors : unit.Quantity
        Final periodic box vectors.
    """
    nonbonded_method = app.PME
    system = ff.createSystem(
        topology,
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
    )

    # Add restraints
    _add_restraints(system, topology, positions, k=restraint_k)

    integrator = LangevinMiddleIntegrator(
        temperature, 1.0 / unit.picosecond, dt * unit.picosecond
    )

    props = platform_properties or {}
    try:
        platform = Platform.getPlatformByName(platform_name)
        simulation = Simulation(topology, system, integrator, platform, props)
    except Exception:
        # Fallback to CPU
        simulation = Simulation(topology, system, integrator)

    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    # EM
    run_minimization(simulation, max_iter=em_max_iter)

    # NVT
    run_nvt(simulation, n_steps=nvt_steps, temperature=temperature, out_prefix=out_prefix + "_nvt")

    # NPT
    run_npt(
        simulation,
        system,
        n_steps=npt_steps,
        temperature=temperature,
        out_prefix=out_prefix + "_npt",
    )

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    final_positions = state.getPositions()
    box_vectors = state.getPeriodicBoxVectors()

    # Save checkpoint
    simulation.saveCheckpoint(out_prefix + "_equil.chk")

    # Save serialized system (without restraints for production)
    # Remove restraint force (last added)
    n_forces = system.getNumForces()
    # Find restraint CustomExternalForce
    for i in range(n_forces - 1, -1, -1):
        f = system.getForce(i)
        if isinstance(f, CustomExternalForce) and "x0" in f.getEnergyFunction():
            system.removeForce(i)
            break

    return system, final_positions, box_vectors
