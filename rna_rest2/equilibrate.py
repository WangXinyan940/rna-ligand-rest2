"""EM -> NVT -> NPT equilibration with heavy-atom positional restraints.

All OpenMM Context creation happens inside _equil_worker(), which is
always executed in a *child* process via ProcessPoolExecutor.  The main
process never touches a Context, avoiding the OpenMM CUDA context
inheritance bug.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

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
    PDBFile,
    Simulation,
    StateDataReporter,
)

HEAVY_ATOMS = {"C", "N", "O", "P", "S", "MG", "ZN", "FE"}
SOLVENT_RES = {"HOH", "WAT", "TIP3", "TIP3P", "NA", "CL", "K"}


# ---------------------------------------------------------------------------
# Internal helpers (run inside child process)
# ---------------------------------------------------------------------------

def _add_restraints(system, topology: app.Topology, positions: unit.Quantity,
                    k: float = 1000.0) -> None:
    restraint = CustomExternalForce(
        "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    restraint.addGlobalParameter(
        "k", k * unit.kilojoules_per_mole / unit.nanometer ** 2
    )
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    pos_nm = positions.value_in_unit(unit.nanometer)
    for atom in topology.atoms():
        if (
            atom.element is not None
            and atom.element.symbol in HEAVY_ATOMS
            and atom.residue.name.upper() not in SOLVENT_RES
        ):
            x0, y0, z0 = pos_nm[atom.index]
            restraint.addParticle(atom.index, [x0, y0, z0])
    system.addForce(restraint)


def _remove_restraints(system) -> None:
    """Remove all CustomExternalForce restraints from the system."""
    to_remove = [
        i for i in range(system.getNumForces())
        if isinstance(system.getForce(i), CustomExternalForce)
        and "x0" in system.getForce(i).getEnergyFunction()
    ]
    for i in reversed(to_remove):
        system.removeForce(i)


def _make_simulation(
    topology: app.Topology,
    system,
    temperature: unit.Quantity,
    dt: float,
    platform_name: str,
    platform_properties: dict,
) -> Simulation:
    integrator = LangevinMiddleIntegrator(
        temperature, 1.0 / unit.picosecond, dt * unit.picosecond
    )
    try:
        platform = Platform.getPlatformByName(platform_name)
        sim = Simulation(topology, system, integrator, platform, platform_properties)
    except Exception:
        sim = Simulation(topology, system, integrator)
    return sim


# ---------------------------------------------------------------------------
# Worker function: runs entirely inside a child process
# ---------------------------------------------------------------------------

def _equil_worker(
    conf_id: int,
    topology: app.Topology,
    system_xml: str,           # serialized System (with no context yet)
    positions_nm: np.ndarray,  # [n_atoms, 3]
    box_vectors_nm: np.ndarray | None,  # [3, 3] or None
    temperature_K: float,
    em_max_iter: int,
    nvt_steps: int,
    npt_steps: int,
    restraint_k: float,
    dt: float,
    report_interval: int,
    out_prefix: str,
    platform_name: str,
    platform_properties: dict,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Runs EM -> NVT -> NPT in a child process.

    Returns
    -------
    system_xml_clean : str
        Serialized System with restraints removed (ready for REST2 production).
    final_positions_nm : np.ndarray  [n_atoms, 3]
    final_box_nm : np.ndarray        [3, 3]
    """
    os.makedirs(os.path.dirname(out_prefix) if os.path.dirname(out_prefix) else ".",
                exist_ok=True)

    temperature = temperature_K * unit.kelvin
    positions = unit.Quantity(positions_nm, unit.nanometer)

    # Deserialize system fresh in this process
    system = XmlSerializer.deserialize(system_xml)

    # Add restraints
    _add_restraints(system, topology, positions, k=restraint_k)

    sim = _make_simulation(topology, system, temperature, dt,
                           platform_name, platform_properties)

    if box_vectors_nm is not None:
        sim.context.setPeriodicBoxVectors(
            *[unit.Quantity(v, unit.nanometer) for v in box_vectors_nm]
        )
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(temperature)

    # --- EM ---
    sim.minimizeEnergy(
        maxIterations=em_max_iter,
        tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
    )
    print(f"[conf {conf_id}] EM done", flush=True)

    # --- NVT1 ---
    sim.reporters.append(
        StateDataReporter(
            f"{out_prefix}_nvt1.csv", report_interval,
            step=True, potentialEnergy=True, temperature=True,
        )
    )
    sim.reporters.append(DCDReporter(f"{out_prefix}_nvt1.dcd", 1))
    sim.integrator.setTemperature(150.0 * unit.kelvin)  # start cold to avoid instabilities
    sim.context.reinitialize(preserveState=True)
    sim.step(nvt_steps)
    sim.reporters.clear()
    print(f"[conf {conf_id}] NVT1 done", flush=True)

    # --- NVT2 ---
    sim.reporters.append(
        StateDataReporter(
            f"{out_prefix}_nvt2.csv", report_interval,
            step=True, potentialEnergy=True, temperature=True,
        )
    )
    sim.integrator.setTemperature(300.0 * unit.kelvin)  # start cold to avoid instabilities
    sim.context.reinitialize(preserveState=True)
    sim.step(nvt_steps)
    sim.reporters.clear()
    print(f"[conf {conf_id}] NVT2 done", flush=True)

    # --- NPT ---
    barostat = MonteCarloBarostat(1.0 * unit.bar, temperature)
    baro_idx = system.addForce(barostat)
    sim.context.reinitialize(preserveState=True)
    sim.reporters.append(
        StateDataReporter(
            f"{out_prefix}_npt.csv", report_interval,
            step=True, potentialEnergy=True, temperature=True,
            density=True, volume=True,
        )
    )
    sim.reporters.append(DCDReporter(f"{out_prefix}_npt.dcd", report_interval))
    sim.step(npt_steps)
    sim.reporters.clear()
    print(f"[conf {conf_id}] NPT done", flush=True)

    # Save checkpoint
    sim.saveCheckpoint(f"{out_prefix}_equil.chk")

    # Collect final state
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    final_pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    box = sim.context.getState().getPeriodicBoxVectors()
    final_box_nm = np.array([
        [v.x, v.y, v.z] for v in
        [box[0].value_in_unit(unit.nanometer),
         box[1].value_in_unit(unit.nanometer),
         box[2].value_in_unit(unit.nanometer)]
    ])

    # Clean system: remove restraints and barostat, re-serialize
    system.removeForce(baro_idx)
    _remove_restraints(system)
    system_xml_clean = XmlSerializer.serialize(system)

    return system_xml_clean, final_pos_nm, final_box_nm


# ---------------------------------------------------------------------------
# Public API: submits work to a ProcessPoolExecutor
# ---------------------------------------------------------------------------

def equilibrate_all_conformations(
    conformations: list,              # list of (topology, positions_nm, box_nm_or_None)
    system_xmls: list[str],           # pre-serialized System per conformation
    temperature_K: float = 300.0,
    em_max_iter: int = 5000,
    nvt_steps: int = 50_000,
    npt_steps: int = 100_000,
    restraint_k: float = 1000.0,
    dt: float = 0.002,
    report_interval: int = 5000,
    out_dir: str = "output",
    platform_name: str = "CUDA",
    platform_properties: dict | None = None,
    n_workers: int = 4,
) -> list:
    """
    Equilibrate all conformations in parallel using a ProcessPool of
    `n_workers` processes.  The main process never creates a Context.

    Returns
    -------
    list of (system_xml_clean: str, final_pos_nm: np.ndarray, final_box_nm: np.ndarray)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    props = platform_properties or {"Precision": "mixed"}
    futures = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for i, ((top, pos_nm, box_nm), sys_xml) in enumerate(
            zip(conformations, system_xmls)
        ):
            out_prefix = os.path.join(out_dir, f"conf{i}", "equil")
            os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
            fut = pool.submit(
                _equil_worker,
                i, top, sys_xml, pos_nm, box_nm,
                temperature_K, em_max_iter, nvt_steps, npt_steps,
                restraint_k, dt, report_interval, out_prefix,
                platform_name, props,
            )
            futures[fut] = i

        results = [None] * len(conformations)
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
                print(f"[INFO] Conformation {i} equilibration complete.", flush=True)
            except Exception as exc:
                raise RuntimeError(f"Equilibration of conformation {i} failed: {exc}") from exc

    return results
