"""EM -> NVT1 -> NVT2 -> NPT equilibration with heavy-atom positional restraints.

Design principles
-----------------
* Each phase builds a completely new Simulation (new System, new Integrator,
  new Context).  No ``reinitialize()`` calls are used anywhere.
* State (positions, velocities, box vectors) is transferred between phases
  via ``Context.getState`` / ``setPositions`` / ``setVelocities`` /
  ``setPeriodicBoxVectors``.
* All Context creation happens inside ``_equil_worker()``, which is always
  executed in a child process via ProcessPoolExecutor.  The main process
  never touches a Context, avoiding the OpenMM CUDA context inheritance bug.

Phases
------
  EM   – energy minimisation, no MD, restraints on heavy solute atoms
  NVT1 – short NVT at half the target temperature (gentle heating)
  NVT2 – NVT at target temperature
  NPT  – NPT at target temperature + 1 bar (box equilibration)

Returns the clean system XML (no restraints, no barostat) plus final
positions, velocities, and box vectors ready for REST2 production.
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
    Simulation,
    StateDataReporter,
)

HEAVY_ATOMS = {"C", "N", "O", "P", "S", "MG", "ZN", "FE"}
SOLVENT_RES = {"HOH", "WAT", "TIP3", "TIP3P", "NA", "CL", "K"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_restraint_force(
    topology: app.Topology,
    positions_nm: np.ndarray,
    k: float,
) -> CustomExternalForce:
    """Build a CustomExternalForce that restrains heavy solute atoms."""
    restraint = CustomExternalForce(
        "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    restraint.addGlobalParameter(
        "k", k * unit.kilojoules_per_mole / unit.nanometer ** 2
    )
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    for atom in topology.atoms():
        if (
            atom.element is not None
            and atom.element.symbol in HEAVY_ATOMS
            and atom.residue.name.upper() not in SOLVENT_RES
        ):
            x0, y0, z0 = positions_nm[atom.index]
            restraint.addParticle(atom.index, [x0, y0, z0])
    return restraint


def _build_sim(
    topology: app.Topology,
    system_xml: str,
    temperature_K: float,
    dt: float,
    platform_name: str,
    platform_properties: dict,
    extra_forces: list,          # list of openmm Force objects to add
) -> Simulation:
    """
    Deserialize a fresh System from XML, attach extra forces, build
    a new Simulation with a new LangevinMiddleIntegrator and Context.
    """
    system = XmlSerializer.deserialize(system_xml)
    for force in extra_forces:
        system.addForce(force)

    integrator = LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        dt * unit.picosecond,
    )
    try:
        platform = Platform.getPlatformByName(platform_name)
        sim = Simulation(topology, system, integrator, platform, platform_properties)
    except Exception:
        sim = Simulation(topology, system, integrator)
    return sim


def _get_state(
    sim: Simulation,
    enforce_pbc: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (positions_nm, velocities_nm_per_ps, box_nm) from a Simulation.
    """
    state = sim.context.getState(
        getPositions=True,
        getVelocities=True,
        enforcePeriodicBox=enforce_pbc,
    )
    pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    vel_nm_ps = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    box_vecs = state.getPeriodicBoxVectors()
    box_nm = np.array(
        [[v.x, v.y, v.z]
         for v in [b.value_in_unit(unit.nanometer) for b in box_vecs]],
        dtype=np.float64,
    )
    return pos_nm, vel_nm_ps, box_nm


def _set_state(
    sim: Simulation,
    pos_nm: np.ndarray,
    vel_nm_ps: np.ndarray | None,
    box_nm: np.ndarray | None,
) -> None:
    """Push positions, optional velocities, and optional box into a Simulation."""
    if box_nm is not None:
        sim.context.setPeriodicBoxVectors(
            *[unit.Quantity(v, unit.nanometer) for v in box_nm]
        )
    sim.context.setPositions(
        unit.Quantity(pos_nm, unit.nanometer)
    )
    if vel_nm_ps is not None:
        sim.context.setVelocities(
            unit.Quantity(vel_nm_ps, unit.nanometer / unit.picosecond)
        )


def _add_reporters(
    sim: Simulation,
    prefix: str,
    report_interval: int,
    dcd: bool = False,
    extra_props: dict | None = None,
) -> None:
    props = dict(step=True, potentialEnergy=True, temperature=True)
    if extra_props:
        props.update(extra_props)
    sim.reporters.append(
        StateDataReporter(f"{prefix}.csv", report_interval, **props)
    )
    if dcd:
        sim.reporters.append(DCDReporter(f"{prefix}.dcd", report_interval))


# ---------------------------------------------------------------------------
# Worker function: runs entirely inside a child process
# ---------------------------------------------------------------------------

def _equil_worker(
    conf_id: int,
    topology: app.Topology,
    system_xml: str,
    positions_nm: np.ndarray,
    box_vectors_nm: np.ndarray | None,
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
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Four-phase equilibration in a child process.

    Returns
    -------
    system_xml_clean : str
        Serialized System with no restraints and no barostat.
    final_positions_nm  : np.ndarray [n_atoms, 3]
    final_velocities_nm_ps : np.ndarray [n_atoms, 3]
    final_box_nm        : np.ndarray [3, 3]
    """
    os.makedirs(
        os.path.dirname(out_prefix) if os.path.dirname(out_prefix) else ".",
        exist_ok=True,
    )
    tag = f"[conf {conf_id}]"
    T_half = max(50.0, temperature_K / 2.0)

    # Reference positions used to pin restraint equilibrium points.
    # We use the *initial* (pre-EM) positions so the restraint origin
    # does not move from the input structure.
    ref_pos_nm = positions_nm.copy()

    # ==================================================================
    # Phase 0: Energy Minimisation
    # ==================================================================
    print(f"{tag} Phase 0: EM start", flush=True)
    restraint_em = _make_restraint_force(topology, ref_pos_nm, restraint_k)
    sim_em = _build_sim(
        topology, system_xml, temperature_K, dt,
        platform_name, platform_properties,
        extra_forces=[restraint_em],
    )
    _set_state(sim_em, positions_nm, vel_nm_ps=None, box_nm=box_vectors_nm)
    sim_em.context.setVelocitiesToTemperature(temperature_K * unit.kelvin)
    sim_em.minimizeEnergy(
        maxIterations=em_max_iter,
        tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
    )
    pos_em, vel_em, box_em = _get_state(sim_em)
    del sim_em
    print(f"{tag} Phase 0: EM done", flush=True)

    # ==================================================================
    # Phase 1: NVT at T/2  (gentle heating with restraints)
    # ==================================================================
    print(f"{tag} Phase 1: NVT1 ({T_half:.1f} K) start", flush=True)
    restraint_nvt1 = _make_restraint_force(topology, ref_pos_nm, restraint_k)
    sim_nvt1 = _build_sim(
        topology, system_xml, T_half, dt,
        platform_name, platform_properties,
        extra_forces=[restraint_nvt1],
    )
    _set_state(sim_nvt1, pos_em, vel_em, box_em)
    _add_reporters(
        sim_nvt1, f"{out_prefix}_nvt1", report_interval,
    )
    sim_nvt1.step(nvt_steps)
    pos_nvt1, vel_nvt1, box_nvt1 = _get_state(sim_nvt1)
    del sim_nvt1
    print(f"{tag} Phase 1: NVT1 done", flush=True)

    # ==================================================================
    # Phase 2: NVT at target temperature  (restraints still on)
    # ==================================================================
    print(f"{tag} Phase 2: NVT2 ({temperature_K:.1f} K) start", flush=True)
    restraint_nvt2 = _make_restraint_force(topology, ref_pos_nm, restraint_k)
    sim_nvt2 = _build_sim(
        topology, system_xml, temperature_K, dt,
        platform_name, platform_properties,
        extra_forces=[restraint_nvt2],
    )
    _set_state(sim_nvt2, pos_nvt1, vel_nvt1, box_nvt1)
    _add_reporters(
        sim_nvt2, f"{out_prefix}_nvt2", report_interval,
    )
    sim_nvt2.step(nvt_steps)
    pos_nvt2, vel_nvt2, box_nvt2 = _get_state(sim_nvt2)
    del sim_nvt2
    print(f"{tag} Phase 2: NVT2 done", flush=True)

    # ==================================================================
    # Phase 3: NPT at target temperature + 1 bar  (restraints still on)
    # ==================================================================
    print(f"{tag} Phase 3: NPT ({temperature_K:.1f} K, 1 bar) start", flush=True)
    restraint_npt = _make_restraint_force(topology, ref_pos_nm, restraint_k)
    barostat = MonteCarloBarostat(1.0 * unit.bar, temperature_K * unit.kelvin)
    sim_npt = _build_sim(
        topology, system_xml, temperature_K, dt,
        platform_name, platform_properties,
        extra_forces=[restraint_npt, barostat],
    )
    _set_state(sim_npt, pos_nvt2, vel_nvt2, box_nvt2)
    _add_reporters(
        sim_npt, f"{out_prefix}_npt", report_interval, dcd=True,
        extra_props={"density": True, "volume": True},
    )
    sim_npt.step(npt_steps)
    pos_npt, vel_npt, box_npt = _get_state(sim_npt)
    del sim_npt
    print(f"{tag} Phase 3: NPT done", flush=True)

    # Return clean system_xml: deserialize once, add nothing, serialize back.
    # This is the System that REST2 production will use (no restraints, no barostat).
    system_clean = XmlSerializer.deserialize(system_xml)
    system_xml_clean = XmlSerializer.serialize(system_clean)

    return system_xml_clean, pos_npt, vel_npt, box_npt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def equilibrate_all_conformations(
    conformations: list,
    system_xmls: list[str],
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
    Equilibrate all conformations in parallel using a ProcessPoolExecutor of
    ``n_workers`` child processes.  The main process never creates a Context.

    Parameters
    ----------
    conformations : list of (topology, positions_nm, box_nm_or_None)
    system_xmls   : pre-serialized System XML string per conformation

    Returns
    -------
    list of (system_xml_clean, positions_nm, velocities_nm_ps, box_nm)
      Each element corresponds to one input conformation.
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
                raise RuntimeError(
                    f"Equilibration of conformation {i} failed: {exc}"
                ) from exc

    return results
