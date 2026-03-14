"""Single replica worker process for REST2 simulation."""
from __future__ import annotations

import os
import time
from multiprocessing import shared_memory
from typing import List, Optional, Tuple

import numpy as np
from openmm import (
    LangevinMiddleIntegrator,
    Platform,
    Simulation,
    XmlSerializer,
    app,
    unit,
)
from openmm.app import DCDReporter, StateDataReporter

from .rest2 import apply_rest2_scaling, get_solute_atom_indices, store_original_parameters
from .exchange import attempt_conformation_swap, metropolis_accept


class ReplicaWorker:
    """
    Manages a single REST2 replica:
      - Holds the OpenMM Simulation
      - Holds a local copy of multi-conformation snapshots
      - Writes coordinates to shared memory for exchange
      - Reads coordinates from shared memory when a swap is accepted
    """

    def __init__(
        self,
        replica_id: int,
        topology: app.Topology,
        system_xml: str,
        conformation_positions: List[np.ndarray],  # [n_conf, n_atoms, 3] in nm
        initial_positions: np.ndarray,             # [n_atoms, 3] nm
        box_vectors: np.ndarray,                   # [3, 3] nm
        temperature: float,                        # Kelvin
        reference_temperature: float,              # Kelvin (T0)
        shm_name: str,                             # shared memory block name
        shm_shape: Tuple[int, int],                # (n_replicas, n_atoms*3)
        shm_dtype: np.dtype,
        dt: float = 0.002,                         # ps
        n_steps_per_block: int = 500,
        swap_interval: int = 500,                  # steps between conformation swaps
        out_dir: str = "output",
        platform_name: str = "CUDA",
        platform_properties: Optional[dict] = None,
    ):
        self.replica_id = replica_id
        self.topology = topology
        self.temperature = temperature * unit.kelvin
        self.reference_temperature = reference_temperature * unit.kelvin
        self.lam = reference_temperature / temperature  # REST2 lambda
        self.conformation_positions = conformation_positions  # list of np arrays
        self.n_steps_per_block = n_steps_per_block
        self.swap_interval = swap_interval
        self.out_dir = out_dir
        self.shm_name = shm_name
        self.shm_shape = shm_shape
        self.shm_dtype = shm_dtype

        system = XmlSerializer.deserialize(system_xml)
        self.original_params = store_original_parameters(system, topology)

        integrator = LangevinMiddleIntegrator(
            self.temperature, 1.0 / unit.picosecond, dt * unit.picosecond
        )

        props = platform_properties or {}
        try:
            platform = Platform.getPlatformByName(platform_name)
            self.simulation = Simulation(topology, system, integrator, platform, props)
        except Exception:
            self.simulation = Simulation(topology, system, integrator)

        # Set initial state
        self.simulation.context.setPositions(
            unit.Quantity(initial_positions, unit.nanometer)
        )
        self.simulation.context.setPeriodicBoxVectors(
            *[unit.Quantity(v, unit.nanometer) for v in box_vectors]
        )
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

        # Apply REST2 scaling
        apply_rest2_scaling(
            system,
            self.simulation.context,
            topology,
            self.original_params,
            self.lam,
        )

        # Setup reporters
        os.makedirs(out_dir, exist_ok=True)
        prefix = os.path.join(out_dir, f"replica_{replica_id}")
        self.simulation.reporters.append(
            DCDReporter(f"{prefix}.dcd", n_steps_per_block)
        )
        self.simulation.reporters.append(
            StateDataReporter(
                f"{prefix}_state.csv",
                n_steps_per_block,
                step=True,
                potentialEnergy=True,
                temperature=True,
                density=True,
                volume=True,
            )
        )

    def _get_positions_nm(self) -> np.ndarray:
        state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        return state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    def _set_positions_nm(self, pos_nm: np.ndarray) -> None:
        self.simulation.context.setPositions(
            unit.Quantity(pos_nm, unit.nanometer)
        )
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

    def _write_to_shm(self, pos_nm: np.ndarray) -> None:
        shm = shared_memory.SharedMemory(name=self.shm_name)
        arr = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=shm.buf)
        arr[self.replica_id, :] = pos_nm.flatten()
        shm.close()

    def _read_from_shm(self, replica_j: int) -> np.ndarray:
        shm = shared_memory.SharedMemory(name=self.shm_name)
        arr = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=shm.buf)
        flat = arr[replica_j, :].copy()
        shm.close()
        n_atoms = self.shm_shape[1] // 3
        return flat.reshape(n_atoms, 3)

    def get_potential_energy(self) -> float:
        """Return current potential energy in kJ/mol."""
        state = self.simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def run_steps(self, n_steps: int) -> None:
        self.simulation.step(n_steps)

    def attempt_conformation_swap_step(self, rng: np.random.Generator) -> bool:
        """
        Try to swap current conformation with one from the stored multi-conformation pool.
        Returns True if swap was accepted.
        """
        return attempt_conformation_swap(
            self.simulation,
            self.topology,
            self.conformation_positions,
            self.temperature,
            rng,
        )

    def sync_positions_to_shm(self) -> None:
        pos = self._get_positions_nm()
        self._write_to_shm(pos)


def replica_main(
    replica_id: int,
    topology,
    system_xml: str,
    conformation_positions: List[np.ndarray],
    initial_positions: np.ndarray,
    box_vectors: np.ndarray,
    temperature: float,
    reference_temperature: float,
    shm_name: str,
    shm_shape: tuple,
    shm_dtype,
    n_total_steps: int,
    n_steps_per_block: int,
    swap_interval: int,
    out_dir: str,
    platform_name: str,
    platform_properties: dict,
    seed: int,
    result_queue,
) -> None:
    """
    Entry point for a replica subprocess. Runs the full production simulation.
    """
    rng = np.random.default_rng(seed)
    worker = ReplicaWorker(
        replica_id=replica_id,
        topology=topology,
        system_xml=system_xml,
        conformation_positions=conformation_positions,
        initial_positions=initial_positions,
        box_vectors=box_vectors,
        temperature=temperature,
        reference_temperature=reference_temperature,
        shm_name=shm_name,
        shm_shape=shm_shape,
        shm_dtype=shm_dtype,
        dt=0.002,
        n_steps_per_block=n_steps_per_block,
        swap_interval=swap_interval,
        out_dir=out_dir,
        platform_name=platform_name,
        platform_properties=platform_properties,
    )

    steps_done = 0
    conf_swap_accepted = 0
    conf_swap_attempted = 0

    while steps_done < n_total_steps:
        block = min(n_steps_per_block, n_total_steps - steps_done)
        worker.run_steps(block)
        steps_done += block

        # Sync positions to shared memory
        worker.sync_positions_to_shm()

        # Conformation swap attempt
        if steps_done % swap_interval == 0 and len(conformation_positions) > 1:
            conf_swap_attempted += 1
            if worker.attempt_conformation_swap_step(rng):
                conf_swap_accepted += 1

    result_queue.put({
        "replica_id": replica_id,
        "conf_swap_rate": conf_swap_accepted / max(1, conf_swap_attempted),
    })
