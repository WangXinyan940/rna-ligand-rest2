"""Single replica worker process for REST2/HREX simulation.

Architecture
------------
Each replica runs in its own subprocess.  Coordination for Hamiltonian
Replica Exchange (HREX) is done through two shared objects that are passed
in at construction time:

* ``exchange_barrier`` – a ``multiprocessing.Barrier`` with party count equal
  to the number of replicas.  All replicas synchronize here before and after
  each round of HREX neighbor exchanges so that energies are read from a
  consistent snapshot.

* ``hrex_queue``  – a ``multiprocessing.Queue`` used to pass exchange decisions
  from even replicas (which propose) to odd replicas (which accept/reject).  A
  replica with id *i* reads results destined for itself from ``hrex_queue``.

HREX exchange round
-------------------
After every ``hrex_interval`` MD steps the following happens:

1. All replicas sync to ``exchange_barrier`` (barrier-1).
2. Every replica writes its current positions and potential energy to shared
   memory.
3. All replicas sync again (barrier-2).
4. Even-phase (round even): replica 0-1, 2-3, 4-5, … attempt exchange.
   Odd-phase (round odd):  replica 1-2, 3-4, 5-6, … attempt exchange.
   The *lower-indexed* replica in each pair decides accept/reject and sets
   positions in shared memory; the higher-indexed replica reads back.
5. All replicas sync (barrier-3), then each replica applies the new positions
   if a swap was accepted.

Conformation library MC
-----------------------
Only the *hottest* replica (highest index = highest temperature) also attempts
conformation swaps from the pre-built library every ``conf_interval`` steps.
This injects structural diversity that percolates down to the cold replicas
through HREX.
"""
from __future__ import annotations

import os
from multiprocessing import shared_memory
from multiprocessing import Barrier, Queue
from typing import List, Optional, Tuple

import numpy as np
from openmm import (
    LangevinMiddleIntegrator,
    Platform,
    XmlSerializer,
    app,
    unit,
)
from openmm.app import DCDReporter, StateDataReporter, Simulation

from .rest2 import apply_rest2_scaling, store_original_parameters
from .exchange import attempt_conformation_swap, attempt_replica_exchange


# ---------------------------------------------------------------------------
# Shared-memory helpers
# ---------------------------------------------------------------------------

def _write_to_shm(
    shm_name: str,
    shm_shape: Tuple[int, int],
    shm_dtype,
    replica_id: int,
    pos_nm: np.ndarray,
) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)
    arr[replica_id, :] = pos_nm.flatten()
    shm.close()


def _read_from_shm(
    shm_name: str,
    shm_shape: Tuple[int, int],
    shm_dtype,
    replica_j: int,
) -> np.ndarray:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shm_shape, dtype=shm.buf)
    arr = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)
    flat = arr[replica_j, :].copy()
    shm.close()
    n_atoms = shm_shape[1] // 3
    return flat.reshape(n_atoms, 3)


# ---------------------------------------------------------------------------
# Shared-memory for energy exchange
# ---------------------------------------------------------------------------

def _write_energy_to_shm(
    energy_shm_name: str,
    n_replicas: int,
    replica_id: int,
    energy: float,
) -> None:
    shm = shared_memory.SharedMemory(name=energy_shm_name)
    arr = np.ndarray((n_replicas,), dtype=np.float64, buffer=shm.buf)
    arr[replica_id] = energy
    shm.close()


def _read_energy_from_shm(
    energy_shm_name: str,
    n_replicas: int,
    replica_j: int,
) -> float:
    shm = shared_memory.SharedMemory(name=energy_shm_name)
    arr = np.ndarray((n_replicas,), dtype=np.float64, buffer=shm.buf)
    val = float(arr[replica_j])
    shm.close()
    return val


# ---------------------------------------------------------------------------
# Replica worker
# ---------------------------------------------------------------------------

class ReplicaWorker:
    """
    Manages a single REST2 replica.
    """

    def __init__(
        self,
        replica_id: int,
        n_replicas: int,
        topology: app.Topology,
        system_xml: str,
        conformation_positions: List[np.ndarray],  # [n_conf, n_atoms, 3] in nm
        initial_positions: np.ndarray,             # [n_atoms, 3] nm
        box_vectors: np.ndarray,                   # [3, 3] nm
        temperature: float,                        # Kelvin
        reference_temperature: float,              # Kelvin (T0)
        temperatures: List[float],                 # all replica temperatures
        shm_name: str,                             # position shared memory block name
        shm_shape: Tuple[int, int],                # (n_replicas, n_atoms*3)
        shm_dtype: np.dtype,
        energy_shm_name: str,                      # energy shared memory block name
        exchange_barrier: Barrier,
        dt: float = 0.002,                         # ps
        n_steps_per_block: int = 500,
        hrex_interval: int = 500,                  # steps between HREX rounds
        conf_interval: int = 1000,                 # steps between conformation swaps (hottest replica only)
        out_dir: str = "output",
        platform_name: str = "CUDA",
        platform_properties: Optional[dict] = None,
    ):
        self.replica_id = replica_id
        self.n_replicas = n_replicas
        self.topology = topology
        self.temperature = temperature * unit.kelvin
        self.reference_temperature = reference_temperature * unit.kelvin
        self.lam = reference_temperature / temperature  # REST2 lambda
        self.temperatures = temperatures
        self.conformation_positions = conformation_positions
        self.n_steps_per_block = n_steps_per_block
        self.hrex_interval = hrex_interval
        self.conf_interval = conf_interval
        self.out_dir = out_dir
        self.shm_name = shm_name
        self.shm_shape = shm_shape
        self.shm_dtype = shm_dtype
        self.energy_shm_name = energy_shm_name
        self.exchange_barrier = exchange_barrier
        self.is_hottest = (replica_id == n_replicas - 1)

        system = XmlSerializer.deserialize(system_xml)
        self.system = system
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

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _get_positions_nm(self) -> np.ndarray:
        state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        return state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    def _set_positions_nm(self, pos_nm: np.ndarray) -> None:
        self.simulation.context.setPositions(
            unit.Quantity(pos_nm, unit.nanometer)
        )
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

    def get_potential_energy(self) -> float:
        state = self.simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def run_steps(self, n_steps: int) -> None:
        self.simulation.step(n_steps)

    # ------------------------------------------------------------------
    # HREX: one exchange round
    # ------------------------------------------------------------------

    def do_hrex_round(self, exchange_round: int, rng: np.random.Generator) -> bool:
        """
        Perform one HREX round using odd/even neighbor pairing.

        exchange_round: monotonically increasing counter used to alternate
                        odd and even pairings.

        Returns True if this replica swapped positions with its neighbor.
        """
        i = self.replica_id
        n = self.n_replicas

        # ----- Phase 1: write positions and energy to shared memory -----
        pos = self._get_positions_nm()
        e_i = self.get_potential_energy()
        _write_to_shm(self.shm_name, self.shm_shape, self.shm_dtype, i, pos)
        _write_energy_to_shm(self.energy_shm_name, n, i, e_i)

        # All replicas must finish writing before anyone reads
        self.exchange_barrier.wait()

        # ----- Phase 2: determine neighbors and attempt exchange -----
        # Even rounds: pair (0,1), (2,3), (4,5) ...
        # Odd  rounds: pair (1,2), (3,4), (5,6) ...
        phase = exchange_round % 2
        swapped = False

        if phase == 0:
            # even phase: lower replica in pair has even index
            if i % 2 == 0 and i + 1 < n:
                j = i + 1
                e_j = _read_energy_from_shm(self.energy_shm_name, n, j)
                accept = attempt_replica_exchange(
                    e_i, e_j, self.temperatures[i], self.temperatures[j], rng
                )
                if accept:
                    pos_j = _read_from_shm(self.shm_name, self.shm_shape, self.shm_dtype, j)
                    # Write own positions into j's slot so j can read them
                    _write_to_shm(self.shm_name, self.shm_shape, self.shm_dtype, j, pos)
                    # Set own positions to j's old positions
                    self._set_positions_nm(pos_j)
                    swapped = True
                # Signal j: accepted(1) or rejected(0) via energy shm temporarily unused slot;
                # use a dedicated flag in energy shm second half if available.
                # Simpler: write a sentinel into the energy array at slot j's "flag position".
                # We use a separate 1-byte convention via the barrier + a flag shm.
                # --- Actually write accept flag into energy array at index n+j ---
                # (We allocate energy shm as 2*n doubles: first n = energies, next n = flags)
                shm = shared_memory.SharedMemory(name=self.energy_shm_name)
                arr = np.ndarray((2 * n,), dtype=np.float64, buffer=shm.buf)
                arr[n + j] = 1.0 if accept else 0.0
                shm.close()
        else:
            # odd phase: lower replica in pair has odd index
            if i % 2 == 1 and i + 1 < n:
                j = i + 1
                e_j = _read_energy_from_shm(self.energy_shm_name, n, j)
                accept = attempt_replica_exchange(
                    e_i, e_j, self.temperatures[i], self.temperatures[j], rng
                )
                if accept:
                    pos_j = _read_from_shm(self.shm_name, self.shm_shape, self.shm_dtype, j)
                    _write_to_shm(self.shm_name, self.shm_shape, self.shm_dtype, j, pos)
                    self._set_positions_nm(pos_j)
                    swapped = True
                shm = shared_memory.SharedMemory(name=self.energy_shm_name)
                arr = np.ndarray((2 * n,), dtype=np.float64, buffer=shm.buf)
                arr[n + j] = 1.0 if accept else 0.0
                shm.close()

        # All proposers must finish writing before acceptors read the flag
        self.exchange_barrier.wait()

        # Higher-indexed replica in each pair reads the flag and applies swap
        if phase == 0:
            if i % 2 == 1 and i - 1 >= 0:  # i is the upper partner
                lower = i - 1
                shm = shared_memory.SharedMemory(name=self.energy_shm_name)
                arr = np.ndarray((2 * n,), dtype=np.float64, buffer=shm.buf)
                accept = arr[n + i] > 0.5
                shm.close()
                if accept:
                    pos_lower = _read_from_shm(self.shm_name, self.shm_shape, self.shm_dtype, i)
                    # slot i now contains the old pos of lower (written by lower)
                    # our old pos was written into slot i by lower already; read it:
                    # Actually lower wrote *our* old pos into slot j=i, so read it:
                    self._set_positions_nm(pos_lower)
                    swapped = True
        else:
            if i % 2 == 0 and i - 1 >= 0 and i > 0:
                lower = i - 1
                shm = shared_memory.SharedMemory(name=self.energy_shm_name)
                arr = np.ndarray((2 * n,), dtype=np.float64, buffer=shm.buf)
                accept = arr[n + i] > 0.5
                shm.close()
                if accept:
                    pos_lower = _read_from_shm(self.shm_name, self.shm_shape, self.shm_dtype, i)
                    self._set_positions_nm(pos_lower)
                    swapped = True

        # Final barrier: all replicas have applied swaps
        self.exchange_barrier.wait()
        return swapped


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------

def replica_main(
    replica_id: int,
    n_replicas: int,
    topology,
    system_xml: str,
    conformation_positions: List[np.ndarray],
    initial_positions: np.ndarray,
    box_vectors: np.ndarray,
    temperature: float,
    reference_temperature: float,
    temperatures: List[float],
    shm_name: str,
    shm_shape: tuple,
    shm_dtype,
    energy_shm_name: str,
    exchange_barrier: Barrier,
    n_total_steps: int,
    n_steps_per_block: int,
    hrex_interval: int,
    conf_interval: int,
    out_dir: str,
    platform_name: str,
    platform_properties: dict,
    seed: int,
    result_queue: Queue,
) -> None:
    """
    Entry point for a replica subprocess.  Runs the full production simulation
    with interleaved HREX neighbor exchanges and (for the hottest replica)
    conformation library MC.
    """
    rng = np.random.default_rng(seed)
    worker = ReplicaWorker(
        replica_id=replica_id,
        n_replicas=n_replicas,
        topology=topology,
        system_xml=system_xml,
        conformation_positions=conformation_positions,
        initial_positions=initial_positions,
        box_vectors=box_vectors,
        temperature=temperature,
        reference_temperature=reference_temperature,
        temperatures=temperatures,
        shm_name=shm_name,
        shm_shape=shm_shape,
        shm_dtype=shm_dtype,
        energy_shm_name=energy_shm_name,
        exchange_barrier=exchange_barrier,
        dt=0.002,
        n_steps_per_block=n_steps_per_block,
        hrex_interval=hrex_interval,
        conf_interval=conf_interval,
        out_dir=out_dir,
        platform_name=platform_name,
        platform_properties=platform_properties,
    )

    steps_done = 0
    exchange_round = 0
    conf_swap_accepted = 0
    conf_swap_attempted = 0
    hrex_accepted = 0
    hrex_attempted = 0

    while steps_done < n_total_steps:
        # --- Run a block of MD ---
        block = min(n_steps_per_block, n_total_steps - steps_done)
        worker.run_steps(block)
        steps_done += block

        # --- HREX neighbor exchange (all replicas participate) ---
        if steps_done % hrex_interval == 0:
            swapped = worker.do_hrex_round(exchange_round, rng)
            hrex_attempted += 1
            if swapped:
                hrex_accepted += 1
            exchange_round += 1

        # --- Conformation library MC (hottest replica only) ---
        if worker.is_hottest and steps_done % conf_interval == 0 and len(conformation_positions) > 1:
            conf_swap_attempted += 1
            accepted = attempt_conformation_swap(
                worker.simulation,
                worker.topology,
                worker.conformation_positions,
                worker.temperature,
                rng,
            )
            if accepted:
                conf_swap_accepted += 1

    result_queue.put({
        "replica_id": replica_id,
        "hrex_swap_rate": hrex_accepted / max(1, hrex_attempted),
        "conf_swap_rate": conf_swap_accepted / max(1, conf_swap_attempted),
    })
