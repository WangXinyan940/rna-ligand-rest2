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

import logging
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
    MonteCarloBarostat,
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
    arr = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)
    flat = arr[replica_j, :].copy()
    shm.close()
    n_atoms = shm_shape[1] // 3
    return flat.reshape(n_atoms, 3)


# ---------------------------------------------------------------------------
# Shared-memory helpers for box vectors
# ---------------------------------------------------------------------------

def _write_box_to_shm(
    shm_name: str,
    shm_shape: Tuple[int, int],  # (n_replicas, 9)
    replica_id: int,
    box_nm: np.ndarray,  # [3, 3] nm
) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shm_shape, dtype=np.float64, buffer=shm.buf)
    arr[replica_id, :] = box_nm.flatten()
    shm.close()


def _read_box_from_shm(
    shm_name: str,
    shm_shape: Tuple[int, int],  # (n_replicas, 9)
    replica_j: int,
) -> np.ndarray:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shm_shape, dtype=np.float64, buffer=shm.buf)
    flat = arr[replica_j, :].copy()
    shm.close()
    return flat.reshape(3, 3)


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
        conformation_positions: list,               # [(pos_nm, vel_nm_ps, box_nm), ...] tuples
        initial_positions: np.ndarray,             # [n_atoms, 3] nm
        box_vectors: np.ndarray,                   # [3, 3] nm
        temperature: float,                        # Kelvin
        reference_temperature: float,              # Kelvin (T0)
        temperatures: List[float],                 # all replica temperatures
        shm_name: str,                             # position shared memory block name
        shm_shape: Tuple[int, int],                # (n_replicas, n_atoms*3)
        shm_dtype: np.dtype,
        vel_shm_name: str,                         # velocity shared memory block name
        box_shm_name: str,                         # box vector shared memory block name
        box_shm_shape: Tuple[int, int],            # (n_replicas, 9)
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
        self.vel_shm_name = vel_shm_name
        self.box_shm_name = box_shm_name
        self.box_shm_shape = box_shm_shape
        self.energy_shm_name = energy_shm_name
        self.exchange_barrier = exchange_barrier
        self.is_hottest = (replica_id == n_replicas - 1)

        system = XmlSerializer.deserialize(system_xml)
        system.addForce(
            MonteCarloBarostat(1.0 * unit.bar, self.temperature)
        )
        self.system = system
        self.original_params = store_original_parameters(system, topology)

        # REST2: all replicas run at T_low in the integrator;
        # effective temperatures are achieved via Hamiltonian scaling only.
        integrator = LangevinMiddleIntegrator(
            self.reference_temperature, 1.0 / unit.picosecond, dt * unit.picosecond
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
        self.simulation.context.setVelocitiesToTemperature(self.reference_temperature)

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
                speed=True,
            )
        )

    # ------------------------------------------------------------------
    # State helpers (positions, velocities, box vectors)
    # ------------------------------------------------------------------

    def _get_positions_nm(self) -> np.ndarray:
        state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        return state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    def _get_velocities_nm_ps(self) -> np.ndarray:
        state = self.simulation.context.getState(getVelocities=True)
        return state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )

    def _get_box_vectors_nm(self) -> np.ndarray:
        state = self.simulation.context.getState(enforcePeriodicBox=True)
        box = state.getPeriodicBoxVectors(asNumpy=True)
        return box.value_in_unit(unit.nanometer)  # [3, 3]

    def _set_state_nm(
        self,
        pos_nm: np.ndarray,
        vel_nm_ps: np.ndarray,
        box_nm: np.ndarray,
    ) -> None:
        """Set positions, velocities, and periodic box vectors."""
        self.simulation.context.setPeriodicBoxVectors(
            *[unit.Quantity(v, unit.nanometer) for v in box_nm]
        )
        self.simulation.context.setPositions(
            unit.Quantity(pos_nm, unit.nanometer)
        )
        self.simulation.context.setVelocities(
            unit.Quantity(vel_nm_ps, unit.nanometer / unit.picosecond)
        )

    def _set_positions_nm(self, pos_nm: np.ndarray) -> None:
        self.simulation.context.setPositions(
            unit.Quantity(pos_nm, unit.nanometer)
        )
        self.simulation.context.setVelocitiesToTemperature(self.reference_temperature)

    def get_potential_energy(self) -> float:
        state = self.simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def compute_energy_for_positions(self, pos_nm: np.ndarray) -> float:
        """Compute potential energy for foreign coordinates under this replica's
        scaled Hamiltonian, then restore own coordinates."""
        own_state = self.simulation.context.getState(getPositions=True)
        self.simulation.context.setPositions(
            unit.Quantity(pos_nm, unit.nanometer)
        )
        energy = self.get_potential_energy()
        self.simulation.context.setPositions(own_state.getPositions())
        return energy

    def run_steps(self, n_steps: int) -> None:
        self.simulation.step(n_steps)

    # ------------------------------------------------------------------
    # HREX: one exchange round
    # ------------------------------------------------------------------

    def do_hrex_round(
        self, exchange_round: int, rng: np.random.Generator
    ) -> Tuple[bool, int]:
        """
        Perform one REST2 HREX round using odd/even neighbor pairing.

        REST2 exchange criterion requires cross-energies: each replica
        evaluates its partner's coordinates under its own scaled Hamiltonian.

        When an exchange is accepted, positions, velocities, AND periodic
        box vectors are swapped between the two replicas.

        exchange_round: monotonically increasing counter used to alternate
                        odd and even pairings.

        Returns (swapped, partner) where partner is -1 if this replica
        had no partner this round.
        """
        i = self.replica_id
        n = self.n_replicas
        T_ref = self.reference_temperature.value_in_unit(unit.kelvin)

        # ----- Phase 1: write positions, velocities, box, and self-energy -----
        pos = self._get_positions_nm()
        vel = self._get_velocities_nm_ps()
        box = self._get_box_vectors_nm()
        e_self = self.get_potential_energy()

        _write_to_shm(self.shm_name, self.shm_shape, self.shm_dtype, i, pos)
        _write_to_shm(self.vel_shm_name, self.shm_shape, self.shm_dtype, i, vel)
        _write_box_to_shm(self.box_shm_name, self.box_shm_shape, i, box)
        _write_energy_to_shm(self.energy_shm_name, n, i, e_self)

        # All replicas must finish writing before anyone reads
        self.exchange_barrier.wait()

        # ----- Phase 2: compute cross-energy E_self(X_partner) -----
        # Even rounds: pair (0,1), (2,3), (4,5) ...
        # Odd  rounds: pair (1,2), (3,4), (5,6) ...
        phase = exchange_round % 2
        swapped = False

        # Determine partner for this round
        partner = -1
        if phase == 0:
            if i % 2 == 0 and i + 1 < n:
                partner = i + 1
            elif i % 2 == 1 and i - 1 >= 0:
                partner = i - 1
        else:
            if i % 2 == 1 and i + 1 < n:
                partner = i + 1
            elif i % 2 == 0 and i - 1 >= 0 and i > 0:
                partner = i - 1

        # Compute cross-energy: E_i(X_partner) under this replica's Hamiltonian
        if partner >= 0:
            pos_partner = _read_from_shm(
                self.shm_name, self.shm_shape, self.shm_dtype, partner
            )
            e_cross = self.compute_energy_for_positions(pos_partner)
        else:
            e_cross = 0.0

        # Write cross-energy to SHM slot n + i
        shm = shared_memory.SharedMemory(name=self.energy_shm_name)
        arr = np.ndarray((2 * n,), dtype=np.float64, buffer=shm.buf)
        arr[n + i] = e_cross
        shm.close()

        # All replicas must finish computing cross-energies
        self.exchange_barrier.wait()

        # ----- Phase 3: proposer decides accept/reject -----
        # The even-indexed replica in each pair proposes the exchange.
        if i % 2 == 0 and partner >= 0 and partner < n:
            j = partner
            e_ii = e_self  # H_i(X_i)
            e_jj = _read_energy_from_shm(self.energy_shm_name, n, j)  # H_j(X_j)
            # Cross-energies from SHM
            e_ij = e_cross  # H_i(X_j) — already computed above
            shm = shared_memory.SharedMemory(name=self.energy_shm_name)
            arr = np.ndarray((2 * n,), dtype=np.float64, buffer=shm.buf)
            e_ji = arr[n + j]  # H_j(X_i) — computed by replica j
            shm.close()

            accept = attempt_replica_exchange(
                e_ii, e_jj, e_ij, e_ji, T_ref, rng
            )
            if accept:
                swapped = True
                # Swap positions in shared memory
                pos_i = _read_from_shm(self.shm_name, self.shm_shape, self.shm_dtype, i)
                pos_j = _read_from_shm(self.shm_name, self.shm_shape, self.shm_dtype, j)
                _write_to_shm(self.shm_name, self.shm_shape, self.shm_dtype, i, pos_j)
                _write_to_shm(self.shm_name, self.shm_shape, self.shm_dtype, j, pos_i)

                # Swap velocities in shared memory
                vel_i = _read_from_shm(self.vel_shm_name, self.shm_shape, self.shm_dtype, i)
                vel_j = _read_from_shm(self.vel_shm_name, self.shm_shape, self.shm_dtype, j)
                _write_to_shm(self.vel_shm_name, self.shm_shape, self.shm_dtype, i, vel_j)
                _write_to_shm(self.vel_shm_name, self.shm_shape, self.shm_dtype, j, vel_i)

                # Swap box vectors in shared memory
                box_i = _read_box_from_shm(self.box_shm_name, self.box_shm_shape, i)
                box_j = _read_box_from_shm(self.box_shm_name, self.box_shm_shape, j)
                _write_box_to_shm(self.box_shm_name, self.box_shm_shape, i, box_j)
                _write_box_to_shm(self.box_shm_name, self.box_shm_shape, j, box_i)

        # All proposers must finish before acceptors read
        self.exchange_barrier.wait()

        # Load new positions, velocities, and box vectors from SHM
        new_pos = _read_from_shm(self.shm_name, self.shm_shape, self.shm_dtype, i)
        new_vel = _read_from_shm(self.vel_shm_name, self.shm_shape, self.shm_dtype, i)
        new_box = _read_box_from_shm(self.box_shm_name, self.box_shm_shape, i)
        self._set_state_nm(new_pos, new_vel, new_box)

        # Final barrier: all replicas have applied swaps
        self.exchange_barrier.wait()
        return swapped, partner


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------

def replica_main(
    replica_id: int,
    n_replicas: int,
    topology,
    system_xml: str,
    conformation_positions: list,             # [(pos_nm, vel_nm_ps, box_nm), ...]
    initial_positions: np.ndarray,
    box_vectors: np.ndarray,
    temperature: float,
    reference_temperature: float,
    temperatures: List[float],
    shm_name: str,
    shm_shape: tuple,
    shm_dtype,
    vel_shm_name: str,
    box_shm_name: str,
    box_shm_shape: tuple,
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
        vel_shm_name=vel_shm_name,
        box_shm_name=box_shm_name,
        box_shm_shape=box_shm_shape,
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

    # --- Setup per-replica exchange logger ---
    xlog = logging.getLogger(f"exchange.replica_{replica_id}")
    xlog.setLevel(logging.INFO)
    xlog.propagate = False
    log_path = os.path.join(out_dir, "exchange_stats.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    xlog.addHandler(fh)

    steps_done = 0
    exchange_round = 0
    conf_swap_accepted = 0
    conf_swap_attempted = 0
    hrex_accepted = 0
    hrex_attempted = 0
    # Per-pair tracking: key = (min(i, partner), max(i, partner))
    pair_attempted: dict = {}
    pair_accepted: dict = {}

    log_every = 10  # log acceptance rates every N exchange rounds

    while steps_done < n_total_steps:
        # --- Run a block of MD ---
        block = min(n_steps_per_block, n_total_steps - steps_done)
        worker.run_steps(block)
        steps_done += block

        # --- HREX neighbor exchange (all replicas participate) ---
        if steps_done % hrex_interval == 0:
            swapped, partner = worker.do_hrex_round(exchange_round, rng)
            hrex_attempted += 1
            if swapped:
                hrex_accepted += 1

            # Track per-pair stats (only for replicas that had a partner)
            if partner >= 0:
                pair_key = (min(replica_id, partner), max(replica_id, partner))
                pair_attempted[pair_key] = pair_attempted.get(pair_key, 0) + 1
                if swapped:
                    pair_accepted[pair_key] = pair_accepted.get(pair_key, 0) + 1

                xlog.info(
                    "HREX  round=%d  pair=(%d,%d)  %s  running_rate=%.3f",
                    exchange_round, pair_key[0], pair_key[1],
                    "ACCEPT" if swapped else "REJECT",
                    pair_accepted.get(pair_key, 0) / pair_attempted[pair_key],
                )

            # Periodic summary every log_every rounds
            if exchange_round > 0 and exchange_round % log_every == 0:
                xlog.info(
                    "HREX  replica=%d  round=%d  overall_rate=%.3f",
                    replica_id, exchange_round,
                    hrex_accepted / max(1, hrex_attempted),
                )

            exchange_round += 1

        # --- Conformation library MC (hottest replica only) ---
        if worker.is_hottest and steps_done % conf_interval == 0 and len(conformation_positions) > 1:
            conf_swap_attempted += 1
            accepted = attempt_conformation_swap(
                worker.simulation,
                worker.topology,
                worker.conformation_positions,
                worker.reference_temperature,
                rng,
            )
            if accepted:
                conf_swap_accepted += 1

            xlog.info(
                "CONF  replica=%d  step=%d  %s  running_rate=%.3f",
                replica_id, steps_done,
                "ACCEPT" if accepted else "REJECT",
                conf_swap_accepted / max(1, conf_swap_attempted),
            )

    # --- Final summary ---
    xlog.info(
        "FINAL  replica=%d  hrex_rate=%.3f (%d/%d)  conf_rate=%.3f (%d/%d)",
        replica_id,
        hrex_accepted / max(1, hrex_attempted), hrex_accepted, hrex_attempted,
        conf_swap_accepted / max(1, conf_swap_attempted), conf_swap_accepted, conf_swap_attempted,
    )
    for pair_key in sorted(pair_attempted):
        acc = pair_accepted.get(pair_key, 0)
        att = pair_attempted[pair_key]
        xlog.info(
            "FINAL  pair=(%d,%d)  rate=%.3f (%d/%d)",
            pair_key[0], pair_key[1], acc / max(1, att), acc, att,
        )

    # Flush and close handler
    fh.flush()
    fh.close()
    xlog.removeHandler(fh)

    result_queue.put({
        "replica_id": replica_id,
        "hrex_swap_rate": hrex_accepted / max(1, hrex_attempted),
        "conf_swap_rate": conf_swap_accepted / max(1, conf_swap_attempted),
        "pair_rates": {
            f"{k[0]}-{k[1]}": pair_accepted.get(k, 0) / max(1, v)
            for k, v in pair_attempted.items()
        },
    })
