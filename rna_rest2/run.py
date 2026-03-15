"""Stage 2: equilibration + REST2/HREX production simulation.

Reads the prep directory written by ``rna-rest2-prep`` (Stage 1) and
never re-invokes ForceField, Modeller, or solvation logic.

Simulation design
-----------------
* N replicas on a geometric temperature ladder [T_low, T_high].
* Each replica runs in its own subprocess with its own OpenMM Context.
* Every ``hrex_interval`` MD steps all replicas synchronize and attempt
  Hamiltonian Replica Exchange (HREX) with alternating odd/even pairing.
* Only the *hottest* replica (index N-1) also performs conformation
  library MC every ``conf_interval`` steps.
* Shared memory is used for inter-process position/energy transfer.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from multiprocessing import shared_memory
from typing import List

import numpy as np
from openmm import XmlSerializer, app, unit
from openmm.app import PDBFile

from .equilibrate import equilibrate_all_conformations
from .replica import replica_main


# ---------------------------------------------------------------------------
# Temperature ladder
# ---------------------------------------------------------------------------

def geometric_temperature_ladder(
    n_replicas: int,
    T_low: float,
    T_high: float,
) -> List[float]:
    """
    T_i = T_low * (T_high / T_low)^(i / (n-1))
    Geometric spacing gives equal acceptance probability under constant
    heat-capacity assumption.
    """
    if n_replicas == 1:
        return [T_low]
    ratio = T_high / T_low
    return [
        T_low * (ratio ** (i / (n_replicas - 1)))
        for i in range(n_replicas)
    ]


# ---------------------------------------------------------------------------
# Prep directory loader & validator
# ---------------------------------------------------------------------------

def load_prep_dir(prep_dir: str):
    """
    Load and validate the prep directory produced by Stage 1.

    Returns
    -------
    system_xml : str
    topology   : app.Topology
    conformers : list of {"positions_nm": np.ndarray, "box_nm": np.ndarray}
    manifest   : dict
    """
    def _require(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required prep artifact not found: {path}\n"
                "Run rna-rest2-prep first."
            )
        return path

    manifest_path = _require(os.path.join(prep_dir, "manifest.json"))
    manifest = json.loads(open(manifest_path).read())
    if not manifest.get("completed"):
        raise RuntimeError(
            f"{prep_dir}/manifest.json exists but 'completed' is False. "
            "Stage 1 may have been interrupted; re-run rna-rest2-prep."
        )

    system_xml_path = _require(os.path.join(prep_dir, "system.xml"))
    system_xml = open(system_xml_path).read()

    ref_pdb_path = _require(os.path.join(prep_dir, "reference_topology.pdb"))
    with open(ref_pdb_path) as f:
        pdb = PDBFile(f)
    topology = pdb.topology

    conf_index_path = _require(os.path.join(prep_dir, "conformers", "index.json"))
    conf_index = json.loads(open(conf_index_path).read())

    conformers = []
    for entry in conf_index:
        i = entry["id"]
        cdir = os.path.join(prep_dir, "conformers", f"conf_{i:03d}")
        pos_nm = np.load(_require(os.path.join(cdir, "positions.npy")))
        box_nm = np.load(_require(os.path.join(cdir, "box.npy")))
        conformers.append({"positions_nm": pos_nm, "box_nm": box_nm})

    # Validate atom count consistency
    system = XmlSerializer.deserialize(system_xml)
    n_atoms_sys = system.getNumParticles()
    for i, c in enumerate(conformers):
        if c["positions_nm"].shape[0] != n_atoms_sys:
            raise ValueError(
                f"Conformer {i} has {c['positions_nm'].shape[0]} atoms but "
                f"system.xml has {n_atoms_sys}. Prep directory may be corrupt."
            )

    print(f"[RUN] Loaded prep directory: {os.path.abspath(prep_dir)}")
    print(f"  Completed at  : {manifest.get('completed_at', 'unknown')}")
    print(f"  Conformations : {len(conformers)}")
    print(f"  Atoms/system  : {n_atoms_sys}")
    print(f"  OpenMM ver    : {manifest.get('openmm_version', 'unknown')}")

    return system_xml, topology, conformers, manifest


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: equilibrate and run REST2/HREX simulation. "
            "Reads the prep directory written by rna-rest2-prep."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prep_dir", type=str, required=True,
        help="Path to the prep directory produced by rna-rest2-prep (Stage 1).",
    )
    # Temperature
    parser.add_argument("--T_low", type=float, default=300.0,
                        help="Lowest replica temperature (K).")
    parser.add_argument("--T_high", type=float, default=400.0,
                        help="Highest replica temperature (K).")
    parser.add_argument("--n_replicas", type=int, default=4,
                        help="Number of REST2 replicas.")
    # Production
    parser.add_argument("--n_steps", type=int, default=5_000_000,
                        help="Total production MD steps per replica.")
    parser.add_argument("--steps_per_block", type=int, default=500,
                        help="Steps per block between exchange checks.")
    parser.add_argument("--hrex_interval", type=int, default=500,
                        help="Steps between HREX neighbor exchange rounds.")
    parser.add_argument("--conf_interval", type=int, default=1000,
                        help="Steps between conformation library MC (hottest replica only).")
    # Equilibration
    parser.add_argument("--em_max_iter", type=int, default=5000)
    parser.add_argument("--nvt_steps", type=int, default=50_000)
    parser.add_argument("--npt_steps", type=int, default=100_000)
    parser.add_argument("--restraint_k", type=float, default=200.0,
                        help="Restraint force constant kJ/mol/nm^2.")
    parser.add_argument("--report_interval", type=int, default=5000)
    # Infrastructure
    parser.add_argument("--outdir", type=str, default="run_out",
                        help="Output directory for trajectories and run logs.")
    parser.add_argument("--platform", type=str, default="CUDA")
    parser.add_argument(
        "--device_index", type=str, default=None,
        help="Comma-separated GPU device indices for replicas, e.g. '0,1,2,3'.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Load prep artifacts ---
    system_xml, solv_top, conformers, manifest = load_prep_dir(args.prep_dir)
    n_conformations = len(conformers)

    # --- Geometric temperature ladder ---
    temperatures = geometric_temperature_ladder(args.n_replicas, args.T_low, args.T_high)
    print("\n[RUN] Replica temperatures (K):")
    for i, T in enumerate(temperatures):
        lam = args.T_low / T
        tag = " <-- hottest" if i == args.n_replicas - 1 else ""
        print(f"  replica {i:2d}: {T:7.2f} K  lambda={lam:.4f}{tag}")

    os.makedirs(args.outdir, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: Parallel equilibration (EM -> NVT -> NPT)
    #   - All Context creation happens inside child processes
    #   - Main process only passes serialized System + numpy arrays
    # ----------------------------------------------------------------
    print(f"\n[RUN] Equilibrating {n_conformations} conformations "
          f"with pool size={args.n_replicas}...")

    conf_inputs = [
        (solv_top, c["positions_nm"], c["box_nm"])
        for c in conformers
    ]
    system_xmls = [system_xml] * n_conformations

    equil_results = equilibrate_all_conformations(
        conformations=conf_inputs,
        system_xmls=system_xmls,
        temperature_K=args.T_low,
        em_max_iter=args.em_max_iter,
        nvt_steps=args.nvt_steps,
        npt_steps=args.npt_steps,
        restraint_k=args.restraint_k,
        report_interval=args.report_interval,
        out_dir=args.outdir,
        platform_name=args.platform,
        platform_properties={"Precision": "mixed"},
        n_workers=args.n_replicas,
    )

    # ----------------------------------------------------------------
    # Step 2: Shared memory setup
    #   - Position SHM: shape (n_replicas, n_atoms * 3), float64
    #   - Energy SHM:   shape (2 * n_replicas,), float64
    #                   first half  = energies
    #                   second half = exchange accept flags
    # ----------------------------------------------------------------
    ref_system_xml, ref_pos_nm, _ = equil_results[0]
    ref_system = XmlSerializer.deserialize(ref_system_xml)
    n_atoms = ref_system.getNumParticles()

    pos_shm_shape = (args.n_replicas, n_atoms * 3)
    pos_shm = shared_memory.SharedMemory(
        create=True, size=int(np.prod(pos_shm_shape)) * 8
    )
    energy_shm = shared_memory.SharedMemory(
        create=True, size=int(2 * args.n_replicas) * 8
    )
    print(f"\n[RUN] Position SHM: {pos_shm.name}, shape={pos_shm_shape}")
    print(f"[RUN] Energy  SHM: {energy_shm.name}, size={2 * args.n_replicas} x float64")

    # Conformation pool: all equilibrated positions
    conformation_pool = [
        equil_results[i][1]
        for i in range(n_conformations)
    ]

    # ----------------------------------------------------------------
    # Step 3: Synchronization barrier for HREX
    # ----------------------------------------------------------------
    exchange_barrier = mp.Barrier(args.n_replicas)

    # ----------------------------------------------------------------
    # Step 4: Launch replica subprocesses
    # ----------------------------------------------------------------
    print(f"\n[RUN] Launching {args.n_replicas} REST2/HREX replica workers...")
    result_queue = mp.Queue()
    processes = []

    for rep_id, T in enumerate(temperatures):
        if args.device_index is not None:
            devices = args.device_index.split(",")
            dev = devices[rep_id % len(devices)]
            platform_props = {"DeviceIndex": dev, "Precision": "mixed"}
        else:
            platform_props = {"Precision": "mixed"}

        init_conf_id = rep_id % n_conformations
        _, init_pos_nm, init_box_nm = equil_results[init_conf_id]

        p = mp.Process(
            target=replica_main,
            args=(
                rep_id,
                args.n_replicas,
                solv_top,
                ref_system_xml,
                [arr.copy() for arr in conformation_pool],
                init_pos_nm.copy(),
                init_box_nm,
                T,
                args.T_low,
                temperatures,
                pos_shm.name,
                pos_shm_shape,
                np.float64,
                energy_shm.name,
                exchange_barrier,
                args.n_steps,
                args.steps_per_block,
                args.hrex_interval,
                args.conf_interval,
                args.outdir,
                args.platform,
                platform_props,
                rep_id * 137 + 42,
                result_queue,
            ),
            daemon=True,
        )
        processes.append(p)

    for p in processes:
        p.start()

    results = [result_queue.get() for _ in processes]
    for p in processes:
        p.join()

    pos_shm.close()
    pos_shm.unlink()
    energy_shm.close()
    energy_shm.unlink()

    print("\n[RUN] REST2/HREX simulation complete.")
    for r in sorted(results, key=lambda x: x["replica_id"]):
        hottest_tag = " (conf library MC)" if r["replica_id"] == args.n_replicas - 1 else ""
        print(
            f"  Replica {r['replica_id']:2d} ({temperatures[r['replica_id']]:.1f} K): "
            f"HREX swap rate = {r['hrex_swap_rate']:.3f}"
            f"{hottest_tag}  conf swap rate = {r['conf_swap_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
