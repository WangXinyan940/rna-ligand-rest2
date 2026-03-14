"""CLI entry point: orchestrates multi-conformation REST2 simulation."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import time
from multiprocessing import shared_memory
from typing import List

import numpy as np
from openmm import XmlSerializer, app, unit

from .forcefield import build_complex_system
from .solvate import solvate_system, equalize_solvation
from .equilibrate import equilibrate_conformation
from .replica import replica_main


def parse_args():
    parser = argparse.ArgumentParser(
        description="REST2 MD simulation for multi-conformation RNA-ligand complexes."
    )
    parser.add_argument(
        "--rna", nargs="+", required=True,
        help="RNA PDB files (one per conformation, atom order must match)."
    )
    parser.add_argument(
        "--ligand", nargs="+", required=True,
        help="Ligand SDF files (one per conformation, atom order must match)."
    )
    parser.add_argument(
        "--n_replicas", type=int, default=4,
        help="Number of REST2 replicas."
    )
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=None,
        help="Temperatures for each replica (K). Auto-generated if not provided."
    )
    parser.add_argument(
        "--reference_temperature", type=float, default=300.0,
        help="Reference (lowest) temperature in K."
    )
    parser.add_argument(
        "--n_steps", type=int, default=5_000_000,
        help="Total production MD steps per replica."
    )
    parser.add_argument(
        "--steps_per_block", type=int, default=500,
        help="Steps per block (between exchange attempts)."
    )
    parser.add_argument(
        "--swap_interval", type=int, default=500,
        help="Steps between conformation swap attempts within each replica."
    )
    parser.add_argument(
        "--outdir", type=str, default="output",
        help="Output directory."
    )
    parser.add_argument(
        "--platform", type=str, default="CUDA",
        help="OpenMM platform (CUDA, OpenCL, CPU)."
    )
    parser.add_argument(
        "--device_index", type=str, default=None,
        help="Comma-separated GPU device indices per replica (e.g., '0,1,2,3')."
    )
    parser.add_argument(
        "--padding", type=float, default=1.2,
        help="Solvation padding in nm."
    )
    parser.add_argument(
        "--ionic_strength", type=float, default=0.15,
        help="Ionic strength in mol/L."
    )
    return parser.parse_args()


def generate_temperatures(n_replicas: int, T_low: float, T_high: float = None) -> List[float]:
    """Geometric temperature ladder."""
    if T_high is None:
        T_high = T_low * (1.1 ** (n_replicas - 1))  # 10% increments
    return [
        T_low * ((T_high / T_low) ** (i / (n_replicas - 1)))
        for i in range(n_replicas)
    ]


def main():
    args = parse_args()

    assert len(args.rna) == len(args.ligand), (
        "Number of RNA PDB files must equal number of ligand SDF files."
    )
    n_conformations = len(args.rna)

    if args.temperatures is not None:
        temperatures = args.temperatures
        assert len(temperatures) == args.n_replicas
    else:
        temperatures = generate_temperatures(args.n_replicas, args.reference_temperature)
    print(f"[INFO] Replica temperatures: {[f'{t:.1f} K' for t in temperatures]}")

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Step 1: Build topology + force field for each conformation ----
    print("[INFO] Building complex systems...")
    complex_systems = []
    for i, (rna_pdb, lig_sdf) in enumerate(zip(args.rna, args.ligand)):
        print(f"  Conformation {i}: {rna_pdb}, {lig_sdf}")
        top, pos, ff = build_complex_system(rna_pdb, lig_sdf)
        complex_systems.append((top, pos, ff))

    # Use topology from first conformation (all identical by assumption)
    ref_topology, _, ref_ff = complex_systems[0]

    # ---- Step 2: Solvate all conformations ----
    print("[INFO] Solvating conformations...")
    solvated = []
    for i, (top, pos, ff) in enumerate(complex_systems):
        print(f"  Solvating conformation {i}...")
        s_top, s_pos = solvate_system(
            top, pos, ff,
            padding=args.padding * unit.nanometer,
            ionic_strength=args.ionic_strength * unit.molar,
        )
        solvated.append((s_top, s_pos))

    print("[INFO] Equalizing solvation across conformations...")
    from .solvate import equalize_solvation
    solvated = equalize_solvation(solvated)

    # Use solvated topology from conformation 0 for system creation
    solv_top, _ = solvated[0]

    # ---- Step 3: Equilibrate each conformation ----
    print("[INFO] Equilibrating conformations...")
    equil_results = []
    for i, (s_top, s_pos) in enumerate(solvated):
        print(f"  Equilibrating conformation {i}...")
        out_prefix = os.path.join(args.outdir, f"conf{i}", "equil")
        system, final_pos, box_vecs = equilibrate_conformation(
            s_top, s_pos, ref_ff,
            temperature=args.reference_temperature * unit.kelvin,
            out_prefix=out_prefix,
            platform_name=args.platform,
        )
        equil_results.append((system, final_pos, box_vecs))

    # ---- Step 4: Prepare shared memory and conformation pool ----
    ref_system, ref_pos, ref_box = equil_results[0]
    n_atoms = ref_system.getNumParticles()
    shm_shape = (args.n_replicas, n_atoms * 3)
    shm_dtype = np.float64

    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shm_shape)) * 8)
    shm_name = shm.name
    print(f"[INFO] Shared memory block: {shm_name}, shape={shm_shape}")

    # Conformation pool: list of [n_atoms, 3] arrays in nm
    conformation_pool = [
        final_pos.value_in_unit(unit.nanometer)
        for _, final_pos, _ in equil_results
    ]

    # Serialize system XML once (same system for all replicas, scaling applied per-replica)
    system_xml = XmlSerializer.serialize(ref_system)

    # ---- Step 5: Launch replica subprocesses ----
    print("[INFO] Launching REST2 replica workers...")
    result_queue = mp.Queue()
    processes = []

    for rep_id, T in enumerate(temperatures):
        # Assign GPU device if provided
        if args.device_index is not None:
            devices = args.device_index.split(",")
            dev = devices[rep_id % len(devices)]
            platform_props = {"DeviceIndex": dev, "Precision": "mixed"}
        else:
            platform_props = {"Precision": "mixed"}

        # Initial positions from equilibrated conformation (round-robin)
        init_pos = conformation_pool[rep_id % n_conformations]
        init_box = equil_results[rep_id % n_conformations][2].value_in_unit(unit.nanometer)

        p = mp.Process(
            target=replica_main,
            args=(
                rep_id,
                solv_top,
                system_xml,
                [arr.copy() for arr in conformation_pool],  # each replica gets its own copy
                init_pos.copy(),
                init_box,
                T,
                args.reference_temperature,
                shm_name,
                shm_shape,
                shm_dtype,
                args.n_steps,
                args.steps_per_block,
                args.swap_interval,
                args.outdir,
                args.platform,
                platform_props,
                rep_id * 137 + 42,  # deterministic seed
                result_queue,
            ),
            daemon=True,
        )
        processes.append(p)

    for p in processes:
        p.start()

    # Collect results
    results = []
    for _ in processes:
        results.append(result_queue.get())

    for p in processes:
        p.join()

    shm.close()
    shm.unlink()

    print("\n[INFO] REST2 simulation complete.")
    for r in sorted(results, key=lambda x: x["replica_id"]):
        print(
            f"  Replica {r['replica_id']}: "
            f"conformation swap rate = {r['conf_swap_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
