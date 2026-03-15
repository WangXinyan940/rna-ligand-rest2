"""CLI entry point: orchestrates multi-conformation REST2/HREX simulation.

Simulation design
-----------------
* N replicas are created on a geometric temperature ladder [T_low, T_high].
* Each replica runs in its own subprocess with its own OpenMM Context.
* Every ``hrex_interval`` MD steps all replicas synchronize and attempt
  Hamiltonian Replica Exchange (HREX) between adjacent (neighbor) replicas
  using an alternating odd/even pairing scheme.
* Only the *hottest* replica (index N-1, highest temperature) also performs
  conformation library MC every ``conf_interval`` steps to inject structural
  diversity.
* Shared memory is used for efficient inter-process position/energy transfer.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from multiprocessing import shared_memory
from typing import List

import numpy as np
from openmm import XmlSerializer, app, unit

from .forcefield import build_complex_system
from .solvate import solvate_system, equalize_solvation
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
    Geometric spacing: T_i = T_low * (T_high / T_low)^(i / (n-1))
    Gives equal acceptance probability across adjacent replicas under
    the assumption that heat capacity is roughly constant.
    """
    if n_replicas == 1:
        return [T_low]
    ratio = T_high / T_low
    return [
        T_low * (ratio ** (i / (n_replicas - 1)))
        for i in range(n_replicas)
    ]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="REST2/HREX MD simulation for multi-conformation RNA-ligand complexes."
    )
    parser.add_argument(
        "--rna", nargs="+", required=True,
        help="RNA PDB files, one per conformation (consistent atom order).",
    )
    parser.add_argument(
        "--ligand", nargs="+", required=True,
        help="Ligand SDF files, one per conformation (consistent atom order).",
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
                        help="Steps between conformation library MC attempts (hottest replica only).")
    # Equilibration
    parser.add_argument("--em_max_iter", type=int, default=5000)
    parser.add_argument("--nvt_steps", type=int, default=50_000)
    parser.add_argument("--npt_steps", type=int, default=100_000)
    parser.add_argument("--restraint_k", type=float, default=1000.0,
                        help="Restraint force constant kJ/mol/nm^2.")
    parser.add_argument("--report_interval", type=int, default=5000)
    # Solvation
    parser.add_argument("--padding", type=float, default=1.2,
                        help="Solvation box padding (nm).")
    parser.add_argument("--ionic_strength", type=float, default=0.15,
                        help="Ionic strength (mol/L).")
    # Infrastructure
    parser.add_argument("--outdir", type=str, default="output")
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
    assert len(args.rna) == len(args.ligand), (
        "Number of RNA PDB and ligand SDF files must match."
    )
    n_conformations = len(args.rna)

    # --- Geometric temperature ladder ---
    temperatures = geometric_temperature_ladder(args.n_replicas, args.T_low, args.T_high)
    print("[INFO] Replica temperatures (K):")
    for i, T in enumerate(temperatures):
        lam = args.T_low / T
        tag = " <-- hottest (conformation library MC enabled)" if i == args.n_replicas - 1 else ""
        print(f"  replica {i:2d}: {T:7.2f} K  lambda={lam:.4f}{tag}")

    os.makedirs(args.outdir, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: Build complex topology + force field
    # ----------------------------------------------------------------
    print("\n[INFO] Building complex systems...")
    topologies, positions_list, ffs = [], [], []
    for i, (rna_pdb, lig_sdf) in enumerate(zip(args.rna, args.ligand)):
        print(f"  Conformation {i}: {rna_pdb} + {lig_sdf}")
        top, pos, ff = build_complex_system(rna_pdb, lig_sdf)
        topologies.append(top)
        positions_list.append(pos)
        ffs.append(ff)

    # ----------------------------------------------------------------
    # Step 2: Solvate
    # ----------------------------------------------------------------
    print("\n[INFO] Solvating conformations...")
    solvated = []
    for i, (top, pos, ff) in enumerate(zip(topologies, positions_list, ffs)):
        print(f"  Solvating conformation {i}...")
        s_top, s_pos = solvate_system(
            top, pos, ff,
            padding=args.padding * unit.nanometer,
            ionic_strength=args.ionic_strength * unit.molar,
        )
        solvated.append((s_top, s_pos))

    print("[INFO] Equalizing solvation...")
    solvated = equalize_solvation(solvated)
    solv_top = solvated[0][0]

    # ----------------------------------------------------------------
    # Step 3: Serialize Systems
    # ----------------------------------------------------------------
    print("\n[INFO] Creating and serializing OpenMM systems...")
    ref_ff = ffs[0]
    system_xmls = []
    conf_inputs = []
    for i, (s_top, s_pos) in enumerate(solvated):
        system = ref_ff.createSystem(
            s_top,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
            removeCMMotion=True,
        )
        system_xmls.append(XmlSerializer.serialize(system))
        pos_nm = s_pos.value_in_unit(unit.nanometer)
        conf_inputs.append((s_top, pos_nm, None))

    # ----------------------------------------------------------------
    # Step 4: Parallel equilibration
    # ----------------------------------------------------------------
    print(f"\n[INFO] Equilibrating {n_conformations} conformations "
          f"with pool size={args.n_replicas}...")
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
    # Step 5: Shared memory setup
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
    print(f"\n[INFO] Position SHM: {pos_shm.name}, shape={pos_shm_shape}")
    print(f"[INFO] Energy  SHM: {energy_shm.name}, size={2 * args.n_replicas} x float64")

    # Conformation pool: all equilibrated positions
    conformation_pool = [
        equil_results[i][1]
        for i in range(n_conformations)
    ]

    # ----------------------------------------------------------------
    # Step 6: Synchronization primitive for HREX
    # ----------------------------------------------------------------
    # 3 barrier waits per HREX round (write, propose, apply)
    exchange_barrier = mp.Barrier(args.n_replicas)

    # ----------------------------------------------------------------
    # Step 7: Launch replica subprocesses
    # ----------------------------------------------------------------
    print(f"\n[INFO] Launching {args.n_replicas} REST2/HREX replica workers...")
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
                temperatures,                  # full temperature list for HREX criterion
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

    print("\n[INFO] REST2/HREX simulation complete.")
    for r in sorted(results, key=lambda x: x["replica_id"]):
        hottest_tag = " (conformation library MC)" if r["replica_id"] == args.n_replicas - 1 else ""
        print(
            f"  Replica {r['replica_id']:2d} ({temperatures[r['replica_id']]:.1f} K): "
            f"HREX swap rate = {r['hrex_swap_rate']:.3f}"
            f"{hottest_tag}  conf swap rate = {r['conf_swap_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
