# rna-ligand-rest2

REST2/HREX molecular dynamics for multi-conformation RNA–ligand complexes
using OpenMM and OpenFF.

## Two-stage workflow

The pipeline is split into two independent, decoupled CLI tools:

| Stage | Command | Responsibility |
|-------|---------|----------------|
| 1 | `rna-rest2-prep` | Force-field parameterization, solvation, solvent equalization → writes `prep_out/` |
| 2 | `rna-rest2-run`  | Reads `prep_out/`, runs EM/NVT/NPT equilibration, then REST2/HREX production |

This separation means you can re-run Stage 2 with different temperatures,
replica counts, or exchange parameters without ever re-doing the expensive
force-field and solvation work.

## Installation

```bash
conda env create -f environment.yml
conda activate rest2
pip install -e .
```

## Stage 1 — Preparation

```bash
rna-rest2-prep \
  --rna   rna_conf0.pdb rna_conf1.pdb rna_conf2.pdb \
  --ligand lig_conf0.sdf lig_conf1.sdf lig_conf2.sdf \
  --padding 1.2 \
  --ionic_strength 0.15 \
  --outdir prep_out
```

### Output directory layout

```
prep_out/
  system.xml                  # serialized OpenMM System
  reference_topology.pdb      # solvated & trimmed reference topology
  solvation.json              # water/ion counts, padding, ionic strength
  manifest.json               # versions, CLI args, completion flag
  conformers/
    index.json                # per-conformation metadata + source SHA256
    conf_000/
      positions.npy           # float64 [n_atoms, 3] nm
      box.npy                 # float64 [3, 3] nm
    conf_001/ ...
```

## Stage 2 — Simulation

```bash
rna-rest2-run \
  --prep_dir prep_out \
  --n_replicas 4 \
  --T_low 300 --T_high 400 \
  --n_steps 5000000 \
  --hrex_interval 500 \
  --conf_interval 1000 \
  --device_index 0,1,2,3 \
  --outdir run_out
```

Stage 2 never reads the original RNA PDB or ligand SDF files — it only
consumes `prep_out/`. You can run multiple Stage 2 experiments against
the same `prep_out/`.

## REST2/HREX algorithm

* **N replicas** on a geometric temperature ladder
  `T_i = T_low × (T_high/T_low)^(i/(N-1))`.
* **HREX**: every `--hrex_interval` steps all replicas synchronize and
  attempt neighbor exchanges with alternating odd/even pairing via
  Metropolis criterion.
* **Conformation library MC**: only the hottest replica (highest temperature)
  attempts a swap with the conformation pool every `--conf_interval` steps,
  injecting structural diversity that propagates down the ladder via HREX.
* **Multiprocessing**: each replica runs in an isolated child process with
  its own CUDA context; positions and energies are exchanged via shared memory.

## Key implementation notes

- OpenMM Context isolation: the main process never creates a Context.
  All EM/NVT/NPT equilibration and production simulations run inside
  child processes, avoiding the CUDA context inheritance bug.
- Equilibration uses a `ProcessPoolExecutor` of size `--n_replicas` to
  parallelize across all conformations.
- System serialization (`system.xml`) is written once in Stage 1 and
  reused identically in Stage 2, ensuring the production run uses exactly
  the same force-field parameters that were equilibrated.
