# RNA-Ligand REST2 MD Simulation

OpenMM-based REST2 (Replica Exchange with Solute Tempering 2) molecular dynamics simulation for RNA-small molecule complexes with multiple conformations.

## Features

- **Multi-conformation support**: Takes multiple RNA-ligand conformations (RNA as PDB, ligand as SDF). Atom order is consistent across conformations.
- **Force fields**: OpenMM built-in ForceField for RNA (amber14); OpenFF Sage for small molecules via `openff-toolkit`.
- **Solvation**: OpenMM Modeller adds water/ions; all replicas are trimmed to the same number of waters and ions.
- **Equilibration**: EM → NVT → NPT with heavy-atom positional restraints for each conformation.
- **REST2**: Scales solute (RNA + ligand) nonbonded parameters (charges × β_eff^0.5, ε × β_eff) via direct modification of `NonbondedForce` and `CustomNonbondedForce` parameters.
- **Parallel replicas**: Python `multiprocessing` with `shared_memory` for coordinate exchange between replicas.
- **Conformation swaps**: At fixed intervals, each replica attempts a Metropolis MC swap with one of the stored multi-conformation snapshots.

## Installation

```bash
conda create -n rest2 python=3.10
conda activate rest2
conda install -c conda-forge openmm openmmforcefields openff-toolkit rdkit parmed
pip install -e .
```

## Quick Start

```bash
python -m rna_rest2.run \
  --rna conformations/rna_conf*.pdb \
  --ligand conformations/lig_conf*.sdf \
  --n_replicas 4 \
  --temperatures 300 310 330 360 \
  --n_steps 5000000 \
  --swap_interval 500 \
  --outdir output/
```

## File Structure

```
rna_rest2/
  __init__.py
  forcefield.py      # RNA + OpenFF ligand force field setup
  solvate.py         # Solvation and ion balancing
  equilibrate.py     # EM / NVT / NPT equilibration
  rest2.py           # REST2 parameter scaling
  replica.py         # Single replica worker (multiprocessing)
  exchange.py        # Metropolis MC exchange logic
  run.py             # CLI entry point
conformations/       # Place your PDB and SDF files here
output/              # Simulation outputs
```

## REST2 Scaling

For replica at temperature `T` with reference temperature `T0`:

- Scale factor: `λ = T0 / T` (0 < λ ≤ 1 for hotter replicas)
- Solute charges: multiplied by `sqrt(λ)`
- Solute LJ epsilon: multiplied by `λ`
- Solute-solvent interactions: multiplied by `sqrt(λ)` for charges, `sqrt(λ)` for LJ epsilon

This follows the Wang et al. 2011 REST2 formulation.

## References

- Wang, L. et al. *J. Phys. Chem. B* **115**, 9431 (2011). REST2.
- Eastman, P. et al. *PLOS Comput. Biol.* OpenMM 7.
- Boothroyd, S. et al. OpenFF Sage force field.
