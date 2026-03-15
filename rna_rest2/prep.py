"""Stage 1: force-field parameterization, solvation, and solvent equalization.

This tool performs all chemistry-setup work and writes a self-contained
``prep_out/`` directory that Stage 2 (``rna-rest2-run``) reads without
ever re-invoking ForceField, Modeller, or solvation logic.

Output directory layout
-----------------------
  prep_out/
    system.xml                  -- serialized OpenMM System (no restraints, no barostat)
    reference_topology.pdb      -- solvated & trimmed topology in PDB format
    solvation.json              -- water/ion counts, padding, ionic_strength
    conformers/
      index.json                -- list of {id, rna_src, ligand_src, n_atoms}
      conf_000/
        positions.npy           -- float64 [n_atoms, 3] nm
        box.npy                 -- float64 [3, 3] nm
      conf_001/ ...
    manifest.json               -- versions, CLI args, completion flag
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import List

import numpy as np
from openmm import XmlSerializer, app, unit
from openmm.app import PDBFile

from .forcefield import build_complex_system
from .solvate import solvate_system, equalize_solvation, count_water_ions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _openmm_version() -> str:
    try:
        import openmm
        return openmm.__version__
    except Exception:
        return "unknown"


def _openff_version() -> str:
    try:
        import openff.toolkit
        return openff.toolkit.__version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1: parameterize, solvate, and equalize conformations. "
            "Writes a prep directory consumed by rna-rest2-run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rna", nargs="+", required=True,
        help="RNA PDB files, one per conformation (consistent atom order).",
    )
    parser.add_argument(
        "--ligand", nargs="+", required=True,
        help="Ligand SDF files, one per conformation (consistent atom order).",
    )
    parser.add_argument("--padding", type=float, default=1.2,
                        help="Solvation box padding (nm).")
    parser.add_argument("--ionic_strength", type=float, default=0.15,
                        help="Ionic strength (mol/L).")
    parser.add_argument("--outdir", type=str, default="prep_out",
                        help="Output directory for prep artifacts.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing prep directory.")
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
    outdir = args.outdir

    manifest_path = os.path.join(outdir, "manifest.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        existing = json.loads(open(manifest_path).read())
        if existing.get("completed"):
            print(f"[INFO] prep_out already completed at {existing.get('completed_at')}. "
                  "Use --overwrite to redo.")
            return

    os.makedirs(outdir, exist_ok=True)
    conf_dir = os.path.join(outdir, "conformers")
    os.makedirs(conf_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: Force-field parameterization
    # ----------------------------------------------------------------
    print("\n[PREP] Step 1/4: Building complex topologies and force fields...")
    topologies, positions_list, ffs = [], [], []
    for i, (rna_pdb, lig_sdf) in enumerate(zip(args.rna, args.ligand)):
        print(f"  Conformation {i}: {rna_pdb} + {lig_sdf}")
        top, pos, ff = build_complex_system(rna_pdb, lig_sdf)
        topologies.append(top)
        positions_list.append(pos)
        ffs.append(ff)

    # ----------------------------------------------------------------
    # Step 2: Solvation
    # ----------------------------------------------------------------
    print("\n[PREP] Step 2/4: Solvating conformations...")
    solvated = []
    for i, (top, pos, ff) in enumerate(zip(topologies, positions_list, ffs)):
        print(f"  Solvating conformation {i}...")
        s_top, s_pos = solvate_system(
            top, pos, ff,
            padding=args.padding * unit.nanometer,
            ionic_strength=args.ionic_strength * unit.molar,
        )
        solvated.append((s_top, s_pos))

    # ----------------------------------------------------------------
    # Step 3: Equalize solvent
    # ----------------------------------------------------------------
    print("\n[PREP] Step 3/4: Equalizing solvation across conformations...")
    solvated = equalize_solvation(solvated)

    # Record solvation stats
    w, p, n = count_water_ions(solvated[0][0])
    solvation_info = {
        "n_water": w, "n_positive_ions": p, "n_negative_ions": n,
        "padding_nm": args.padding,
        "ionic_strength_mol_per_L": args.ionic_strength,
    }
    with open(os.path.join(outdir, "solvation.json"), "w") as f:
        json.dump(solvation_info, f, indent=2)
    print(f"  Waters: {w}, Pos ions: {p}, Neg ions: {n}")

    # ----------------------------------------------------------------
    # Step 4: Serialize system + save topology + conformer positions
    # ----------------------------------------------------------------
    print("\n[PREP] Step 4/4: Serializing system and saving conformer data...")
    ref_ff = ffs[0]
    solv_top, _ = solvated[0]

    # Build system once from the reference (equalized) topology
    system = ref_ff.createSystem(
        solv_top,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
    )
    system_xml = XmlSerializer.serialize(system)
    with open(os.path.join(outdir, "system.xml"), "w") as f:
        f.write(system_xml)
    print(f"  system.xml written ({len(system_xml)//1024} KB)")

    # Write reference topology as PDB
    ref_pdb_path = os.path.join(outdir, "reference_topology.pdb")
    _, ref_pos = solvated[0]
    with open(ref_pdb_path, "w") as f:
        PDBFile.writeFile(solv_top, ref_pos, f)
    print(f"  reference_topology.pdb written")

    # Save per-conformation positions and box vectors
    conformer_index = []
    n_atoms = system.getNumParticles()
    for i, (s_top, s_pos) in enumerate(solvated):
        cdir = os.path.join(conf_dir, f"conf_{i:03d}")
        os.makedirs(cdir, exist_ok=True)

        pos_nm = s_pos.value_in_unit(unit.nanometer)
        pos_npy = np.array([(p.x, p.y, p.z) for p in pos_nm])
        np.save(os.path.join(cdir, "positions.npy"), pos_npy.astype(np.float64))

        box = s_top.getPeriodicBoxVectors()
        if box is not None:
            box_nm = np.array([[v.x, v.y, v.z]
                                for v in [b.value_in_unit(unit.nanometer) for b in box]],
                               dtype=np.float64)
        else:
            box_nm = np.zeros((3, 3), dtype=np.float64)
        np.save(os.path.join(cdir, "box.npy"), box_nm)

        conformer_index.append({
            "id": i,
            "rna_src": os.path.abspath(args.rna[i]),
            "ligand_src": os.path.abspath(args.ligand[i]),
            "n_atoms": int(pos_npy.shape[0]),
            "rna_sha256": _file_sha256(args.rna[i]),
            "ligand_sha256": _file_sha256(args.ligand[i]),
        })
        print(f"  Conformation {i}: {pos_npy.shape[0]} atoms saved")

    with open(os.path.join(conf_dir, "index.json"), "w") as f:
        json.dump(conformer_index, f, indent=2)

    # Write manifest
    manifest = {
        "completed": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_conformations": n_conformations,
        "n_atoms": n_atoms,
        "openmm_version": _openmm_version(),
        "openff_version": _openff_version(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cli_args": vars(args),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[PREP] Done. Prep directory: {os.path.abspath(outdir)}")
    print(f"  Conformations : {n_conformations}")
    print(f"  Atoms/system  : {n_atoms}")
    print(f"  Next step     : rna-rest2-run --prep_dir {outdir} --n_replicas 4 ...")


if __name__ == "__main__":
    main()
