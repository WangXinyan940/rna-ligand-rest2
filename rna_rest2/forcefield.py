"""Force field setup for RNA (OpenMM amber14) and ligand (OpenFF Sage)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from openmm import app, unit
from openmm.app import ForceField, Modeller, PDBFile
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from openff.toolkit import Molecule as OFFMolecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
except ImportError as e:
    raise ImportError(
        "openff-toolkit and openmmforcefields are required. "
        "Install with: conda install -c conda-forge openmmforcefields openff-toolkit"
    ) from e


def load_rna_pdb(pdb_path: str) -> Tuple[app.Topology, unit.Quantity]:
    """Load RNA from PDB file, returning (topology, positions)."""
    pdb = PDBFile(pdb_path)
    return pdb.topology, pdb.positions


def load_ligand_sdf(sdf_path: str) -> OFFMolecule:
    """Load small molecule from SDF, add hydrogens if missing."""
    mol = OFFMolecule.from_file(sdf_path, file_format="SDF")
    return mol


def build_forcefield(ligand_molecule: OFFMolecule, ff_xmls: list[str] | None = None) -> ForceField:
    """
    Build an OpenMM ForceField that combines:
      - amber14-all.xml + amber14/tip3pfb.xml for RNA/protein/water
      - OpenFF Sage (via SMIRNOFFTemplateGenerator) for the small molecule
    """
    if ff_xmls is None:
        ff_xmls = ["amber14-all.xml", "amber14/tip3pfb.xml"]

    smirnoff_gen = SMIRNOFFTemplateGenerator(molecules=[ligand_molecule], forcefield="openff-2.1.0")
    ff = ForceField(*ff_xmls)
    ff.registerTemplateGenerator(smirnoff_gen.generator)
    return ff


def build_complex_system(
    rna_pdb: str,
    ligand_sdf: str,
    ff_xmls: list[str] | None = None,
) -> Tuple[app.Topology, unit.Quantity, ForceField]:
    """
    Load RNA + ligand, merge topologies, build combined ForceField.

    Returns
    -------
    topology : app.Topology
        Merged RNA+ligand topology (unsolvated).
    positions : unit.Quantity
        Merged positions (nm).
    ff : ForceField
        Force field object ready for createSystem().
    """
    rna_top, rna_pos = load_rna_pdb(rna_pdb)
    lig_mol = load_ligand_sdf(ligand_sdf)

    # Write ligand to temp PDB via RDKit for topology merge
    lig_rdmol = lig_mol.to_rdkit()
    tmp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    Chem.MolToPDBFile(lig_rdmol, tmp_pdb.name)
    lig_pdb = PDBFile(tmp_pdb.name)
    os.unlink(tmp_pdb.name)

    modeller = Modeller(rna_top, rna_pos)
    modeller.add(lig_pdb.topology, lig_pdb.positions)

    ff = build_forcefield(lig_mol, ff_xmls)
    return modeller.topology, modeller.positions, ff
