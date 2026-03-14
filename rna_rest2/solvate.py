"""Solvation: add water/ions with OpenMM Modeller, then trim to equal counts across conformations."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from openmm import app, unit, Vec3
from openmm.app import ForceField, Modeller


def solvate_system(
    topology: app.Topology,
    positions: unit.Quantity,
    ff: ForceField,
    padding: unit.Quantity = 1.2 * unit.nanometer,
    ionic_strength: unit.Quantity = 0.15 * unit.molar,
    water_model: str = "tip3p",
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
) -> Tuple[app.Topology, unit.Quantity]:
    """
    Add water box and ions around the complex.
    Returns solvated (topology, positions).
    """
    modeller = Modeller(topology, positions)
    modeller.addSolvent(
        ff,
        model=water_model,
        padding=padding,
        ionicStrength=ionic_strength,
        positiveIon=positive_ion,
        negativeIon=negative_ion,
    )
    return modeller.topology, modeller.positions


def count_water_ions(topology: app.Topology) -> Tuple[int, int, int]:
    """
    Count waters, positive ions, negative ions.
    Returns (n_water, n_pos, n_neg).
    """
    n_water = n_pos = n_neg = 0
    for res in topology.residues():
        name = res.name.upper()
        if name in ("HOH", "WAT", "TIP3", "TIP3P"):
            n_water += 1
        elif name in ("NA", "K", "MG", "ZN"):
            n_pos += 1
        elif name in ("CL",):
            n_neg += 1
    return n_water, n_pos, n_neg


def trim_to_target(
    topology: app.Topology,
    positions: unit.Quantity,
    target_water: int,
    target_pos: int,
    target_neg: int,
) -> Tuple[app.Topology, unit.Quantity]:
    """
    Remove excess water/ion residues to match target counts.
    Keeps solute residues intact; removes from the end of each class.
    """
    pos_nm = positions.value_in_unit(unit.nanometer)

    water_res, pos_res, neg_res, solute_res = [], [], [], []
    for res in topology.residues():
        name = res.name.upper()
        if name in ("HOH", "WAT", "TIP3", "TIP3P"):
            water_res.append(res)
        elif name in ("NA", "K", "MG", "ZN"):
            pos_res.append(res)
        elif name in ("CL",):
            neg_res.append(res)
        else:
            solute_res.append(res)

    def _excess(lst, target):
        return set(r.index for r in lst[target:])

    remove_indices = (
        _excess(water_res, target_water)
        | _excess(pos_res, target_pos)
        | _excess(neg_res, target_neg)
    )

    if not remove_indices:
        return topology, positions

    # Build new topology and positions without removed residues
    new_top = app.Topology()
    new_top.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    new_positions = []
    atom_map = {}  # old atom index -> new atom index

    for chain in topology.chains():
        new_chain = new_top.addChain(chain.id)
        for res in chain.residues():
            if res.index in remove_indices:
                continue
            new_res = new_top.addResidue(res.name, new_chain, res.id)
            for atom in res.atoms():
                new_atom = new_top.addAtom(atom.name, atom.element, new_res)
                atom_map[atom.index] = new_atom.index
                new_positions.append(pos_nm[atom.index])

    for bond in topology.bonds():
        a1, a2 = bond
        if a1.index in atom_map and a2.index in atom_map:
            atoms = list(new_top.atoms())
            new_top.addBond(atoms[atom_map[a1.index]], atoms[atom_map[a2.index]])

    new_pos = unit.Quantity(
        np.array(new_positions, dtype=np.float64), unit.nanometer
    )
    return new_top, new_pos


def equalize_solvation(
    solvated_systems: List[Tuple[app.Topology, unit.Quantity]],
) -> List[Tuple[app.Topology, unit.Quantity]]:
    """
    Given a list of solvated (topology, positions), trim all to the minimum
    water/ion count across systems so every conformation has the same environment size.
    """
    counts = [count_water_ions(top) for top, _ in solvated_systems]
    min_water = min(c[0] for c in counts)
    min_pos = min(c[1] for c in counts)
    min_neg = min(c[2] for c in counts)

    result = []
    for top, pos in solvated_systems:
        t, p = trim_to_target(top, pos, min_water, min_pos, min_neg)
        result.append((t, p))
    return result
