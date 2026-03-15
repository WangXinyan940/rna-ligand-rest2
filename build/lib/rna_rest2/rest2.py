"""REST2 (Replica Exchange with Solute Tempering 2) parameter scaling.

Scaling convention (Wang et al. 2011):
  lambda = T0 / T   (1.0 for reference replica, <1 for hotter)
  q_i_scaled    = q_i * sqrt(lambda)      [solute-solute and solute-solvent charges]
  eps_i_scaled  = eps_i * lambda          [solute-solute LJ epsilon]
  solute-solvent LJ epsilon: eps_sw = sqrt(eps_s_scaled * eps_w_original)

Implementation uses direct parameter modification on:
  - NonbondedForce  (charges, sigma, epsilon per particle + exceptions)
  - CustomNonbondedForce if present (openff sometimes creates one)
"""
from __future__ import annotations

import copy
from typing import List, Set

import numpy as np
from openmm import NonbondedForce, CustomNonbondedForce, HarmonicBondForce
from openmm import unit


def get_solute_atom_indices(
    topology,
    include_ligand: bool = True,
) -> Set[int]:
    """
    Return indices of solute (RNA + ligand) atoms.
    Solvent = water + ions. Everything else = solute.
    """
    SOLVENT_RESIDUES = {"HOH", "WAT", "TIP3", "TIP3P", "NA", "CL", "K", "MG", "ZN"}
    solute_indices = set()
    for atom in topology.atoms():
        if atom.residue.name.upper() not in SOLVENT_RESIDUES:
            solute_indices.add(atom.index)
    return solute_indices


def _get_nbforce(system) -> NonbondedForce | None:
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, NonbondedForce):
            return f
    return None


def _get_custom_nb_forces(system) -> List[CustomNonbondedForce]:
    return [
        system.getForce(i)
        for i in range(system.getNumForces())
        if isinstance(system.getForce(i), CustomNonbondedForce)
    ]


def store_original_parameters(system, topology) -> dict:
    """
    Store original NonbondedForce parameters for all atoms.
    Call this ONCE on the unscaled system before any REST2 scaling.
    Returns a dict with 'nb_params' and 'nb_exceptions'.
    """
    nbf = _get_nbforce(system)
    if nbf is None:
        raise ValueError("No NonbondedForce found in system.")

    nb_params = []
    for i in range(nbf.getNumParticles()):
        charge, sigma, epsilon = nbf.getParticleParameters(i)
        nb_params.append((
            charge.value_in_unit(unit.elementary_charge),
            sigma.value_in_unit(unit.nanometer),
            epsilon.value_in_unit(unit.kilojoule_per_mole),
        ))

    nb_exceptions = []
    for i in range(nbf.getNumExceptions()):
        p1, p2, chprod, sigma, epsilon = nbf.getExceptionParameters(i)
        nb_exceptions.append((
            p1, p2,
            chprod.value_in_unit(unit.elementary_charge**2),
            sigma.value_in_unit(unit.nanometer),
            epsilon.value_in_unit(unit.kilojoule_per_mole),
        ))

    return {"nb_params": nb_params, "nb_exceptions": nb_exceptions}


def apply_rest2_scaling(
    system,
    context,
    topology,
    original_params: dict,
    lam: float,
) -> None:
    """
    Apply REST2 scaling with factor `lam` = T0/T.

    Modifies NonbondedForce particle parameters and exceptions in-place
    and calls context.updateParametersInContext().

    Solute-solute:   q -> q*sqrt(lam), eps -> eps*lam
    Solute-solvent:  q -> q*sqrt(lam), eps -> sqrt(eps_s*lam * eps_w)
    Solvent-solvent: unchanged
    """
    solute_idx = get_solute_atom_indices(topology)
    sqrt_lam = float(np.sqrt(lam))

    nbf = _get_nbforce(system)
    nb_params = original_params["nb_params"]
    nb_exceptions = original_params["nb_exceptions"]

    # Scale particle parameters
    for i in range(nbf.getNumParticles()):
        q0, sig0, eps0 = nb_params[i]
        if i in solute_idx:
            new_q = q0 * sqrt_lam
            new_eps = eps0 * lam
        else:
            new_q = q0
            new_eps = eps0
        nbf.setParticleParameters(
            i,
            new_q * unit.elementary_charge,
            sig0 * unit.nanometer,
            new_eps * unit.kilojoule_per_mole,
        )

    # Scale exceptions (1-4 pairs etc.)
    for idx, (p1, p2, chprod0, sig0, eps0) in enumerate(nb_exceptions):
        p1_solute = p1 in solute_idx
        p2_solute = p2 in solute_idx
        if p1_solute and p2_solute:
            new_chprod = chprod0 * lam          # q1*q2 * lam
            new_eps = eps0 * lam
        elif p1_solute or p2_solute:
            new_chprod = chprod0 * sqrt_lam     # mixed: q_s*q_w * sqrt(lam)
            new_eps = eps0 * sqrt_lam
        else:
            new_chprod = chprod0
            new_eps = eps0
        nbf.setExceptionParameters(
            idx,
            p1, p2,
            new_chprod * unit.elementary_charge**2,
            sig0 * unit.nanometer,
            new_eps * unit.kilojoule_per_mole,
        )

    nbf.updateParametersInContext(context)
