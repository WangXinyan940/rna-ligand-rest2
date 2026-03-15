#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from rdkit import Chem


def add_hs_to_sdf(input_sdf, output_sdf, add_coords=True):
    supplier = Chem.SDMolSupplier(input_sdf, sanitize=True, removeHs=False)
    writer = Chem.SDWriter(output_sdf)

    count_in = 0
    count_out = 0
    count_fail = 0

    for mol in supplier:
        count_in += 1

        if mol is None:
            count_fail += 1
            continue

        try:
            mol_h = Chem.AddHs(mol, addCoords=add_coords)
            writer.write(mol_h)
            count_out += 1
        except Exception as e:
            count_fail += 1
            print(f"[WARN] molecule #{count_in} failed: {e}")

    writer.close()
    print(f"Input molecules : {count_in}")
    print(f"Written molecules: {count_out}")
    print(f"Failed molecules : {count_fail}")
    print(f"Output file      : {output_sdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Add explicit hydrogens to molecules in an SDF using RDKit."
    )
    parser.add_argument("-i", "--input", required=True, help="input SDF file")
    parser.add_argument("-o", "--output", required=True, help="output SDF file")
    parser.add_argument(
        "--no-coords",
        action="store_true",
        help="do not generate coordinates for added H atoms"
    )
    args = parser.parse_args()

    add_hs_to_sdf(
        input_sdf=args.input,
        output_sdf=args.output,
        add_coords=not args.no_coords
    )


if __name__ == "__main__":
    main()
