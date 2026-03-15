from setuptools import setup, find_packages

setup(
    name="rna-ligand-rest2",
    version="0.2.0",
    description="REST2 MD simulation for RNA-ligand complexes with OpenMM and OpenFF",
    author="WangXinyan940",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "openmm>=8.0",
        "openmmforcefields>=0.12",
        "openff-toolkit>=0.14",
        "rdkit",
    ],
    entry_points={
        "console_scripts": [
            # Stage 1: parameterization + solvation + solvent equalization
            "rna-rest2-prep=rna_rest2.prep:main",
            # Stage 2: equilibration + REST2/HREX production
            "rna-rest2-run=rna_rest2.run:main",
        ]
    },
)
