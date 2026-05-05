import rdkit

from rdkit import Chem

default_atom_map = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
}


def smiles_to_pattern_and_types(mol, atom_map: dict | None = None) -> tuple[list[tuple[int, int]], list[int]]:
    if atom_map is None:
        atom_map = default_atom_map

    pattern = []
    types = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        pattern.append((begin_idx, end_idx))

    for atom in mol.GetAtoms():
        types.append(atom_map[atom.GetSymbol()])

    return pattern, types


def get_matching_node_ids(target_mol, pattern_mol) -> tuple:
    match = target_mol.GetSubstructMatch(pattern_mol)

    return tuple(match)


def main():
    import numpy as np
    from datavis import visualize_atom_importance

    target_smiles = "C1=CC=C2C(=C1)C(=CN2)CCC(=O)O"  # https://en.wikipedia.org/wiki/3-Indolepropionic_acid
    pattern_smiles = "C1=CC=C2C(=C1)C=CN2"  # Indole

    target_mol = Chem.MolFromSmiles(target_smiles)
    pattern_mol = Chem.MolFromSmiles(pattern_smiles)

    print("Pattern of pattern", smiles_to_pattern_and_types(pattern_mol))
    print("Pattern of target", smiles_to_pattern_and_types(target_mol))

    match = get_matching_node_ids(target_mol, pattern_mol)

    print("Matching node IDs:", match)

    importance_list = [0.0] * target_mol.GetNumAtoms()

    for idx in match:
        importance_list[idx] = 1.0

    img = visualize_atom_importance(target_smiles, importance_list)
    img.show()


if __name__ == "__main__":
    main()
