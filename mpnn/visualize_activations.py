import torch

from string import ascii_uppercase
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image


def visualize_activations(mol: Chem.Mol, activations: torch.Tensor, layer_number: int) -> Image.Image:
    assert len(activations.shape) == 2, "Activations must be a 2D tensor."
    assert activations.shape[
               0] == mol.GetNumAtoms(), "Number of activations must match number of atoms in the molecule."

    activations_numeric = activations
    activations = activations > 0.5
    mol = Chem.Mol(mol)

    for i, atom in enumerate(mol.GetAtoms()):
        atom_activations = []
        for j in range(activations.shape[1]):
            if activations[i, j]:
                atom_activations.append(
                    f"{ascii_uppercase[j]}{layer_number}")  # ({activations_numeric[i, j]:.2f})")
        if atom_activations:
            atom.SetProp('atomNote', ', '.join(atom_activations))

    img = Draw.MolToImage(mol, size=(1000, 1000))

    return img


def visualize_all_activations(mol: Chem.Mol, activations: list[torch.Tensor]):
    images = []
    for i, layer_activations in enumerate(activations):
        img = visualize_activations(mol, layer_activations, i + 1)
        images.append(img)

    return images
