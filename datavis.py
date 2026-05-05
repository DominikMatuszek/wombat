import rdkit
import numpy as np
import io
from PIL import Image

from rdkit.Chem.Draw import SimilarityMaps
from matplotlib import pyplot as plt


def visualize_atom_importance_from_mol(mol: rdkit.Chem.Mol, attributions: list, length: int = 4000,
                                       ) -> Image.Image:
    d = rdkit.Chem.Draw.MolDraw2DCairo(length, length)
    options = d.drawOptions()
    current_font_scale = options.annotationFontScale
    options.annotationFontScale = 0.9

    for i, atom in enumerate(mol.GetAtoms()):
        if i < len(attributions):
            atom.SetProp('atomNote', f"{attributions[i]:.2f}")

    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        attributions,
        colorMap='bwr',
        draw2d=d,
        sigma=0.2,
        size=(length, length),
        contourLines=0,
    )
    d.FinishDrawing()

    img_buffer = io.BytesIO()
    img_buffer.write(d.GetDrawingText())
    img = plt.imread(img_buffer, format='PNG')
    img_buffer.close()

    img = Image.fromarray((img * 255).astype(np.uint8))

    return img


def visualize_atom_importance(smiles: str, attributions: list, length: int = 4000) -> Image.Image:
    mol = rdkit.Chem.MolFromSmiles(smiles)

    return visualize_atom_importance_from_mol(mol, attributions, length)
