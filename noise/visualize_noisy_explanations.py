import pathlib

import pandas as pd
import numpy as np
import torch

from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib import pyplot as plt
from torch_geometric.nn import MessagePassing
from captum.attr import IntegratedGradients, Saliency
from rdkit.Chem import rdMolAlign
from rdkit.Chem import AllChem

from dataset import PubchemProcessedSMILESDataset
from mpnn import visualize_all_activations, MoleculeODetector, MoleculeKDetector, MoleculeFDetector
from highlight_smarts import highlight_atoms_in_mol, visualize_smarts_match
from krfp_models import krfp_models
from captum_attributions import explain_with_ig, explain_with_saliency, explain_with_shap_sampling, \
    explain_with_input_x_gradient
from mpnn import one_hot_encode, smiles_to_torch, mol_to_torch, MoleculeCDetector
from datavis import visualize_atom_importance, visualize_atom_importance_from_mol
from mpnn.mpnn_arch import AllNonZeroSumReadout
from noise import NoisyNetwork


def main():
    model_fn = MoleculeFDetector
    df = pd.read_csv("data/validation_datasets_small/MoleculeFDetector.csv")
    df = df[df["ORIGIN"] == "POSITIVE"]
    smiles_list = df["SMILES"].tolist()[1:2]

    mol = Chem.MolFromSmiles(smiles_list[0])
    Chem.RemoveAllHs(mol)

    _, _, pattern_smarts = krfp_models[2]

    fig, axs = plt.subplots(5, 5, figsize=(30, 30), squeeze=False)

    for noise, ax in zip([elem / 20 for elem in range(0, 101)], axs.flatten()):
        ax.axis('off')

        model = model_fn()
        model = NoisyNetwork(model, noise_std=noise)
        lit_up_atoms = set(highlight_atoms_in_mol(mol, pattern_smarts))

        prediction_mask = [1 if i in lit_up_atoms else 0 for i in range(mol.GetNumAtoms())]
        prediction_mask = np.array(prediction_mask)

        print("Prediction mask:", prediction_mask)

        example_x, edge_index, example_e = mol_to_torch(mol, add_hydrogen_ohe=True)

        model.pyg_mode = True

        method = explain_with_ig

        pred, activations = model(example_x, example_e, edge_index, dump_activations=True)

        if pred < 0.5:
            continue

        node_attrs, edge_attrs = method(
            model=model,
            example_x=example_x,
            example_e=example_e,
            edge_index=edge_index,
            # target_smiles=pattern_smiles,
            # infer_node_mask=True
        )

        activation_imgs = visualize_all_activations(mol, activations)

        node_attrs = torch.sum(node_attrs, dim=1)
        # node_attrs = torch.sum(torch.abs(node_attrs), dim=1)

        try:
            auroc = roc_auc_score(prediction_mask, node_attrs.squeeze().detach().numpy())
            ap = average_precision_score(prediction_mask, node_attrs.squeeze().detach().numpy())
            ap_baseline = len(lit_up_atoms) / mol.GetNumAtoms()
        except ValueError:
            continue

        importance_img = visualize_atom_importance_from_mol(
            mol,
            node_attrs.squeeze().detach().numpy().tolist(),
            length=1000
        )

        ax.imshow(importance_img)
        ax.set_title(f"noise = {noise}; pred = {pred.item()};\n auROC: {auroc:.4f}; AP {ap:.4f} ({ap_baseline:.4f})",
                     fontsize=12)

    fig.suptitle(f"{method.__name__} atom importance", fontsize=20)
    fig.set_dpi(300)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()


if __name__ == "__main__":
    main()
