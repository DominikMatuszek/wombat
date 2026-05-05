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

from mpnn import visualize_all_activations
from highlight_smarts import highlight_atoms_in_mol, visualize_smarts_match
from krfp_models import krfp_models
from captum_attributions import explain_with_ig, explain_with_saliency, explain_with_shap_sampling, \
    explain_with_input_x_gradient
from mpnn import one_hot_encode, smiles_to_torch, mol_to_torch, MoleculeCDetector
from datavis import visualize_atom_importance, visualize_atom_importance_from_mol
from pyg_attributions import explain_with_gnn_explainer, explain_with_pg_explainer


def main():
    # smiles_list = pd.read_csv("data/bxaic/data.csv")["smiles"].tolist()
    # smiles_list = get_smiles_with_mol_b()

    smiles_list = ["C.CC[N+](C)(CC)CCOC(=O)C(C)CSCCC[Si](OC)(OC)OC.C1=CC(=CC=C1C(=O)[O-])[N+](=O)[O-]",
                   "CC[NH+](CCOC(C(CSCCC[Si](OC)(OC)OC)C)=O)CC",
                   "CC[NH+](CCOC(C(CSCCC[Si](OC)(OC)OC)C)=O)C"
                   ]

    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]  # type: list[Chem.Mol]
    mols = [mol for mol in mols if mol is not None]

    mols = [Chem.RemoveAllHs(mol) for mol in mols]

    for mol in mols:
        AllChem.Compute2DCoords(mol)
        rdMolAlign.AlignMol(mol, mols[0])

    model, _, pattern_smarts = krfp_models[3]
    pattern_mol = Chem.MolFromSmarts(pattern_smarts)

    fig, axs = plt.subplots(3, min(5, len(mols)), figsize=(30, 20), squeeze=False)

    for ax in axs.flatten():
        ax.axis('off')

    for mol, ax_result, ax_baseline, ax_acts in zip(mols, axs[0].flatten(), axs[1].flatten(),
                                                    axs[2].flatten()):  # type: Chem.Mol
        lit_up_atoms = set(highlight_atoms_in_mol(mol, pattern_smarts))

        prediction_mask = [1 if i in lit_up_atoms else 0 for i in range(mol.GetNumAtoms())]
        prediction_mask = np.array(prediction_mask)

        example_x, edge_index, example_e = mol_to_torch(mol, add_hydrogen_ohe=True)

        model.pyg_mode = True

        method = explain_with_ig

        node_attrs, edge_attrs = method(
            model=model,
            example_x=example_x,
            example_e=example_e,
            edge_index=edge_index,
            # target_smiles=pattern_smiles,
            # infer_node_mask=True
        )

        _, activations = model(example_x, example_e, edge_index, dump_activations=True)
        activation_imgs = visualize_all_activations(mol, activations)

        node_attrs = torch.sum(node_attrs, dim=1)
        # node_attrs = torch.sum(torch.abs(node_attrs), dim=1)
        auroc = roc_auc_score(prediction_mask, node_attrs.squeeze().detach().numpy())
        ap = average_precision_score(prediction_mask, node_attrs.squeeze().detach().numpy())
        ap_baseline = len(lit_up_atoms) / mol.GetNumAtoms()

        importance_img = visualize_atom_importance_from_mol(
            mol,
            node_attrs.squeeze().detach().numpy().tolist(),
            length=1000
        )

        baseline_importance_img = visualize_smarts_match(
            mol,
            pattern_smarts,
            size=1000
        )

        ax_result.imshow(importance_img)
        ax_result.set_title(f"auROC: {auroc:.4f}; AP {ap:.4f} ({ap_baseline:.4f})", fontsize=12)
        ax_baseline.imshow(baseline_importance_img)
        ax_baseline.set_title("Ground truth", fontsize=12)
        ax_acts.imshow(activation_imgs[-1])
        ax_acts.set_title("Activations in the last MPNN layer", fontsize=12)

    fig.suptitle(f"{method.__name__} atom importance", fontsize=20)
    fig.set_dpi(300)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()


if __name__ == "__main__":
    main()
