import torch
import rdkit

from rdkit.Chem.Draw import SimilarityMaps
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

from bxaic_dataset import XAIMolecularDataset, SYMBOLS
from whitebox_gnn import get_whitebox_gcn
from datavis import visualize_atom_importance
from utils import smiles_to_pattern_and_types, get_matching_node_ids
from xai import get_attributions, captum_callback_factory, gnn_explainer_callback, ground_truth_callback, \
    graph_mask_explainer_callback, raw_gradient_callback


def main():
    ds = XAIMolecularDataset(
        root="data/bxaic",
        name="indole",
        explanations=True,
        cutoff=100
    )

    pattern_mol = rdkit.Chem.MolFromSmiles("C1=CC=C2C(=C1)C=CN2")
    pattern, types = smiles_to_pattern_and_types(pattern_mol)

    for i, entry in tqdm(enumerate(ds)):
        if i < 2:
            continue

        target_mol = rdkit.Chem.MolFromSmiles(entry.smiles)

        x = entry.x
        x = torch.unsqueeze(x, dim=1)

        edge_index = entry.edge_index

        model = get_whitebox_gcn(
                pattern,
                types,
                max_procedural_permutations=int(1e6),
                device="cpu", 
                softeq="rbf"
        )

        attributions = get_attributions(
                model,
                x,
                edge_index,
                target_mol,
                pattern_mol,
                #captum_callback_factory(attribution_method="Saliency"),
                captum_callback_factory(attribution_method="IntegratedGradients"),
                # gnn_explainer_callback
                # graph_mask_explainer_callback
                #ground_truth_callback
        )

        importance = visualize_atom_importance(
                entry.smiles,
                attributions.squeeze().detach().numpy().tolist()
        )

        pred = model(x, edge_index)

        plt.imshow(importance)
        plt.gcf().set_dpi(500)
        plt.title(f"Pred {pred}")
        plt.show()


if __name__ == "__main__":
    main()
