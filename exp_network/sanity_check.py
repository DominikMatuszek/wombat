import torch
import rdkit
import itertools

from tqdm import tqdm
from matplotlib import pyplot as plt
from bxaic_dataset import XAIMolecularDataset, SYMBOLS
from whitebox_gnn import WhiteboxGCN

def main():
    ds = XAIMolecularDataset(
        root="data/bxaic",
        name="indole",
        explanations=True,
        cutoff=5000
    )

    #print(ds[0].expl_node_mask); exit()

    print("Number of graphs in dataset:", len(ds))

    carbon_idx = 1
    nitrogen_idx = 2

    model = WhiteboxGCN(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (4, 6),
            (6, 7),
            (7, 8),
            (8, 5)
        ],
        [
            carbon_idx,
            carbon_idx,
            carbon_idx,
            carbon_idx,
            carbon_idx,
            carbon_idx,
            carbon_idx,
            carbon_idx,
            nitrogen_idx
        ]
    ) # Say hello to benzene-finding GNN

    model.set_max_procedural_permutations(int(1e6))

    for entry in tqdm(ds):        
        x = entry.x 
        x = torch.unsqueeze(x, dim=1)
        edge_index = entry.edge_index
        important_nodes = entry.expl_node_mask
        important_node_ids = torch.nonzero(important_nodes.squeeze(), as_tuple=False).squeeze().tolist()

        important_node_count = important_nodes.sum().item()

        if important_node_count != 9 and important_node_count != 0: # Only 1 indole per molecule because I have a bad idea
            continue

        if important_nodes.sum() == 0:
            model.clear_guaranteed_permutations()
        else:
            important_permutations = list(itertools.permutations(important_node_ids))
            model.set_guaranteed_permutations(important_permutations)

        #print(x.shape)
        #print("X = ", x)
        #print(edge_index.shape)
        #print(entry.smiles)

        out = model(x, edge_index=edge_index)
        #print("Model output:", out)
        #print("Y:", entry.y)



        mol = rdkit.Chem.MolFromSmiles(entry.smiles)
        img = rdkit.Chem.Draw.MolToImage(mol, highlightAtoms=important_node_ids)

        y = entry.y.item() > 0.5
        y_hat = out.item() > 0.5 

        assert y == y_hat, f"Failure for SMILES {entry.smiles}"

        #plt.imshow(img)
        #plt.title(f"Indole check: {entry.y.item() > 0.5}, Model's indole check: {out.item() > 0.5}")
        #plt.show()

if __name__ == "__main__":
    main()