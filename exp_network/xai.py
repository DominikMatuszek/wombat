import rdkit
import torch

from rdkit.Chem import Mol
from torch_geometric.explain import Explainer, CaptumExplainer, GNNExplainer, GraphMaskExplainer

from utils import get_matching_node_ids

from whitebox_gnn import WhiteboxGCN


def get_attributions(
        model: WhiteboxGCN,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_mol: rdkit.Chem.Mol,
        pattern_mol: rdkit.Chem.Mol,
        callback: callable
):
    assert len(x.shape) == 2

    important_permutation = get_matching_node_ids(target_mol, pattern_mol)
    if len(important_permutation) == 0:
        model.clear_guaranteed_permutations()
    else:
        model.set_guaranteed_permutations([important_permutation])

    attributions = callback(
        model,
        x,
        edge_index,
        target_mol,
        pattern_mol
    )

    return attributions


def captum_callback_factory(attribution_method: str):
    def callback(
            model: WhiteboxGCN,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            target_mol: rdkit.Chem.Mol,
            pattern_mol: rdkit.Chem.Mol
    ):
        x = x.float()
        x = x.requires_grad_(True)

        explainer = Explainer(
            model=model,
            algorithm=CaptumExplainer(attribution_method=attribution_method),  # show_progress=True, n_samples=5),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='node',
                return_type='raw',
            ),
        )
        explanation = explainer(x, edge_index)

        return explanation.node_mask

    return callback

def raw_gradient_callback(
            model: WhiteboxGCN,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            target_mol: rdkit.Chem.Mol,
            pattern_mol: rdkit.Chem.Mol
    ):
        x = x.float()
        x = x.requires_grad_(True)

        result = model(x, edge_index)
        result.backward()

        print(x.grad)

        return x.grad

def gnn_explainer_callback(
        model: WhiteboxGCN,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_mol: rdkit.Chem.Mol,
        pattern_mol: rdkit.Chem.Mol
):
    x = x.float()
    x = x.requires_grad_(True)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='node',
            return_type='raw',
        ),
    )
    explanation = explainer(x, edge_index)

    return explanation.node_mask


def graph_mask_explainer_callback(
        model: WhiteboxGCN,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_mol: rdkit.Chem.Mol,
        pattern_mol: rdkit.Chem.Mol
):
    x = x.float()
    x = x.requires_grad_(True)

    explainer = Explainer(
        model=model,
        algorithm=GraphMaskExplainer(1),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='node',
            return_type='raw',
        ),
    )
    explanation = explainer(x, edge_index)

    return explanation.node_mask


def ground_truth_callback(
        model: WhiteboxGCN,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_mol: rdkit.Chem.Mol,
        pattern_mol: rdkit.Chem.Mol
):
    important_permutation = get_matching_node_ids(target_mol, pattern_mol)
    node_mask = torch.zeros(x.shape[0]).float()
    for idx in important_permutation:
        node_mask[idx] = 1.0

    return node_mask
