import torch
import networkx as nx 

from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from matplotlib import pyplot as plt
from captum.attr import IntegratedGradients

from topology_layer import TopologyLayer
from exp_readout import FastExpReadout, ExpReadout, rbf_softeq, relu_softeq


class WhiteboxGCN(torch.nn.Module):
    def __init__(self, exp_readout: ExpReadout, softeq: callable):
        super().__init__()
        self.tl = TopologyLayer()
        self.exp_readout = exp_readout
        self.edge_index = None

        self.softeq = softeq

    def x_to_incidence(self, x: torch.Tensor, num_nodes: int):
        incidence_matrix = torch.zeros((num_nodes, num_nodes), device=x.device)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    neighbours = x[i, 1:]  # The first element is the node identifier
                    exp_diffs = self.softeq(neighbours, j)
                    incidence_matrix[i, j] = torch.sum(exp_diffs)

        return incidence_matrix

    def x_to_type(self, x: torch.Tensor, atomic_number: int):
        x = x[:, 1]
        x = -x - 1

        x = self.softeq(x, atomic_number)

        return x

    def readout(self, x: torch.Tensor, num_nodes: int):
        incidence_matrix = self.x_to_incidence(x, num_nodes)

        hydrogens = self.x_to_type(x, atomic_number=1)
        carbons = self.x_to_type(x, atomic_number=6)
        nitrogens = self.x_to_type(x, atomic_number=7)
        oxygens = self.x_to_type(x, atomic_number=8)
        fluorines = self.x_to_type(x, atomic_number=9)

        onehot = torch.stack([hydrogens, carbons, nitrogens, oxygens, fluorines], dim=1)

        return self.exp_readout(incidence_matrix, onehot)

    def _resolve_edge_index(self, edge_index):
        if edge_index is None:
            assert self.edge_index is not None
            edge_index = self.edge_index
        return edge_index

    def forward(self, x, edge_index = None):
        x = -x - 1 # Hydrogen -> -2, etc.

        edge_index = self._resolve_edge_index(edge_index)

        assert len(x.shape) == 2
        num_nodes, _ = x.shape
        
        ids = torch.arange(num_nodes, device=x.device).unsqueeze(1).float()
        x = torch.cat([ids, x], dim=1)

        x = self.tl(x, edge_index)

        return self.readout(x, num_nodes)
    
    def set_edge_index(self, edge_index):
        self.edge_index = edge_index

    def set_guaranteed_permutations(self, perms: list):
        self.exp_readout.set_guaranteed_perm_list(perms)
    
    def clear_guaranteed_permutations(self):
        self.exp_readout.set_guaranteed_perm_list([])
    
    def set_max_procedural_permutations(self, max_perms: int | None):
        self.exp_readout.set_max_procedural_permutations(max_perms)


def get_whitebox_gcn(pattern: list[tuple[int, int]], types: list[int], max_procedural_permutations: int | None = None, device: str = "cpu", softeq: str = "rbf") -> WhiteboxGCN:
    if softeq == "rbf":
        softeq_fn = rbf_softeq
    elif softeq == "relu":
        softeq_fn = relu_softeq
    else:
        raise NotImplementedError()
    
    if max_procedural_permutations is None:
        exp_readout = ExpReadout(pattern, types, device=device, softeq=softeq_fn)
    else:
        exp_readout = FastExpReadout(pattern, types, device=device, softeq=softeq_fn)
        exp_readout.set_max_procedural_permutations(max_procedural_permutations)

    model = WhiteboxGCN(exp_readout, softeq=softeq_fn)
    return model
