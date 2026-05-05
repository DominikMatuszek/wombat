import torch 
import networkx as nx
import numpy as np
import random

from itertools import permutations
from matplotlib import pyplot as plt
from captum.attr import IntegratedGradients

def rbf_softeq(x: torch.Tensor, expected: torch.Tensor):
    diffs = x - expected
    diffs = diffs ** 2
    c = 10

    return torch.exp(-c * diffs)

def relu_softeq(x: torch.Tensor, expected: torch.Tensor):
    # 0 iff equal
    c = 1000
    
    diffs = torch.nn.ReLU()(x - expected) + torch.nn.ReLU()(expected - x)
    diffs = diffs * c

    return torch.nn.ReLU()(1 - diffs)

class ExpReadout(torch.nn.Module):
    def __init__(self, pattern: list[tuple[int, int]], types: list[int], device: str = "cpu", softeq: callable = rbf_softeq):
        super(ExpReadout, self).__init__()
        self.pattern = pattern
        self.types = types

        # Number of distinct nodes in the pattern
        self.pattern_v = len(set([entry[0] for entry in pattern] + [entry[1] for entry in pattern]))
        assert len(types) == self.pattern_v, "Types list must have an entry for each distinct node in the pattern."

        # Number of edges in the pattern
        self.pattern_e = len(pattern)
        self.device = device

        self.softeq = softeq
        
    def get_perm_list(self, node_numbers: list, pattern_v: int):
        return list(permutations(node_numbers, pattern_v))

    def forward(self, adjacency_matrix: torch.Tensor, type_matrix: torch.Tensor):
        adjacency_matrix = torch.squeeze(adjacency_matrix, dim=0).to(self.device)
        type_matrix = torch.squeeze(type_matrix, dim=0).to(self.device)

        assert len(adjacency_matrix.shape) == 2
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]

        assert len(type_matrix.shape) == 2
        assert type_matrix.shape[0] == adjacency_matrix.shape[0]

        n = adjacency_matrix.shape[0]
        node_numbers = list(range(n))

        pattern_x = [entry[0] for entry in self.pattern]
        pattern_y = [entry[1] for entry in self.pattern]

        expected_type = torch.Tensor(self.types).long()

        perm_list = self.get_perm_list(node_numbers, self.pattern_v)
        perm_tensor = torch.Tensor(perm_list).long().to(self.device)

        connections_left = perm_tensor[:, pattern_x]
        connections_right = perm_tensor[:, pattern_y]

        connected = adjacency_matrix[connections_left, connections_right]

        type_checks = type_matrix[perm_tensor, expected_type]

        correct = torch.cat([connected, type_checks], dim=1)

        connected = torch.sum(correct, dim=1)
        expected_result = self.pattern_e + self.pattern_v
        connected = self.softeq(connected, expected_result)
        total = torch.sum(connected).cpu()

        return total.unsqueeze(0).unsqueeze(0)

class FastExpReadout(ExpReadout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.guaranteed_perm_list = []
        self.max_procedural_permutations : int | None = None 

        # Caching is extremely important here. If we sample
        # new permutations every forward pass, we may mess with IG/other explainers.
        # And that would be bad.
        # NOTE: This cache does NOT care about node_numbers/pattern_v changes, as we assume these won't change!
        # NOTE: Invalidate by setting guaranteed perms or max procedural perms.
        self.cache : list[tuple[int]] | None = None

    def set_guaranteed_perm_list(self, perm_list: list[tuple[int]]):
        self.guaranteed_perm_list = perm_list
        self.cache = None
    
    def set_max_procedural_permutations(self, max_permutations: int):
        self.max_procedural_permutations = max_permutations
        self.cache = None

    def get_perm_list(self, node_numbers: list, pattern_v: int):
        if self.cache is not None:
            return self.cache
        
        if self.max_procedural_permutations is not None:
            perm_list = [tuple(random.sample(node_numbers, pattern_v)) for _ in range(self.max_procedural_permutations)]
            perm_list = self.guaranteed_perm_list + perm_list
        else:
            # Guaranteed perms will be here either way
            perms = permutations(node_numbers, pattern_v)
            perm_list = list(perms)
        
        self.cache = perm_list
      
        return perm_list