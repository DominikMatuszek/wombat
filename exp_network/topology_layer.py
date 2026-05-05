import torch

from torch_geometric.nn import MessagePassing
from typing import Optional


class TopologyLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr=None)

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, ptr: Optional[torch.Tensor] = None,
                  dim_size: Optional[int] = None):
        aggregated = []

        for address in range(dim_size):
            mask = (index == address)
            relevant_inputs = inputs[mask]
            aggregated.append(relevant_inputs)

        longest_length = max(len(tensor) for tensor in aggregated)

        for i in range(len(aggregated)):
            current_length = len(aggregated[i])
            if current_length < longest_length:
                padding = -1 * torch.ones((longest_length - current_length, inputs.size(1)), device=inputs.device)
                aggregated[i] = torch.cat([aggregated[i], padding], dim=0)

        return torch.stack(aggregated, dim=0).squeeze(2)

    def forward(self, x, edge_index):
        output = self.propagate(edge_index, x=x)

        return output

    def message(self, x_j):
        identifier = x_j[:, 0]
        identifier = torch.unsqueeze(identifier, dim=1)
        return identifier

    def update(self, aggr_out, x):
        # 0 -> original id
        # 1 -> atomic number (-atomic_number-1 so that readout does not mistake it for ID or padding)
        # rest -> neighbours, -1 if padding
        return torch.cat([x, aggr_out], dim=1)


def main():
    l = TopologyLayer()

    print(
        l.aggregate(
            torch.Tensor([
                [0],
                [0],
                [1],
                [1],
                [2],
                [2]
            ]),
            torch.Tensor([1, 2, 0, 2, 0, 1]).long(),
            dim_size=3
        )
    )

    print("#" * 20, "Test forward", "#" * 20)

    print(
        l.forward(
            torch.Tensor([
                [0, 42],
                [1, 42],
                [2, 42]
            ]),
            torch.Tensor([
                [0, 1, 2, 0, 1, 2],
                [1, 2, 0, 2, 0, 1]
            ]).long()
        )
    )


if __name__ == "__main__":
    main()
