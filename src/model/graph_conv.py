import torch
from torch_geometric.typing import OptTensor
from torch import Tensor
from torch_scatter import scatter_add, scatter_softmax
from torch_geometric.nn.conv import MessagePassing

class GraphConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, norm_type):
        super(GraphConvLayer, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
    
    # LightGCN normalization
    def gcn_norm(self, edge_index, num_nodes):
        edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    # Target-guided denoising attention
    def tda_norm(self, edge_index, x, target_emb, num_nodes):
        key, query = edge_index[0], edge_index[1] # row, col
        attention_score = (target_emb[key] * x[query]).sum(dim=-1, keepdim=True)
        edge_weight = scatter_softmax(attention_score, key, dim=0)
        return edge_index, edge_weight
        
    def forward(self, x, edge_index, target_emb=None):
        num_nodes = x.size(0)
        
        if self.norm_type == 'tda': 
            edge_index, edge_weight = self.tda_norm(edge_index, x, target_emb, num_nodes)
        elif self.norm_type == 'gcn':
            edge_index, edge_weight = self.gcn_norm(edge_index, num_nodes)
        else:
            raise ValueError('Invalid normalization type')
        
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j