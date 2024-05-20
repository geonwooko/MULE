import torch
import torch.nn as nn
import torch.nn.functional as F

class MGA(nn.Module):
    """Multi-grained embedding aggregator."""
    def __init__(self, emb_dim, bsg_types, trbg_types):
        super(MGA, self).__init__()
        self.emb_dim = emb_dim
        self.bsg_types = bsg_types
        self.trbg_types = trbg_types
        self.lin = nn.ModuleList([nn.Linear(emb_dim*2, 1),
                                  nn.Linear(emb_dim*2, 1)])
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin[0].weight)
        nn.init.xavier_uniform_(self.lin[1].weight)
    
    def forward(self, emb_dict):
        key = emb_dict['buy'] # [N, D]
        for i, behavior_types in enumerate([self.bsg_types, self.trbg_types]):
            key = key.unsqueeze(1).repeat(1, len(behavior_types), 1) # [N, |B|, D]
            query_emb = torch.stack([emb_dict[behavior] for behavior in behavior_types], dim=1) # [N, |B|, D]
            concat_emb = torch.concat([key, query_emb], dim=2) # [N, |B|, 2D]
            attention = self.lin[i](concat_emb).softmax(dim=1) # [N, |B|, 1]
            updated_emb = (attention * query_emb).sum(dim=1) # [N, D]
            key = updated_emb
        
        return updated_emb
