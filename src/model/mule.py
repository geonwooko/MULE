import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_conv import GraphConvLayer
from .mga import MGA

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        return loss.mean()


class MuLe(nn.Module):
    '''Multi-Grained Graph Learning for Multi-Behavior Recommendation'''
    def __init__(self, data, emb_dim, gnn_layers, tda_layers):
        super(MuLe, self).__init__()
        self.edge_dict = data['edge_dict']
        
        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.emb_dim = emb_dim
        self.gnn_layers = gnn_layers
        self.tda_layers = tda_layers
        
        self.bsg_types = data['bsg_types']
        self.tcb_types = data['tcb_types'] # target-complemented behaviors
        self.tib_types = data['tib_types'] # target-intersected behaviors
        self.trbg_types = self.tcb_types + self.tib_types
        self.total_behaviors = ['ubg'] + self.bsg_types + self.trbg_types
        
        self.mga = MGA(emb_dim, self.bsg_types, self.trbg_types)
        self.bpr_loss = BPRLoss()

        self.user_embedding = nn.Embedding(self.n_users+1, emb_dim, padding_idx=0) # index 0 is padding
        self.item_embedding = nn.Embedding(self.n_items+1, emb_dim, padding_idx=0) # index 0 is padding
        
        self.convs = nn.ModuleDict()
        for behavior_type in self.total_behaviors:
            if behavior_type in self.tcb_types:
                self.convs[behavior_type] = nn.ModuleList([GraphConvLayer(emb_dim, emb_dim, 'tda') for _ in range(self.tda_layers)])
            else:
                self.convs[behavior_type] = nn.ModuleList([GraphConvLayer(emb_dim, emb_dim, 'gcn') for _ in range(self.gnn_layers)])
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.mga.reset_parameters()
            
    def propagate(self, x, edge_index, behavior_type, target_emb=None):
        result = [x]
        for i, conv in enumerate(self.convs[behavior_type]):
            x = conv(x, edge_index, target_emb)
            x = F.normalize(x, dim=1)
            result.append(x/(i+1))
        result = torch.stack(result, dim=1)
        x = result.sum(dim=1)
        return x
    
    def forward(self):
        edge_dict = self.edge_dict
        emb_dict = dict()

        ## Unified behavior graph aggregation ##
        init_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        ubg_emb = self.propagate(init_emb, edge_dict['ubg'], 'ubg')
        emb_dict['ubg'] = ubg_emb
        
        ## Behavior-speicfic graph aggregation ##
        for behavior_type in self.bsg_types:
            previous_emb = ubg_emb
            bsg_emb = self.propagate(previous_emb, edge_dict[behavior_type], behavior_type)
            emb_dict[behavior_type] = bsg_emb
        
        ## Target-related behavior graph aggregation ##
        # Target-intersected behavior graph aggregation
        for behavior_type in self.tib_types:
            previous_behavior = behavior_type.split('_')[0] # view or cart or collect
            previous_emb = emb_dict[previous_behavior]
            tib_emb = self.propagate(previous_emb, edge_dict[behavior_type], behavior_type)
            emb_dict[behavior_type] = tib_emb
            
        # Target-complemented behavior graph aggregation
        for behavior_type in self.tcb_types:
            previous_behavior = behavior_type.split('_')[0] # view or cart or collect
            previous_emb = emb_dict[previous_behavior]
            target_emb = emb_dict['buy']
            tcb_emb = self.propagate(previous_emb, edge_dict[behavior_type], behavior_type, target_emb=target_emb)
            emb_dict[behavior_type] = tcb_emb
            
        final_emb = self.mga(emb_dict)
        emb_dict['final'] = final_emb
        return emb_dict
    
    def loss(self, users, pos_idx, neg_idx):
        emb_dict = self.forward()
        user_emb, item_emb = torch.split(emb_dict['final'], [self.n_users+1, self.n_items+1], dim=0)
        p_score = (user_emb[users] * item_emb[pos_idx]).sum(dim=1)
        n_score = (user_emb[users] * item_emb[neg_idx]).sum(dim=1)
        return self.bpr_loss(p_score, n_score)

    def predict(self, users):
        final_embeddings = self.forward()['final']
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.n_users + 1, self.n_items + 1])

        user_emb = final_user_emb[users.long()]
        scores = torch.matmul(user_emb, final_item_emb.transpose(0, 1))
        return scores