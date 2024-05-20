import os
import torch
import random
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader 
from loguru import logger

data2nusers = {
    'taobao': 15449,
    'tmall': 41738,
    'jdata': 93334
}

data2nitems = {
    'taobao': 11953,
    'tmall': 11953,
    'jdata': 24624
}

# Specific behavior
data2bsg = {
    'taobao': ['view', 'cart', 'buy'],
    'tmall': ['view', 'collect', 'cart', 'buy'],
    'jdata': ['view', 'collect', 'cart', 'buy']
}

 # Target-complemented behaviors
data2tcb = {
    'taobao': ['view_not_buy', 'cart_not_buy'],
    'tmall': ['view_not_buy', 'collect_not_buy', 'cart_not_buy'],
    'jdata': ['view_not_buy', 'collect_not_buy', 'cart_not_buy']
}

# Target-intersected behaviors
data2tib = {
    'taobao': ['view_buy', 'cart_buy'],
    'tmall': ['view_buy', 'collect_buy', 'cart_buy'],
    'jdata': ['view_buy', 'collect_buy', 'cart_buy']
}


class BPRDataset(Dataset):
    def __init__(self, buy_interactions, n_items):
        self.buy_interactions = {int(k):set(v) for k,v in buy_interactions.items()}
        self.all_items = set(range(1, n_items+1))
        self.users = list(self.buy_interactions.keys())
        self.total_samples = []
        
        for user in self.users:
            for pos_item in self.buy_interactions[user]:
                self.total_samples.append((user, pos_item))
                
    def __len__(self):
        return len(self.total_samples)

    def __getitem__(self, idx):
        user, pos_item = self.total_samples[idx]
        neg_candidates = list(self.all_items - self.buy_interactions[user])
        neg_item = random.choice(neg_candidates)
        
        return torch.tensor(user, dtype=torch.long), torch.tensor(pos_item, dtype=torch.long), torch.tensor(neg_item, dtype=torch.long)


def convert_edge(edge_list, n_users):
    # Item index starts from n_users + 2
    edge_list[1] += n_users + 1
    # Add reverse item to user edges 
    edge_list = torch.cat([edge_list, edge_list.flip(0)], dim=1)
    return edge_list
    
def load_data(data_dir, dataset, device, batch_size):
    logger.info('Load data')
    n_users, n_items = data2nusers[dataset], data2nitems[dataset]
    bsg_types, tcb_types, tib_types = data2bsg[dataset], data2tcb[dataset], data2tib[dataset]
    trbg_types = tcb_types + tib_types
    
    # load edges
    edge_dict = dict()
    data_dir = os.path.join(data_dir, dataset)
    for behavior_type in ['ubg'] + bsg_types + trbg_types:
        edge_list = np.loadtxt(f'{data_dir}/{behavior_type}.txt', dtype=int)
        edge_list = torch.from_numpy(edge_list).to(device).T
        edge_dict[behavior_type] = convert_edge(edge_list, n_users)
    
    # load train/test buy interactions
    with open(f'{data_dir}/train.json', 'r') as f:
        train_buy = json.load(f)
    with open(f'{data_dir}/test.json', 'r') as f:
        test_buy = json.load(f)    
    train_dataset = BPRDataset(train_buy, n_items)
    test_dataset = BPRDataset(test_buy, n_items)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    test_gt_length = [len(items) for items in test_buy.values()]
    
    data = dict()
    data['edge_dict'] = edge_dict
    data['n_users'] = n_users
    data['n_items'] = n_items
    data['bsg_types'] = bsg_types
    data['trbg_types'] = trbg_types
    data['tcb_types'] = tcb_types
    data['tib_types'] = tib_types
    data['train_loader'] = train_loader
    data['test_loader'] = test_loader
    data['train_gt'] = train_buy
    data['test_gt'] = test_buy
    data['test_gt_length'] = test_gt_length
    return data
    