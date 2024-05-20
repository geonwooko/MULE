import torch
import numpy as np

from loguru import logger
from tqdm import tqdm
from .metrics import ndcg, hit

class Trainer:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args
        self.topk = args.topk
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        train_loader = self.data['train_loader']
        with tqdm(total=len(train_loader), desc='Training', unit='batch', leave=False) as pbar:
            for user_indices, pos_indices, neg_indices in train_loader:
                user_indices, pos_indices, neg_indices = user_indices.to(self.args.device), pos_indices.to(self.args.device), neg_indices.to(self.args.device)
                loss = self.model.loss(user_indices, pos_indices, neg_indices)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description(f'Batch Loss: {loss.item():.4f}')
                pbar.update()
                total_loss += loss.item()
        return total_loss / len(pbar)
    
    def train_model(self):
        num_epochs = self.args.num_epochs
        pbar = tqdm(range(num_epochs), desc='Epoch', unit='epoch', leave=False)
        for epoch in pbar:
            loss = self.train_epoch(epoch)
            pbar.set_description(f'Epoch {epoch+1} total loss: {loss:.4f}')
            pbar.update()

        torch.save(self.model.state_dict(), f'{self.args.checkpoint_dir}/{self.args.dataset}/model.pt')

    def evaluate(self):
        device = self.args.device
        
        self.model.eval()
        topk_list = []
        with torch.no_grad():
            for user_indices, pos_indices, neg_indices in tqdm(self.data['test_loader'], desc='Test', leave=False):
                user_indices, pos_indices, neg_indices = user_indices.to(device), pos_indices.to(device), neg_indices.to(device)
                scores = self.model.predict(user_indices)
                
                for user_idx in range(user_indices.size(0)):
                    user = user_indices[user_idx].item()
                    train_items = self.data['train_gt'].get(str(user), [])
                    scores[user_idx, train_items] = -np.inf
                
                _, topk_indices = torch.topk(scores, self.topk, dim=1)
                
                for idx, user in enumerate(user_indices):
                    gt_items = np.array(self.data['test_gt'][str(user.item())])
                    topk_items = topk_indices[idx].cpu().numpy()
                    mask = np.isin(topk_items, gt_items)
                    topk_list.append(mask)
        
        topk_list = np.vstack(topk_list) 
        hr_res = hit(topk_list, self.data['test_gt_length']).mean(axis=0)
        ndcg_res = ndcg(topk_list, self.data['test_gt_length']).mean(axis=0)
        hr_topk = hr_res[self.topk-1]
        ndcg_topk = ndcg_res[self.topk-1]
        
        return hr_topk, ndcg_topk