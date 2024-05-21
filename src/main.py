import os
import torch
from data import load_data
from model import MuLe, Trainer
from parser import parse_args
from utils import set_seed, print_args
from loguru import logger

def main(args):
    print_args(args)
    if args.seed is not None:
        set_seed(args.seed)
    
    # Load data
    data = load_data(args.data_dir, args.dataset, args.device, args.batch_size)
    
    # Build model
    model = MuLe(data, args.emb_dim, args.gnn_layers, args.tda_layers).to(args.device)
    trainer = Trainer(model, data, args)
    
    if args.load_checkpoint:
        # Load Checkpoint
        logger.info(f"Load checkpoint from {os.path.join(args.checkpoint_dir, args.dataset, 'model.pt')}")
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.dataset, 'model.pt'),
                                         map_location=args.device))    
    else:
        # Train model
        logger.info("Start training the model")
        trainer.train_model()
    
    # Evaluate trained model
    logger.info("Start evaluating the model")
    hr, ndcg = trainer.evaluate()
    logger.info(f"Test HR@{args.topk}: {hr:.4f}, NDCG@{args.topk}: {ndcg:.4f}")
    
if __name__ == '__main__':
    args = parse_args()
    
    main(args)