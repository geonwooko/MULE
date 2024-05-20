import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MUG Settings')
    parser.add_argument('--dataset', type=str, default='taobao', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing the data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='Directory of model checkpoint')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for target data')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--gnn_layers', type=int , default=1, help='Number of gnn layers')
    parser.add_argument('--tda_layers', type=int, default=4, help='Number of tda layers')
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device')
    parser.add_argument('--topk', type=int, default=10, help='Top-k items')
    return parser.parse_args()