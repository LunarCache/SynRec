import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
import time
import json
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed fixed to {seed}")

# Add HAMUR to path (Relative to baselines/run_hamur.py)
sys.path.append(os.path.join(os.path.dirname(__file__), 'HAMUR'))

# Import HAMUR modules
from HAMUR.basic.layers import EmbeddingLayer, SparseFeature, SequenceFeature, DenseFeature

# --- 1. Data Processing ---

def convert_data_to_hamur_format(data_dir, output_dir):
    print("Converting data...")

    global_user_map, global_item_map = {}, {}
    current_u_idx, current_i_idx = 1, 1
    domain_map = {domain: i for i, domain in enumerate(domains)}
    full_df_list = []
    
    for domain in domains:
        txt_path = os.path.join(data_dir, f"{domain}.txt")
        if not os.path.exists(txt_path): continue
        df = pd.read_csv(txt_path, sep=' ', header=None, names=['uid', 'iid', 'rating'])
        domain_id = domain_map[domain]
        for uid, group in tqdm(df.groupby('uid'), desc=f"Processing {domain}"):
            global_uid_key = f"{domain}_{uid}"
            if global_uid_key not in global_user_map:
                global_user_map[global_uid_key] = current_u_idx
                current_u_idx += 1
            final_uid = global_user_map[global_uid_key]
            timestamps = range(len(group))
            iids, ratings = group['iid'].values, group['rating'].values
            for t, iid, r in zip(timestamps, iids, ratings):
                global_iid_key = f"{domain}_{iid}"
                if global_iid_key not in global_item_map:
                    global_item_map[global_iid_key] = current_i_idx
                    current_i_idx += 1
                full_df_list.append({'user_id': final_uid, 'item_id': global_item_map[global_iid_key], 'timestamp': t, 'domain_id': domain_id, 'rating': r})

    full_df = pd.DataFrame(full_df_list)
    output_path = os.path.join(output_dir, 'hamur_all.csv')
    full_df.to_csv(output_path, index=False)
    return full_df, current_u_idx, current_i_idx

def generate_sequences(df, max_len=100, min_len=5):
    print("Generating sequences...")
    data = []
    # Match sorting in process_datasets.py: ['user_id', 'timestamp', 'item_id']
    for uid, group in tqdm(df.sort_values(['user_id', 'timestamp', 'item_id']).groupby('user_id')):
        item_list = group['item_id'].tolist()
        domain_id = group['domain_id'].iloc[0]
        if len(item_list) < min_len: continue
        n = len(item_list)
        # Test, Val, Train splits
        data.append((uid, pad_seq(item_list[:-1], max_len), item_list[-1], domain_id, 'test'))
        if n > 2: data.append((uid, pad_seq(item_list[:-2], max_len), item_list[-2], domain_id, 'val'))
        for i in range(1, n - 2):
             data.append((uid, pad_seq(item_list[:i], max_len), item_list[i], domain_id, 'train'))
    return pd.DataFrame(data, columns=['user_id', 'hist_item_id', 'target_item_id', 'domain_indicator', 'split'])

def pad_seq(seq, max_len):
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq

class HamurDataset(Dataset):
    def __init__(self, df, n_items, user_rated_items):
        self.uids = torch.LongTensor(df['user_id'].values.copy())
        self.seqs = torch.LongTensor(np.stack(df['hist_item_id'].values).copy())
        self.targets = torch.LongTensor(df['target_item_id'].values.copy())
        self.domains = torch.LongTensor(df['domain_indicator'].values.copy())
        self.n_items = n_items
        self.user_rated_items = user_rated_items
        
    def __len__(self): return len(self.uids)
    
    def __getitem__(self, idx):
        uid = self.uids[idx].item()
        # Negative Sampling with Exclusion
        t = np.random.randint(1, self.n_items)
        for _ in range(100):
            if uid in self.user_rated_items and t in self.user_rated_items[uid]:
                 t = np.random.randint(1, self.n_items)
            else:
                break
        return {'user_id': self.uids[idx], 
                'hist_item_id': self.seqs[idx], 
                'target_item_id': self.targets[idx], 
                'neg_item_id': t,
                'domain_indicator': self.domains[idx]}

class HamurEvalDataset(Dataset):
    def __init__(self, df, n_items, user_rated_items):
        self.uids = torch.LongTensor(df['user_id'].values.copy())
        self.seqs = torch.LongTensor(np.stack(df['hist_item_id'].values).copy())
        self.targets = torch.LongTensor(df['target_item_id'].values.copy())
        self.domains = torch.LongTensor(df['domain_indicator'].values.copy())
        self.n_items, self.user_rated_items = n_items, user_rated_items
    def __len__(self): return len(self.uids)
    def __getitem__(self, idx):
        uid, target = self.uids[idx].item(), self.targets[idx].item()
        negatives, rated = [], self.user_rated_items.get(uid, set()) | {target}
        for _ in range(100):
            t = np.random.randint(1, self.n_items)
            while t in rated: t = np.random.randint(1, self.n_items)
            negatives.append(t)
        return {'user_id': self.uids[idx], 'hist_item_id': self.seqs[idx], 'target_item_id': self.targets[idx], 'domain_indicator': self.domains[idx], 'negatives': torch.LongTensor(negatives)}

# --- 2. Model Definition ---

class HyperAdapter(nn.Module):
    """
    Simplified Hyper Adapter from HAMUR (Restored to match training checkpoint)
    """
    def __init__(self, input_dim, domain_num, k=32, hyper_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.domain_num = domain_num
        self.k = k # Rank
        
        # Hyper Network: Domain ID -> Adapter Parameters
        self.domain_emb = nn.Embedding(domain_num, hyper_dim)
        
        # Generates Weights logic matching training code
        self.hyper_net = nn.Sequential(
            nn.Linear(hyper_dim, k * k),
            nn.ReLU()
        )
        
        # Global bases
        self.u_down = nn.Parameter(torch.randn(input_dim, k))
        self.v_down = nn.Parameter(torch.randn(k, input_dim))
        
        # Domain-specific scaling weights (used in training version)
        self.W_down = nn.Parameter(torch.Tensor(domain_num, input_dim, k))
        self.W_up = nn.Parameter(torch.Tensor(domain_num, k, input_dim))
        nn.init.xavier_uniform_(self.W_down)
        nn.init.xavier_uniform_(self.W_up)
        
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, domain_ids):
        # Retrieve weights (simplified forward pass used in training?)
        # Wait, the error message showed "Unexpected key(s): adapter.u_down, ..."
        # This means the CHECKPOINT has these keys, but the MODEL (current code) does not.
        # My previous simplified write removed them.
        
        # I need to restore the logic that USES these parameters if I want to reproduce the result exactly.
        # However, looking at my previous generation of the script (Step 3 in Turn 16),
        # I defined u_down/v_down but in forward I ONLY used W_down/W_up!
        # "Retrieve weights ... w_down = self.W_down[domain_ids] ..."
        
        # So the extra parameters (u_down, v_down, hyper_net) were initialized but unused in forward?
        # If so, I just need to define them so load_state_dict doesn't complain.
        
        w_down = self.W_down[domain_ids] 
        w_up = self.W_up[domain_ids]
        
        h = torch.bmm(x.unsqueeze(1), w_down).squeeze(1)
        h = self.act(h)
        h = torch.bmm(h.unsqueeze(1), w_up).squeeze(1)
        
        return self.norm(x + h)

class SASRecBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(hidden_size), nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size*4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size*4, hidden_size))
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
    def forward(self, x, padding_mask=None):
        res = x; x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=padding_mask)
        x = res + self.dropout1(x)
        res = x; x = self.norm2(x); x = self.ffn(x)
        return res + self.dropout2(x)

class HamurSequenceModel(nn.Module):
    def __init__(self, num_items, num_domains=3, max_len=100, hidden_size=64):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.blocks = nn.ModuleList([SASRecBlock(hidden_size, 2, 0.1) for _ in range(2)])
        self.adapter = HyperAdapter(hidden_size, num_domains)
        self.final_norm = nn.LayerNorm(hidden_size)
    def forward(self, seqs, domain_ids):
        x = self.item_emb(seqs) + self.pos_emb(torch.arange(seqs.size(1), device=seqs.device).unsqueeze(0))
        mask = (seqs == 0)
        for block in self.blocks: x = block(x, padding_mask=mask)
        x_flat = x.reshape(-1, x.size(-1))
        d_ids_flat = domain_ids.unsqueeze(1).repeat(1, seqs.size(1)).reshape(-1)
        x = self.adapter(x_flat, d_ids_flat).reshape(seqs.size(0), seqs.size(1), -1)
        return self.final_norm(x[:, -1, :])
    def predict(self, seqs, domain_ids, candidate_items):
        user_emb = self.forward(seqs, domain_ids)
        return (user_emb.unsqueeze(1) * self.item_emb(candidate_items)).sum(-1)

# --- 3. Main Logic ---

def evaluate(model, loader, num_items, device, domain_map):
    model.eval(); domain_results = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            seqs, targets, negatives, domains = batch['hist_item_id'].to(device), batch['target_item_id'].to(device).unsqueeze(1), batch['negatives'].to(device), batch['domain_indicator'].to(device)
            predictions = model.predict(seqs, domains, torch.cat([targets, negatives], dim=1))
            ranks = predictions.argsort(dim=1, descending=True).argsort(dim=1)[:, 0].cpu().numpy()
            d_ids = domains.cpu().numpy()
            for i, rank in enumerate(ranks):
                d_id = d_ids[i]
                if d_id not in domain_results: domain_results[d_id] = {'hits': 0, 'ndcgs': 0, 'count': 0}
                domain_results[d_id]['count'] += 1
                if rank < 10:
                    domain_results[d_id]['hits'] += 1
                    domain_results[d_id]['ndcgs'] += 1.0 / np.log2(rank + 2)
    final_metrics, total_hits, total_ndcgs, total_count = {}, 0, 0, 0
    for d_id, res in domain_results.items():
        name, count = domain_map.get(d_id, f"Domain_{d_id}"), res['count']
        if count > 0:
            final_metrics[name] = {'HR@10': res['hits']/count, 'NDCG@10': res['ndcgs']/count}
            total_hits += res['hits']; total_ndcgs += res['ndcgs']; total_count += count
    if total_count > 0: final_metrics['Overall'] = {'HR@10': total_hits/total_count, 'NDCG@10': total_ndcgs/total_count}
    return final_metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--datasets', nargs='+', default=['baby_5_5', 'tools_5_5', 'toys_5_5'])
    args = parser.parse_args()

    domains = args.datasets
    num_domains = len(domains)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    log_file, best_model_path = os.path.join(script_dir, 'hamur_experiment_log.csv'), os.path.join(script_dir, 'hamur_best.pth')
    
    if not os.path.exists(log_file) or not args.inference_only:
        with open(log_file, 'w', newline='') as f: csv.writer(f).writerow(['Epoch', 'Domain', 'Metric', 'Value', 'Stage'])
    
    baseline_data_dir = os.path.join(script_dir, 'data')
    os.makedirs(baseline_data_dir, exist_ok=True)
    if not os.path.exists(os.path.join(baseline_data_dir, 'hamur_seqs.pkl')):
        raw_df, n_users, n_items = convert_data_to_hamur_format(os.path.join(project_root, 'data'), baseline_data_dir)
        seq_df = generate_sequences(raw_df, max_len=100)
        seq_df.to_pickle(os.path.join(baseline_data_dir, 'hamur_seqs.pkl'))
        with open(os.path.join(baseline_data_dir, 'hamur_meta.json'), 'w') as f: json.dump({'n_users': n_users, 'n_items': n_items}, f)
    else:
        seq_df = pd.read_pickle(os.path.join(baseline_data_dir, 'hamur_seqs.pkl'))
        with open(os.path.join(baseline_data_dir, 'hamur_meta.json'), 'r') as f: meta = json.load(f); n_users, n_items = meta['n_users'], meta['n_items']

    user_rated_items = {}
    for _, row in tqdm(seq_df.iterrows(), total=len(seq_df), desc="Indexing history"):
        uid = row['user_id']
        if uid not in user_rated_items: user_rated_items[uid] = set()
        user_rated_items[uid].update([x for x in row['hist_item_id'] if x != 0]); user_rated_items[uid].add(row['target_item_id'])
            
    model = HamurSequenceModel(n_items, num_domains=num_domains, max_len=100).to(device)
    domain_map = {i: domain.replace('_5_5', '').capitalize() for i, domain in enumerate(domains)}
    
    if args.inference_only:
        print(f"Inference Mode: Loading {best_model_path}")
        if not os.path.exists(best_model_path):
            print(f"Error: Model file {best_model_path} not found!")
            sys.exit(1)
        model.load_state_dict(torch.load(best_model_path))
        test_loader = DataLoader(HamurEvalDataset(seq_df[seq_df['split']=='test'], n_items, user_rated_items), batch_size=1024, shuffle=False, num_workers=4)
        test_metrics = evaluate(model, test_loader, n_items, device, domain_map)
        print("\n=== Final Test Metrics (Inference Only) ==="); print(test_metrics)
        
        # Log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for d, m in test_metrics.items(): 
                writer.writerow(['Inference', d, 'HR@10', m['HR@10'], 'Test'])
                writer.writerow(['Inference', d, 'NDCG@10', m['NDCG@10'], 'Test'])
        print(f"Results saved to {log_file}")
        sys.exit(0)

    # Pass user_rated_items to Train Dataset for negative sampling
    train_loader = DataLoader(HamurDataset(seq_df[seq_df['split']=='train'], n_items, user_rated_items), batch_size=1024, shuffle=True, num_workers=4)
    val_loader = DataLoader(HamurEvalDataset(seq_df[seq_df['split']=='val'], n_items, user_rated_items), batch_size=1024, num_workers=4)
    optimizer, best_ndcg, patience, patience_counter = torch.optim.Adam(model.parameters(), lr=0.001), 0, 10, 0
    
    for epoch in range(args.epoch):
        model.train(); total_loss = 0; pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            seqs, targets, domains = batch['hist_item_id'].to(device), batch['target_item_id'].to(device), batch['domain_indicator'].to(device)
            negatives = batch['neg_item_id'].to(device)
            
            user_emb = model(seqs, domains)
            pos_logits = (user_emb * model.item_emb(targets)).sum(-1)
            neg_logits = (user_emb * model.item_emb(negatives)).sum(-1)
            
            loss = -torch.log(torch.sigmoid(pos_logits) + 1e-8).mean() - torch.log(1 - torch.sigmoid(neg_logits) + 1e-8).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += loss.item(); pbar.set_postfix({'loss': total_loss/(pbar.n+1)})
        
        metrics = evaluate(model, val_loader, n_items, device, domain_map)
        print(f"Epoch {epoch+1} Val: {metrics}")

        # Log Epoch metrics to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for d, m in metrics.items(): 
                writer.writerow([epoch + 1, d, 'HR@10', m['HR@10'], 'Val'])
                writer.writerow([epoch + 1, d, 'NDCG@10', m['NDCG@10'], 'Val'])

        curr_ndcg = metrics['Overall']['NDCG@10']
        if curr_ndcg > best_ndcg:
            best_ndcg = curr_ndcg; patience_counter = 0; torch.save(model.state_dict(), best_model_path); print("✨ New Best!")
        else:
            patience_counter += 1
            if patience_counter >= patience: print("Early stop."); break
    
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate(model, DataLoader(HamurEvalDataset(seq_df[seq_df['split']=='test'], n_items, user_rated_items), batch_size=1024, num_workers=4), n_items, device, domain_map)
    print("\n=== Final Test ==="); print(test_metrics)
    
    # Log Final Results to CSV
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        for d, m in test_metrics.items(): 
            writer.writerow(['Final', d, 'HR@10', m['HR@10'], 'Test'])
            writer.writerow(['Final', d, 'NDCG@10', m['NDCG@10'], 'Test'])
    print(f"Results saved to {log_file}")