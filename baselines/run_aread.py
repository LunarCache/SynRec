import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
import json
import copy
import time
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

# --- 1. Data Processing ---

def convert_data_to_aread_format(data_dir, output_dir, domains):
    print("Converting data...")
    
    global_user_map, global_item_map = {}, {}
    current_u_idx, current_i_idx = 1, 1
    domain_map = {domain: j for j, domain in enumerate(domains)}
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
    output_path = os.path.join(output_dir, 'aread_all.csv')
    full_df.to_csv(output_path, index=False)
    return full_df, current_u_idx, current_i_idx

def make_augmentation(df, aug_ratio=0.1):
    print("Generating Counterfactual Augmentation...")
    item_popularity = df.groupby('item_id').agg({'rating': ['count', 'sum']})
    item_popularity.columns = ['total_count', 'positive_count']
    # Add smoothing
    positive_smoothing, total_smoothing = 1, 2
    item_popularity['popularity'] = (item_popularity['positive_count'] + positive_smoothing) / (item_popularity['total_count'] + total_smoothing)
    
    # Identify "Cold" items (using quantile as in AREAD's code roughly)
    pop_threshold = item_popularity['popularity'].quantile(0.2) 
    cold_items = item_popularity[item_popularity['popularity'] < pop_threshold].index.to_numpy()
    
    # Identify domains
    domain_counts = df['domain_id'].value_counts()
    # In SynRec, domains are roughly balanced, but we follow logic: 
    # "Minority domains more likely to receive". If balanced, uniform.
    small_domains = domain_counts.index.tolist() # All are candidates if balanced
    
    # Candidates for augmentation: Positive interactions with cold items
    candidates = df[df['item_id'].isin(cold_items) & (df['rating'] == 1)]
    if len(candidates) == 0: candidates = df[df['item_id'].isin(cold_items)] # Fallback
        
    if len(candidates) == 0:
        print("No candidates for augmentation found.")
        return pd.DataFrame()

    # Sampling weights (Inverse Popularity)
    aug_len = int(len(df) * aug_ratio)
    # Map popularity to candidates to align with candidates index
    candidate_pop = candidates['item_id'].map(item_popularity['popularity'])
    weights = 1.0 / (candidate_pop + 1e-6)
    weights = weights / weights.sum()
    
    augmented_samples = candidates.sample(n=aug_len, replace=True, weights=weights).copy()
    
    # Re-assign domain (Simulate counterfactual: "What if this user/item was in another domain?")
    augmented_samples['domain_id'] = np.random.choice(small_domains, size=aug_len)
    augmented_samples['is_augmented'] = True
    
    print(f"Generated {len(augmented_samples)} augmented samples.")
    return augmented_samples

def generate_sequences(df, max_len=100, min_len=5):
    print("Generating sequences...")
    data = []
    # Match the sorting logic in process_datasets.py: ['uid', 'timestamp', 'sid']
    # Here uid is user_id, sid is item_id
    for uid, group in tqdm(df.sort_values(['user_id', 'timestamp', 'item_id']).groupby('user_id')):
        item_list = group['item_id'].tolist()
        if len(item_list) < min_len: continue
        
        # Determine split (Last item test, second last val)
        # Match standard SASRec evaluation: 
        # Train: [1, n-2], Val: [1, n-1], Test: [1, n]
        
        n = len(item_list)
        # Test (target is the very last item)
        data.append((uid, pad_seq(item_list[:-1], max_len), item_list[-1], group['domain_id'].iloc[-1], 'test'))
        # Val (target is the second to last item)
        if n > 2: 
            data.append((uid, pad_seq(item_list[:-2], max_len), item_list[-2], group['domain_id'].iloc[-2], 'val'))
        # Train (all previous prefixes)
        for i in range(1, n - 2):
            data.append((uid, pad_seq(item_list[:i], max_len), item_list[i], group['domain_id'].iloc[i], 'train'))
            
    return pd.DataFrame(data, columns=['user_id', 'hist_item_id', 'target_item_id', 'domain_indicator', 'split'])

def process_aug_data(aug_df, full_df, max_len=100):
    # Aug data needs sequences too. 
    # Must use the same sorting: ['user_id', 'timestamp', 'item_id']
    print("Processing augmented sequences...")
    user_histories = {}
    for uid, group in full_df.sort_values(['user_id', 'timestamp', 'item_id']).groupby('user_id'):
        items = group['item_id'].tolist()
        if len(items) > 2:
            user_histories[uid] = items[:-2] # Use training part of history
        else:
            user_histories[uid] = items
            
    aug_data = []
    for idx, row in tqdm(aug_df.iterrows(), total=len(aug_df)):
        uid = row['user_id']
        if uid in user_histories:
            hist = user_histories[uid]
            if len(hist) > 0:
                aug_data.append((uid, pad_seq(hist, max_len), row['item_id'], row['domain_id'], 'aug_train'))
                
    return pd.DataFrame(aug_data, columns=['user_id', 'hist_item_id', 'target_item_id', 'domain_indicator', 'split'])
def pad_seq(seq, max_len):
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq

class AREADDataset(Dataset):
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
        target = self.targets[idx].item()
        
        # Negative Sampling with Exclusion
        # We need 1 negative per positive for BPR/LogLoss usually
        t = np.random.randint(1, self.n_items)
        # Fallback for safety if user rated almost everything (unlikely)
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

class AREADEvalDataset(Dataset):
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

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = curr_dim
    def forward(self, x):
        return self.net(x)

class AREADLayer(nn.Module):
    def __init__(self, input_dim, n_domain, n_tower=[3, 3], tower_dims=[[32], [32]], embed_dim=64):
        super().__init__()
        self.n_domain = n_domain
        self.n_tower = n_tower # List of towers per level
        self.n_level = len(n_tower)
        self.tower_dims = tower_dims
        self.embed_dim = embed_dim
        
        # Domain Embeddings
        self.domain_emb = nn.Embedding(n_domain, embed_dim)
        # Group Embedding (Simplified: Just one group for now or simple clustering)
        self.group_embedding = nn.Embedding(n_tower[0], embed_dim) # Same as AREAD logic roughly
        
        self.towers = nn.ModuleList()
        self.tower_gates = nn.ModuleList()
        
        curr_in_dim = input_dim
        for l in range(self.n_level):
            # Towers
            # tower_dims[l] is a list of hidden dims for this level's towers
            level_towers = nn.ModuleList([
                MultiLayerPerceptron(curr_in_dim, tower_dims[l]) for _ in range(self.n_tower[l])
            ])
            self.towers.append(level_towers)
            
            # Gates (connecting l-1 to l)
            if l > 0:
                level_gates = nn.ModuleList([
                    nn.Sequential(nn.Linear(2 * embed_dim, self.n_tower[l-1])) for _ in range(self.n_tower[l])
                ])
                self.tower_gates.append(level_gates)
            
            curr_in_dim = tower_dims[l][-1] # Output of current level is input to next
            
        # Final prediction head
        # The input to the final linear layer is the output of the last level towers
        final_dim = tower_dims[-1][-1]
        self.towers_linear = nn.ModuleList([
             nn.Linear(final_dim, 1, bias=False) for _ in range(self.n_tower[-1])
        ])
        self.output_layers = nn.ModuleList([nn.Identity() for _ in range(self.n_tower[-1])]) # Sigmoid done in loss
        
        # Domain Mask (Initialized to full)
        self.domain_mask = [[torch.ones((1, self.n_tower[0]), dtype=torch.bool), ] + 
                            [torch.ones((self.n_tower[l-1], self.n_tower[l]), dtype=torch.bool) for l in range(1, self.n_level)] + 
                            [torch.ones((self.n_tower[-1], 1), dtype=torch.bool)] for _ in range(n_domain)]
                            
    def forward(self, x, domain_ids, mode='bagging', mask=None):
        # x: (batch, input_dim)
        batch_size = x.size(0)
        domain_embed = self.domain_emb(domain_ids)
        
        # Simplified "Group" embedding (just mean of domain emb for now as placeholder)
        group_embed = domain_embed # In AREAD it depends on cluster.
        
        gate_inputs = torch.cat([domain_embed, group_embed], dim=1)
        
        # Forward pass is tricky with masks.
        # We process each domain in the batch separately if they have different masks.
        # But for speed, if we use 'bagging' (ensemble), we might average all valid towers.
        
        # Simplified Forward:
        # Just use ALL towers (Full Mask) if mode != 'mask'
        # To strictly follow AREAD, we should route.
        
        # "Routing" Simulation:
        # We calculate weights for each tower based on gates.
        
        outputs = []
        
        # We iterate towers.
        # Level 0
        l0_outs = []
        for t in range(self.n_tower[0]):
             l0_outs.append(self.towers[0][t](x)) # (batch, dim)
             
        prev_outs = l0_outs
        
        for l in range(1, self.n_level):
            curr_outs = []
            for t in range(self.n_tower[l]):
                # Gate
                gate_out = F.softmax(self.tower_gates[l-1][t](gate_inputs), dim=1) # (batch, n_prev)
                # Weighted sum of prev_outs
                weighted_input = torch.zeros_like(prev_outs[0])
                for pt in range(self.n_tower[l-1]):
                    weighted_input += prev_outs[pt] * gate_out[:, pt].unsqueeze(1)
                
                curr_outs.append(self.towers[l][t](weighted_input))
            prev_outs = curr_outs
            
        # Final Level
        y_stack = []
        for t in range(self.n_tower[-1]):
             y_stack.append(self.towers_linear[t](prev_outs[t]))
             
        y = torch.stack(y_stack, dim=1).mean(dim=1) # Average of last towers
        
        return y

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

class AREADSequenceModel(nn.Module):
    def __init__(self, num_items, num_domains=3, max_len=100, hidden_size=64):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.blocks = nn.ModuleList([SASRecBlock(hidden_size, 2, 0.1) for _ in range(2)])
        
        # AREAD Adapter taking Dense Vector (hidden_size) and outputting Score (Scalar? No, Vector?)
        # Standard SASRec outputs (Batch, Hidden). Then multiply with Item Embeddings to get Logits.
        # AREAD outputs a scalar "Click Probability".
        # To Adapt: We use AREAD to generate a "Context Vector" or "Adapter Output" which we add to the User Embedding.
        # Then standard dot product.
        
        # Configure AREAD Layer: 2 Levels. Level 0: Input(64)->32. Level 1: 32->32.
        self.aread_adapter = AREADLayer(hidden_size, num_domains, n_tower=[3, 3], tower_dims=[[32], [32]], embed_dim=hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)
        
        # Projection to map AREAD scalar/vector back to hidden_size?
        # AREAD logic is: Input Features -> Towers -> Scalar.
        # We change it to: Input (User Hist) -> Towers -> Adaptation Vector (hidden_size).
        # So we modify AREADLayer last layer to output `hidden_size`.
        final_tower_dim = self.aread_adapter.tower_dims[-1][-1]
        self.aread_adapter.towers_linear = nn.ModuleList([
             nn.Linear(final_tower_dim, hidden_size, bias=False) for _ in range(self.aread_adapter.n_tower[-1])
        ])

    def forward(self, seqs, domain_ids):
        x = self.item_emb(seqs) + self.pos_emb(torch.arange(seqs.size(1), device=seqs.device).unsqueeze(0))
        mask = (seqs == 0)
        for block in self.blocks: x = block(x, padding_mask=mask)
        
        # User Representation from SASRec
        user_emb = self.final_norm(x[:, -1, :]) # (Batch, Hidden)
        
        # Apply AREAD Adapter
        # Note: AREAD expects features. We treat 'user_emb' as the features.
        adaptation = self.aread_adapter(user_emb, domain_ids) # (Batch, Hidden)
        
        # Residual Connection (Ensemble)
        final_user_emb = user_emb + adaptation
        
        return final_user_emb

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
            total_hits += res['hits']; total_ndcgs += res['ndcgs']; total_count += res['count']
    if total_count > 0: final_metrics['Overall'] = {'HR@10': total_hits/total_count, 'NDCG@10': total_ndcgs/total_count}
    return final_metrics

if __name__ == '__main__':
    fix_random_seed(42)
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
    log_file, best_model_path = os.path.join(script_dir, 'aread_experiment_log.csv'), os.path.join(script_dir, 'aread_best.pth')
    
    if not os.path.exists(log_file) or not args.inference_only:
        with open(log_file, 'w', newline='') as f: csv.writer(f).writerow(['Epoch', 'Domain', 'Metric', 'Value', 'Stage'])
    
    baseline_data_dir = os.path.join(script_dir, 'data')
    os.makedirs(baseline_data_dir, exist_ok=True)
    
    if not os.path.exists(os.path.join(baseline_data_dir, 'aread_seqs.pkl')):
        # 1. Convert
        raw_df, n_users, n_items = convert_data_to_aread_format(os.path.join(project_root, 'data'), baseline_data_dir, domains)
        
        # 2. Augment (Counterfactual)
        aug_df = make_augmentation(raw_df, aug_ratio=0.1)
        
        # 3. Generate Sequences
        seq_df = generate_sequences(raw_df, max_len=100)
        aug_seq_df = process_aug_data(aug_df, raw_df, max_len=100)
        
        seq_df.to_pickle(os.path.join(baseline_data_dir, 'aread_seqs.pkl'))
        aug_seq_df.to_pickle(os.path.join(baseline_data_dir, 'aread_aug_seqs.pkl'))
        
        with open(os.path.join(baseline_data_dir, 'aread_meta.json'), 'w') as f: json.dump({'n_users': n_users, 'n_items': n_items}, f)
    else:
        seq_df = pd.read_pickle(os.path.join(baseline_data_dir, 'aread_seqs.pkl'))
        aug_seq_df = pd.read_pickle(os.path.join(baseline_data_dir, 'aread_aug_seqs.pkl'))
        with open(os.path.join(baseline_data_dir, 'aread_meta.json'), 'r') as f: meta = json.load(f); n_users, n_items = meta['n_users'], meta['n_items']

    user_rated_items = {}
    for _, row in tqdm(seq_df.iterrows(), total=len(seq_df), desc="Indexing history"):
        uid = row['user_id']
        if uid not in user_rated_items: user_rated_items[uid] = set()
        user_rated_items[uid].update([x for x in row['hist_item_id'] if x != 0]); user_rated_items[uid].add(row['target_item_id'])
            
    model = AREADSequenceModel(n_items, num_domains=num_domains, max_len=100).to(device)
    domain_map = {i: domain.replace('_5_5', '').capitalize() for i, domain in enumerate(domains)}
    
    if args.inference_only:
        print(f"Inference Mode: Loading {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        test_loader = DataLoader(AREADEvalDataset(seq_df[seq_df['split']=='test'], n_items, user_rated_items), batch_size=1024, shuffle=False, num_workers=4)
        test_metrics = evaluate(model, test_loader, n_items, device, domain_map)
        print("\n=== Final Test Metrics ==="); print(test_metrics)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for d, m in test_metrics.items(): 
                writer.writerow(['Inference', d, 'HR@10', m['HR@10'], 'Test'])
                writer.writerow(['Inference', d, 'NDCG@10', m['NDCG@10'], 'Test'])
        sys.exit(0)

    # Combined Train Data (Normal + Augmented)
    # Actually AREAD uses Augmented data mostly for 'Regroup' steps.
    # But as a baseline, training on Augmented data is also part of the 'Counterfactual Augmentation' strategy.
    # We will mix them for the main training loop for simplicity and robustness.
    
    train_df = pd.concat([seq_df[seq_df['split']=='train'], aug_seq_df], ignore_index=True)
    
    # Pass user_rated_items to Train Dataset for negative sampling
    train_loader = DataLoader(AREADDataset(train_df, n_items, user_rated_items), batch_size=1024, shuffle=True, num_workers=4)
    val_loader = DataLoader(AREADEvalDataset(seq_df[seq_df['split']=='val'], n_items, user_rated_items), batch_size=1024, num_workers=4)
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
    test_metrics = evaluate(model, DataLoader(AREADEvalDataset(seq_df[seq_df['split']=='test'], n_items, user_rated_items), batch_size=1024, num_workers=4), n_items, device, domain_map)
    print("\n=== Final Test ==="); print(test_metrics)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        for d, m in test_metrics.items(): 
            writer.writerow(['Final', d, 'HR@10', m['HR@10'], 'Test'])
            writer.writerow(['Final', d, 'NDCG@10', m['NDCG@10'], 'Test'])
