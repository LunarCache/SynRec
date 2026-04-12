import sys
import os
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from tqdm import tqdm

# ANSI color codes for terminal output
COLORS = {
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'RESET': '\033[0m'  # Resets the color to default
}

def build_index(dataset_name):

    delimiter = ' ' # 默认分隔符为空格
    if dataset_name == 'ml-100k' or dataset_name == 'beauty' or dataset_name == 'games':
        delimiter = ',' # ml-100k 使用逗号分隔

    # 使用指定的分隔符加载数据，并且只加载前两列 (userid, itemid)
    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32, delimiter=delimiter, usecols=(0, 1))

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    # Add a retry limit to prevent infinite loops
    max_retries = 100
    for _ in range(max_retries):
        t = np.random.randint(l, r)
        if t not in s:
            return t
    # If we fail to find a negative sample after max_retries, return the last one anyway.
    # This is a fallback to prevent getting stuck, even if it's a positive sample.
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    
    delimiter = ' ' # 默认分隔符为空格
    if fname == 'ml-100k' or fname == 'beauty_rated' or fname == 'games':
        delimiter = ',' # ml-100k 使用逗号分隔

    for line in f:
        parts = line.rstrip().split(delimiter) 
        u = int(parts[0]) # 获取 userid
        i = int(parts[1]) # 获取 itemid
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# --- Batched Evaluation ---

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, user_train, user_eval_data, user_to_domain, maxlen, itemnum, domain_to_item_range, use_domain_sampling, user_full_interaction, negative_sample_size=100):
        self.user_train = user_train
        self.user_eval_data = user_eval_data
        self.user_to_domain = user_to_domain
        self.maxlen = maxlen
        self.itemnum = itemnum
        self.domain_to_item_range = domain_to_item_range
        self.use_domain_sampling = use_domain_sampling
        self.negative_sample_size = negative_sample_size
        self.user_full_interaction = user_full_interaction

        # Filter for users who are actually in the evaluation set
        self.users = [u for u, items in self.user_eval_data.items() if len(items) > 0 and len(self.user_train.get(u, [])) > 0]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        
        # Correctly extract item & rating sequence from (item, rating) tuples
        seq_tuples = self.user_train.get(u, [])
        item_seq = [item[0] for item in seq_tuples]
        rating_seq = [item[1] for item in seq_tuples]
        if len(item_seq) > self.maxlen:
            item_seq = item_seq[-self.maxlen:]
            rating_seq = rating_seq[-self.maxlen:]
        
        rated = set(self.user_full_interaction.get(u, []))
        rated.add(0)
        
        # Correctly extract the true item ID from the (item, rating) tuple in the evaluation set
        true_item_tuple = self.user_eval_data[u][0]
        true_item = true_item_tuple[0]
        
        item_idx = [true_item]
        domain_id = self.user_to_domain[u]

        # Negative sampling
        item_range = (1, self.itemnum)
        if self.use_domain_sampling:
            domain_range = self.domain_to_item_range.get(domain_id)
            if domain_range:
                item_range = domain_range

        for _ in range(self.negative_sample_size):
            t = np.random.randint(item_range[0], item_range[1] + 1)
            while t in rated:
                t = np.random.randint(item_range[0], item_range[1] + 1)
            item_idx.append(t)
            
        return {
            'uid': u,
            'seq': torch.LongTensor(item_seq),
            # Provide rating prefix aligned with item_seq (exclude last target item implicitly as eval set uses last interaction)
            'rating_seq': torch.LongTensor(rating_seq),
            'item_idx': torch.LongTensor(item_idx),
            'domain_id': domain_id,
            'true_item': true_item
        }

def eval_collate_fn(batch):
    uids = [item['uid'] for item in batch]
    seqs = [item['seq'] for item in batch]
    rating_seqs = [item['rating_seq'] for item in batch]
    item_indices = torch.stack([item['item_idx'] for item in batch])
    domain_ids = [item['domain_id'] for item in batch]
    true_items = [item['true_item'] for item in batch]

    # Pad sequences
    batch_maxlen = max(len(s) for s in seqs)
    padded_seqs = torch.zeros(len(batch), batch_maxlen, dtype=torch.long)
    padded_rating_seqs = torch.zeros(len(batch), batch_maxlen, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded_seqs[i, -len(s):] = s
    for i, rs in enumerate(rating_seqs):
        padded_rating_seqs[i, -len(rs):] = rs

    return torch.LongTensor(uids), padded_seqs, padded_rating_seqs, item_indices, torch.LongTensor(domain_ids), torch.LongTensor(true_items)


def evaluate_batched(model, dataset, args, eval_type='valid'):
    model.eval() # Set model to evaluation mode
    
    [train, valid, test, user_to_domain, usernum, itemnum, domain_to_item_range] = dataset
    
    eval_data = valid if eval_type == 'valid' else test
    desc = f"{COLORS['GREEN']}Evaluating ({eval_type.capitalize()}){COLORS['RESET']}"

    # Create a complete history of user interactions for correct negative sampling
    user_full_interaction = defaultdict(list)
    for u, items_with_ratings in train.items(): user_full_interaction[u].extend([item[0] for item in items_with_ratings])
    for u, items_with_ratings in valid.items(): user_full_interaction[u].extend([item[0] for item in items_with_ratings])
    for u, items_with_ratings in test.items(): user_full_interaction[u].extend([item[0] for item in items_with_ratings])

    eval_dataset = EvalDataset(
        train,
        eval_data,
        user_to_domain,
        args.maxlen,
        itemnum,
        domain_to_item_range,
        args.use_domain_sampling_for_evaluation,
        user_full_interaction,
        getattr(args, 'eval_negative_sample_size', 100)
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=eval_collate_fn,
        pin_memory=True
    )

    domain_metrics = defaultdict(lambda: {
        'NDCG@5': 0.0, 'HT@5': 0.0, 'MRR@5': 0.0,
        'NDCG@10': 0.0, 'HT@10': 0.0, 'MRR@10': 0.0,
        'count': 0
    })

    import time
    t_eval_start = time.time()
    with torch.no_grad():
        for u, seq, rating_seq, item_idx, domain_id_batch, true_items in tqdm(eval_loader, desc=desc, colour='green'):
            # Move tensors to device for consistent indexing and computation
            u = u.to(args.device)
            seq = seq.to(args.device)
            rating_seq = rating_seq.to(args.device)
            item_idx = item_idx.to(args.device)
            domain_id_batch = domain_id_batch.to(args.device)
            true_items = true_items.to(args.device)
            
            if getattr(args, 'full_ranking_eval', False):
                # Full ranking: compute rank of true item among all candidates (domain-limited or global)
                batch_size = seq.size(0)
                ranks = torch.zeros(batch_size, dtype=torch.long, device=args.device)

                # 1) Score of true item per user
                true_items_unsq = true_items.unsqueeze(1)
                s_true = model.predict(u, seq, true_items_unsq, domain_id_batch, rating_seqs=rating_seq).squeeze(1)

                # 2) Iterate by domain groups within the batch to reuse candidate sets
                unique_domains = torch.unique(domain_id_batch)
                for d in unique_domains:
                    mask = (domain_id_batch == d)
                    idxs = torch.where(mask)[0]
                    if idxs.numel() == 0:
                        continue

                    # Candidate range
                    if args.use_domain_sampling_for_evaluation:
                        start_item, end_item = dataset[6][d.item()]
                        candidates = torch.arange(start_item, end_item + 1, device=args.device, dtype=torch.long)
                    else:
                        candidates = torch.arange(1, dataset[5] + 1, device=args.device, dtype=torch.long)

                    # Chunked scoring against candidates and count how many beat the true item
                    item_chunk = getattr(args, 'eval_item_batch_size', 4096)
                    greater_counts = torch.zeros(idxs.numel(), device=args.device, dtype=torch.long)
                    # Extract subgroup tensors
                    u_sub = u[idxs]
                    seq_sub = seq[idxs]
                    rating_sub = rating_seq[idxs]
                    domain_sub = domain_id_batch[idxs]
                    s_true_sub = s_true[idxs]

                    for start in range(0, candidates.numel(), item_chunk):
                        cand_chunk = candidates[start:start + item_chunk]
                        # Broadcast candidate chunk to all users in subgroup
                        cand_mat = cand_chunk.unsqueeze(0).expand(idxs.numel(), -1)
                        scores_chunk = model.predict(u_sub, seq_sub, cand_mat, domain_sub, rating_seqs=rating_sub)
                        # Filter out scores for items in user history
                        for i in range(idxs.numel()):
                            u_id = u_sub[i].item()
                            user_history = torch.tensor(user_full_interaction[u_id], device=args.device, dtype=torch.long)
                            mask = ~torch.isin(cand_chunk, user_history)
                            valid_scores = scores_chunk[i][mask]
                            greater_counts[i] += (valid_scores > s_true_sub[i]).sum()

                    ranks[idxs] = greater_counts + 1
            else:
                # Sampling evaluation (1 positive + N negatives)
                predictions = model.predict(u, seq, item_idx, domain_id_batch, rating_seqs=rating_seq)
                ranks = predictions.argsort(dim=1, descending=True).argsort(dim=1)[:, 0]

            for i, rank in enumerate(ranks):
                rank_item = rank.item()
                domain_id = domain_id_batch[i].item()
                metrics = domain_metrics[domain_id]
                metrics['count'] += 1
                
                if rank_item < 10:
                    metrics['NDCG@10'] += 1 / np.log2(rank_item + 2)
                    metrics['HT@10'] += 1
                    metrics['MRR@10'] += 1.0 / (rank_item + 1)
                if rank_item < 5:
                    metrics['NDCG@5'] += 1 / np.log2(rank_item + 2)
                    metrics['HT@5'] += 1
                    metrics['MRR@5'] += 1.0 / (rank_item + 1)

    # Aggregate and calculate final results
    results = {}
    domain_averages = defaultdict(list)  # 存储每个领域的平均值
    
    for domain_id, metrics in sorted(domain_metrics.items()):
        count = metrics['count']
        if count > 0:
            for key in ['NDCG@5', 'HT@5', 'MRR@5', 'NDCG@10', 'HT@10', 'MRR@10']:
                metric_val = metrics[key] / count
                results[f'domain_{domain_id}_{key}'] = metric_val
                domain_averages[key].append(metric_val)  # 收集每个领域的平均值

    # 计算各领域指标的算术平均值作为overall指标
    for key, domain_values in domain_averages.items():
        results[f'overall_{key}'] = sum(domain_values) / len(domain_values) if len(domain_values) > 0 else 0
    # 计算按领域样本数加权的 overall_weighted_ 指标
    # 收集加权需要的 (metric_name -> list of (value, count))
    weighted_collect = defaultdict(list)
    for domain_id, metrics in sorted(domain_metrics.items()):
        count = metrics['count']
        if count <= 0:
            continue
        for key in ['NDCG@5', 'HT@5', 'MRR@5', 'NDCG@10', 'HT@10', 'MRR@10']:
            val = metrics[key] / count if count > 0 else 0.0
            weighted_collect[key].append((val, count))
    for key, pairs in weighted_collect.items():
        total_c = sum(c for _, c in pairs)
        if total_c > 0:
            w_avg = sum(v * c for v, c in pairs) / total_c
        else:
            w_avg = 0.0
        results[f'overall_weighted_{key}'] = w_avg
    # 附加效率指标
    total_eval_count = 0
    for metrics in domain_metrics.values():
        total_eval_count += metrics['count']
    eval_seconds = time.time() - t_eval_start
    results['overall_eval_seconds'] = eval_seconds
    results['overall_eval_users'] = total_eval_count
    results['overall_eval_throughput_users_s'] = (total_eval_count / eval_seconds) if eval_seconds > 0 else 0.0
    
    return results

def partition_multi_domain(fnames, shared_user_ids=False):
    """
    Loads and partitions multiple datasets from the SASRec.pytorch/python/data/ directory.
    - Handles integer IDs by offsetting them to ensure global uniqueness.
    - Assigns a domain_id to each user.
    - fnames: list of dataset names, e.g., ['beauty', 'games', 'ml-100k']
    - shared_user_ids: if True, assumes input files already have global/shared IDs (no offsetting).
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    user_to_domain = {}
    domain_to_item_range = {}

    user_offset = 0
    item_offset = 0

    print('{:-^100}'.format(f"Multi-domain data partitioning (Offset Strategy: {'DISABLED (Linked)' if shared_user_ids else 'ENABLED (Disjoint)'})"))

    for domain_id, fname in enumerate(fnames):
        print(f"Processing domain {domain_id}: {fname}")
        
        file_path = os.path.join('data', f"{fname}.txt")
        if not os.path.exists(file_path):
            print(f"Warning: Data file not found at {file_path}. Skipping.")
            continue

        # Determine delimiter
        delimiter = ' '
        # if fname in ['ml-100k', 'beauty_rated', 'games']:
        #     delimiter = ','

        # First pass: find max user/item ID in the current domain to calculate offset
        local_usernum = 0
        local_itemnum = 0
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(delimiter)
                u = int(parts[0])
                i = int(parts[1])
                local_usernum = max(u, local_usernum)
                local_itemnum = max(i, local_itemnum)

        # Record the item range for this domain
        if shared_user_ids:
             # In linked mode, we need to find the range by scanning the file or assume known?
             # My process_linked_data.py makes item IDs disjoint but global.
             # So for this domain, min item ID and max item ID in the file would be the range.
             # Let's verify min/max from file content for range.
             # Re-reading file is okay.
             current_min_item = float('inf')
             current_max_item = 0
             with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(delimiter)
                    i = int(parts[1])
                    current_min_item = min(current_min_item, i)
                    current_max_item = max(current_max_item, i)
             domain_start_item = current_min_item
             domain_end_item = current_max_item
        else:
            domain_start_item = item_offset + 1
            domain_end_item = item_offset + local_itemnum
            
        domain_to_item_range[domain_id] = (domain_start_item, domain_end_item)

        # Second pass: load and process data with offsets
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) < 3: 
                    print(f"Warning: Skipping line without rating: {line}")
                    continue # Skip lines without rating
                
                if shared_user_ids:
                    # No offsets, trust the pre-processing
                    u = int(parts[0])
                    i = int(parts[1])
                else:
                    # Apply offset to create global unique IDs
                    u = int(parts[0]) + user_offset
                    i = int(parts[1]) + item_offset
                
                r = int(float(parts[2])) # Ratings can be float, convert to int
                
                usernum = max(u, usernum)
                itemnum = max(i, itemnum)
                User[u].append((i, r)) # Store as (item, rating) tuple
                user_to_domain[u] = domain_id
        
        print(f"  Domain '{fname}' -> Local Users: {local_usernum}, Local Items: {local_itemnum}")
        if not shared_user_ids:
            print(f"  Applying Offsets -> User: +{user_offset}, Item: +{item_offset}")
            # Update offsets for the next domain
            user_offset += local_usernum
            item_offset += local_itemnum
            
        print(f"  Domain Item Range: [{domain_start_item}, {domain_end_item}]")
        print(f"  Updated Global Users: {usernum}, Global Items: {itemnum}")

    print('{:-^100}'.format("Final Data Statistics"))
    print(f"Total unique users: {usernum}")
    print(f"Total unique items: {itemnum}")
    print(f"Total interactions: {sum(len(v) for v in User.values())}")
    print(f"Number of users with < 3 interactions: {sum(1 for v in User.values() if len(v) < 3)}")
    print("-" * 100)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]
            
    return [user_train, user_valid, user_test, user_to_domain, usernum, itemnum, domain_to_item_range]


# --- MoE Integration: A new, robust, iterable sampler ---
# Refactored to use PyTorch's DataLoader for efficient, parallel data loading.

class MoerecDataset(torch.utils.data.Dataset):
    def __init__(self, user_train, user_to_domain, usernum, itemnum, maxlen, args, domain_to_item_range):
        self.user_train = user_train
        self.user_to_domain = user_to_domain
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.args = args
        self.domain_to_item_range = domain_to_item_range

        # Store a list of all user IDs that have sequences longer than 1
        self.valid_users = [u for u, seq in user_train.items() if len(seq) > 1]
        # Create a mapping from user ID to its index in the valid_users list
        self.user_to_idx = {u: i for i, u in enumerate(self.valid_users)}
        
        print(f"MoerecDataset initialized with {len(self.valid_users)} users.")

    def __len__(self):
        return len(self.valid_users)

    def __getitem__(self, idx):
        # The index now directly corresponds to the position in self.valid_users
        uid = self.valid_users[idx]
        user_sequence = self.user_train[uid]

        # Truncate the sequence if it's longer than the max allowed length
        if len(user_sequence) > self.maxlen:
            user_sequence = user_sequence[-self.maxlen:]
        
        # The input sequence is all items except the last one
        item_sequence = [item[0] for item in user_sequence]
        rating_sequence = [item[1] for item in user_sequence]

        seq = item_sequence[:-1]
        rating_seq = rating_sequence[:-1]
        pos = item_sequence[1:]
        
        # Sample a negative item for each positive item
        neg = []
        ts = set(item_sequence)
        domain_id = self.user_to_domain[uid]
        
        # Determine the item range for negative sampling based on args
        if self.args.use_domain_sampling:
            item_range = self.domain_to_item_range.get(domain_id)
            if item_range is None:
                item_range = (1, self.itemnum)
        else:
            item_range = (1, self.itemnum)

        for _ in range(len(pos)):
            neg_item = random_neq(item_range[0], item_range[1] + 1, ts)
            neg.append(neg_item)

        # Return a dictionary of tensors/values
        return {
            'uid': uid,
            'seq': np.array(seq, dtype=np.int32),
            'rating_seq': np.array(rating_seq, dtype=np.int32),
            'pos': np.array(pos, dtype=np.int32),
            'neg': np.array(neg, dtype=np.int32),
            'domain_id': domain_id
        }

class StratifiedSampler(torch.utils.data.Sampler):
    """
    A custom sampler for PyTorch DataLoader that implements stratified sampling.
    It ensures that each batch contains a proportional representation of users
    from different domains, preserving the original training logic while benefiting
    from DataLoader's performance.
    """
    def __init__(self, dataset: MoerecDataset):
        self.dataset = dataset
        
        # 1. Group user *indices* by domain
        self.domain_to_indices = defaultdict(list)
        for i, uid in enumerate(self.dataset.valid_users):
            domain_id = self.dataset.user_to_domain[uid]
            self.domain_to_indices[domain_id].append(i)
            
        # 2. Calculate domain weights for proportional sampling
        self.total_size = len(self.dataset)
        self.domain_weights = {d: len(u) / self.total_size for d, u in self.domain_to_indices.items()}
        
        print(f"Stratified Sampler initialized: {self.total_size} users.")
        print(f"Domain distribution: { {d: f'{w:.2%}' for d, w in self.domain_weights.items()} }")

    def __iter__(self):
        # 1. Shuffle indices within each domain
        shuffled_domain_indices = {d: np.random.permutation(indices) for d, indices in self.domain_to_indices.items()}
        
        # 2. Create a flat list of all indices, maintaining stratified order
        all_indices_stratified = []
        domain_iters = {d: iter(indices) for d, indices in shuffled_domain_indices.items()}
        
        local_domain_weights = self.domain_weights.copy()
        domain_keys = list(local_domain_weights.keys())
        
        if not domain_keys:
            return iter([])

        domain_p = np.array(list(local_domain_weights.values()))
        
        while len(all_indices_stratified) < self.total_size:
            # Normalize probabilities in each step to handle exhausted domains
            current_total_p = domain_p.sum()
            if current_total_p == 0: break
            
            normalized_p = domain_p / current_total_p
            
            # Choose a domain based on the current weights
            chosen_domain_idx = np.random.choice(len(domain_keys), p=normalized_p)
            chosen_domain = domain_keys[chosen_domain_idx]

            try:
                # Try to get the next user index from the chosen domain
                user_idx = next(domain_iters[chosen_domain])
                all_indices_stratified.append(user_idx)
            except StopIteration:
                # This domain is exhausted. Set its probability to 0 for future draws
                # and continue to the next iteration of the while loop to choose another domain.
                domain_p[chosen_domain_idx] = 0
                continue

        # Shuffle the final list of indices to ensure randomness across batches
        np.random.shuffle(all_indices_stratified)
        return iter(all_indices_stratified)

    def __len__(self):
        return self.total_size

class MoerecCollator:
    def __init__(self, maxlen):
        self.maxlen = maxlen

    def __call__(self, batch):
        # --- DYNAMIC PADDING LOGIC ---
        uids = [item['uid'] for item in batch]
        seqs = [item['seq'] for item in batch]
        rating_seqs = [item['rating_seq'] for item in batch]
        poss = [item['pos'] for item in batch]
        negs = [item['neg'] for item in batch]
        domain_ids = [item['domain_id'] for item in batch]

        batch_maxlen = max(len(s) for s in seqs if len(s) > 0) if any(len(s) > 0 for s in seqs) else 0
        if batch_maxlen == 0:
            # Handle empty batch case
            return None, None, None, None, None, None

        # Create zero-filled numpy arrays for the padded batch
        padded_seqs = np.zeros([len(seqs), batch_maxlen], dtype=np.int32)
        padded_rating_seqs = np.zeros([len(seqs), batch_maxlen], dtype=np.int32)
        padded_poss = np.zeros([len(seqs), batch_maxlen], dtype=np.int32)
        padded_negs = np.zeros([len(seqs), batch_maxlen], dtype=np.int32)

        # Fill the arrays, right-aligning the sequences (padding at the beginning)
        for i, (seq, rating_seq, pos, neg) in enumerate(zip(seqs, rating_seqs, poss, negs)):
            seq_len = len(seq)
            if seq_len > 0:
                padded_seqs[i, -seq_len:] = seq
                padded_rating_seqs[i, -seq_len:] = rating_seq
                padded_poss[i, -seq_len:] = pos
                padded_negs[i, -seq_len:] = neg
        
        return (
            torch.LongTensor(uids),
            torch.LongTensor(padded_seqs),
            torch.LongTensor(padded_rating_seqs),
            torch.LongTensor(padded_poss),
            torch.LongTensor(padded_negs),
            torch.LongTensor(domain_ids)
        )

# --- End MoE Integration ---
