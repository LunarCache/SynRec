import sys
import os
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# 该版本为没有使用动态填充的版本


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
    if fname == 'ml-100k' or fname == 'beauty' or fname == 'games':
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

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, user_to_domain, usernum, itemnum, domain_to_item_range] = copy.deepcopy(dataset)

    # --- MoE Integration: Per-domain evaluation ---
    domain_metrics = defaultdict(lambda: {'NDCG': 0.0, 'HT': 0.0, 'count': 0})
    # --- End MoE Integration ---

    users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        domain_id = user_to_domain[u]
        if args.use_domain_sampling_for_evaluation:
            item_range = domain_to_item_range[domain_id]
            if item_range is None: continue  # Skip if no items in this domain
            for _ in range(100):
                # Sample negative items from the domain's item range
                # Ensure the sampled item is not in the rated set
                t = np.random.randint(item_range[0], item_range[1] + 1)
                while t in rated: t = np.random.randint(item_range[0], item_range[1] + 1)
                item_idx.append(t)
        else:
            for _ in range(100):
                # Sample negative items globally
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [domain_id]]])
        # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        # --- MoE Integration: Per-domain evaluation ---
        # domain_id = user_to_domain[u]
        domain_metrics[domain_id]['count'] += 1
        if rank < 10:
            domain_metrics[domain_id]['NDCG'] += 1 / np.log2(rank + 2)
            domain_metrics[domain_id]['HT'] += 1
        
        if sum(d['count'] for d in domain_metrics.values()) % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    
    # Calculate and return results
    results = {}
    total_ndcg = 0.0
    total_ht = 0.0
    total_users = 0
    
    for domain_id, metrics in sorted(domain_metrics.items()):
        count = metrics['count']
        if count > 0:
            ndcg = metrics['NDCG'] / count
            ht = metrics['HT'] / count
            results[f'domain_{domain_id}_NDCG'] = ndcg
            results[f'domain_{domain_id}_HT'] = ht
            total_ndcg += metrics['NDCG']
            total_ht += metrics['HT']
            total_users += count

    results['overall_NDCG'] = total_ndcg / total_users if total_users > 0 else 0
    results['overall_HT'] = total_ht / total_users if total_users > 0 else 0
    
    return results


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, user_to_domain, usernum, itemnum, domain_to_item_range] = copy.deepcopy(dataset)

    # --- MoE Integration: Per-domain evaluation ---
    domain_metrics = defaultdict(lambda: {'NDCG': 0.0, 'HT': 0.0, 'count': 0})
    # --- End MoE Integration ---

    users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        domain_id = user_to_domain[u]
        if args.use_domain_sampling_for_evaluation:
            item_range = domain_to_item_range[domain_id]
            if item_range is None: continue  # Skip if no items in this domain
        
            for _ in range(100):
                t = np.random.randint(item_range[0], item_range[1] + 1)
                while t in rated: t = np.random.randint(item_range[0], item_range[1] + 1)
                item_idx.append(t)
        else:
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [domain_id]]])
        # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        # --- MoE Integration: Per-domain evaluation ---
        domain_metrics[domain_id]['count'] += 1
        if rank < 10:
            domain_metrics[domain_id]['NDCG'] += 1 / np.log2(rank + 2)
            domain_metrics[domain_id]['HT'] += 1
        
        if sum(d['count'] for d in domain_metrics.values()) % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # Calculate and return results
    results = {}
    total_ndcg = 0.0
    total_ht = 0.0
    total_users = 0
    
    for domain_id, metrics in sorted(domain_metrics.items()):
        count = metrics['count']
        if count > 0:
            ndcg = metrics['NDCG'] / count
            ht = metrics['HT'] / count
            results[f'domain_{domain_id}_NDCG'] = ndcg
            results[f'domain_{domain_id}_HT'] = ht
            total_ndcg += metrics['NDCG']
            total_ht += metrics['HT']
            total_users += count

    results['overall_NDCG'] = total_ndcg / total_users if total_users > 0 else 0
    results['overall_HT'] = total_ht / total_users if total_users > 0 else 0
    
    return results

def partition_multi_domain(fnames):
    """
    Loads and partitions multiple datasets from the SASRec.pytorch/python/data/ directory.
    - Handles integer IDs by offsetting them to ensure global uniqueness.
    - Assigns a domain_id to each user.
    - fnames: list of dataset names, e.g., ['beauty', 'games', 'ml-100k']
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
    
    print("Starting multi-domain data partitioning (Offset ID Strategy)...")

    for domain_id, fname in enumerate(fnames):
        print(f"Processing domain {domain_id}: {fname}")
        
        file_path = os.path.join('data', f"{fname}.txt")
        if not os.path.exists(file_path):
            print(f"Warning: Data file not found at {file_path}. Skipping.")
            continue

        # Determine delimiter
        delimiter = ' '
        if fname in ['ml-100k', 'beauty', 'games']:
            delimiter = ','

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
        domain_start_item = item_offset + 1
        domain_end_item = item_offset + local_itemnum
        domain_to_item_range[domain_id] = (domain_start_item, domain_end_item)

        # Second pass: load and process data with offsets
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(delimiter)
                # Apply offset to create global unique IDs
                u = int(parts[0]) + user_offset
                i = int(parts[1]) + item_offset
                
                usernum = max(u, usernum)
                itemnum = max(i, itemnum)
                User[u].append(i)
                user_to_domain[u] = domain_id
        
        print(f"  Domain '{fname}' -> Local Users: {local_usernum}, Local Items: {local_itemnum}")
        print(f"  Applying Offsets -> User: +{user_offset}, Item: +{item_offset}")
        print(f"  Domain Item Range: [{domain_start_item}, {domain_end_item}]")
        
        # Update offsets for the next domain
        user_offset += local_usernum
        item_offset += local_itemnum
        print(f"  Updated Global Users: {usernum}, Global Items: {itemnum}")

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
class MoerecStyleSampler(object):
    def __init__(self, user_train, user_to_domain, usernum, itemnum, batch_size, maxlen, args, domain_to_item_range):
        self.user_train = user_train
        self.user_to_domain = user_to_domain
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.args = args
        self.domain_to_item_range = domain_to_item_range

        # --- Stratified Sampling Implementation ---
        # 1. Group valid users by domain
        self.domain_to_users = defaultdict(list)
        for u, seq in user_train.items():
            if len(seq) > 1:
                self.domain_to_users[self.user_to_domain[u]].append(u)

        # 2. Calculate total number of valid users and batches
        self.total_valid_users = sum(len(users) for users in self.domain_to_users.values())
        self.num_batches = (self.total_valid_users - 1) // self.batch_size + 1
        
        # 3. Calculate domain weights for proportional sampling
        self.domain_weights = {d: len(u) / self.total_valid_users for d, u in self.domain_to_users.items()}
        
        print(f"Stratified Sampler initialized: {self.total_valid_users} users, {self.num_batches} batches.")
        print(f"Domain distribution: { {d: f'{w:.2%}' for d, w in self.domain_weights.items()} }")


    def _sample_one_user(self, uid):
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        nxt = self.user_train[uid][-1]
        idx = self.maxlen - 1
        
        domain_id = self.user_to_domain[uid]

        ts = set(self.user_train[uid])
        for i in reversed(self.user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                if self.args.use_domain_sampling:
                    item_range = self.domain_to_item_range[domain_id]
                    neg[idx] = random_neq(item_range[0], item_range[1] + 1, ts)
                else:
                    neg[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        return (uid, seq, pos, neg, domain_id)

    def __iter__(self):
        # 1. Shuffle users within each domain
        shuffled_domain_users = {d: np.random.permutation(u) for d, u in self.domain_to_users.items()}
        
        # 2. Create a flat list of all users, maintaining stratified order
        all_users_stratified = []
        
        # Create iterators for each domain's shuffled list
        domain_iters = {d: iter(u) for d, u in shuffled_domain_users.items()}
        
        # Proportional sampling to build the final epoch list
        # --- FIX: Use a local copy of weights to avoid modifying the class attribute ---
        local_domain_weights = self.domain_weights.copy()
        domain_keys = list(local_domain_weights.keys())
        domain_p = list(local_domain_weights.values())
        
        for _ in range(self.total_valid_users):
            chosen_domain = np.random.choice(domain_keys, p=domain_p)
            try:
                user = next(domain_iters[chosen_domain])
                all_users_stratified.append(user)
            except StopIteration:
                # If one domain runs out, remove it and re-normalize weights
                del domain_iters[chosen_domain]
                del local_domain_weights[chosen_domain]
                if not local_domain_weights: break
                
                domain_keys = list(local_domain_weights.keys())
                total_p = sum(local_domain_weights.values())
                domain_p = [p / total_p for p in local_domain_weights.values()]


        # 3. Yield batches from the stratified list
        for i in range(0, len(all_users_stratified), self.batch_size):
            user_batch_ids = all_users_stratified[i:i+self.batch_size]
            
            one_batch = [self._sample_one_user(uid) for uid in user_batch_ids]
            
            if one_batch:
                yield zip(*one_batch)

    def __len__(self):
        return self.num_batches
    
    def close(self):
        pass
# --- End MoE Integration ---
