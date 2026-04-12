import os
import sys
import pandas as pd
import numpy as np
import random
import subprocess
import re
from tqdm import tqdm

def write_and_run_inspeq():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    domains = ['tools_5_5', 'toys_5_5', 'baby_5_5']
    data_dir = os.path.join(script_dir, '../data')
    inspeq_master_dir = os.path.join(script_dir, 'INSPEQ/INSPEQ-master')
    inspeq_data_dir = os.path.join(inspeq_master_dir, 'data/')
    
    os.makedirs(inspeq_data_dir, exist_ok=True)
    
    # 1. Update sequential_data_list in utils.py
    utils_path = os.path.join(inspeq_master_dir, 'utils.py')
    with open(utils_path, 'r', encoding='utf-8') as f:
        utils_code = f.read()
    
    # Ensure our domains are in the list
    for domain in domains:
        if domain not in utils_code:
            # Simple patch: append to sequential_data_list
            utils_code = re.sub(
                r"sequential_data_list = \[(.*?)\]", 
                r"sequential_data_list = [\1, '" + domain + "']", 
                utils_code
            )
    with open(utils_path, 'w', encoding='utf-8') as f:
        f.write(utils_code)

    print("Updated utils.py with target domains.")

    # 2. Process data
    for domain in domains:
        txt_path = os.path.join(data_dir, f"{domain}.txt")
        if not os.path.exists(txt_path):
            print(f"Skipping {domain} (data missing)")
            continue
            
        print(f"Processing data for {domain}...")
        df = pd.read_csv(txt_path, sep=' ', header=None, names=['uid', 'iid', 'rating'])
        
        # Dense mapping for items (INSPEQ uses 1-indexed padding)
        unique_items = df['iid'].unique()
        item_map = {old: new for new, old in enumerate(unique_items, start=1)}
        num_items = len(item_map)
        
        unique_users = df['uid'].unique()
        user_map = {old: new for new, old in enumerate(unique_users, start=1)}
        
        df_mapped = df.copy()
        df_mapped['iid'] = df['iid'].map(item_map)
        df_mapped['uid'] = df['uid'].map(user_map)
        
        # Build sequences manually
        user_seqs = []
        user_samples = []
        
        grouped = df_mapped.groupby('uid')
        for uid, group in tqdm(grouped, desc=f"Building sequences {domain}"):
            seq = group['iid'].tolist()
            if len(seq) < 3:
                continue
                
            user_seqs.append((uid, seq))
            
            # Sample 99 negatives
            pos_set = set(seq)
            negatives = []
            while len(negatives) < 99:
                t = random.randint(1, num_items)
                if t not in pos_set:
                    negatives.append(t)
            user_samples.append((uid, negatives))
            
        # Write to txt correctly
        out_txt = os.path.join(inspeq_data_dir, f"{domain}.txt")
        out_sample = os.path.join(inspeq_data_dir, f"{domain}_sample.txt")
        
        with open(out_txt, 'w') as f:
            for uid, seq in user_seqs:
                seq_str = " ".join(map(str, seq))
                f.write(f"{uid} {seq_str}\n")
                
        with open(out_sample, 'w') as f:
            for uid, seq in user_samples:
                seq_str = " ".join(map(str, seq))
                f.write(f"{uid} {seq_str}\n")
                
        # 3. Run INSPEQ
        print(f"Running INSPEQ on {domain}...")
        # change working directory so INSPEQ finds its paths correctly
        os.chdir(inspeq_master_dir)
        cmd = [
            sys.executable, 'main.py',
            '--data_dir', 'data/',
            '--data_name', domain,
            '--epochs', '300', # Change if needed
            '--batch_size', '256',
            '--log_freq', '1',
        ]
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd)
        os.chdir(script_dir) # Go back to baselines/
        
if __name__ == '__main__':
    write_and_run_inspeq()