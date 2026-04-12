import pandas as pd
import os
import sys
from pathlib import Path

def load_data(csv_path):
    # Load raw csv without header: uid, sid, rating, timestamp
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['uid', 'sid', 'rating', 'timestamp']
    return df

def filter_k_core(df, k=5):
    # Iterative k-core filtering
    while True:
        user_counts = df.groupby('uid').size()
        item_counts = df.groupby('sid').size()
        
        valid_users = user_counts[user_counts >= k].index
        valid_items = item_counts[item_counts >= k].index
        
        df_new = df[df['uid'].isin(valid_users) & df['sid'].isin(valid_items)]
        
        if len(df_new) == len(df):
            break
        df = df_new
    return df

def main():
    print("Processing Linked Data for Beauty and Games...")
    
    # 1. Load Data
    beauty_path = 'new_data/beauty/beauty.csv'
    games_path = 'new_data/games/games.csv'
    
    df_beauty = load_data(beauty_path)
    df_games = load_data(games_path)
    
    print(f"Original Beauty: {len(df_beauty)}")
    print(f"Original Games: {len(df_games)}")
    
    # 2. Independent K-Core Filtering (to match baseline logic)
    print("Filtering Beauty (5-core)...")
    df_beauty = filter_k_core(df_beauty, 5)
    print("Filtering Games (5-core)...")
    df_games = filter_k_core(df_games, 5)
    
    print(f"Filtered Beauty: {len(df_beauty)}")
    print(f"Filtered Games: {len(df_games)}")
    
    # 3. Global Mapping
    print("Building Global ID Maps...")
    
    # Users: Shared space
    all_users = pd.concat([df_beauty['uid'], df_games['uid']]).unique()
    user_map = {u: i+1 for i, u in enumerate(all_users)} # 1-based
    
    # Items: Disjoint space
    # Beauty items first
    beauty_items = df_beauty['sid'].unique()
    item_map = {i: idx+1 for idx, i in enumerate(beauty_items)}
    beauty_max_id = len(beauty_items)
    
    # Games items next
    games_items = df_games['sid'].unique()
    # Check for overlap in item IDs (should be none for ASINs usually, but good to be safe)
    # Actually Amazon ASINs might overlap if same item is in both categories.
    # For SynRec disjoint assumption, we force them to be separate IDs even if same ASIN?
    # Or do we share Item ID too? 
    # Manuscript says: "Disjoint item set". So we should force disjointness.
    # But if an item is literally the same, sharing embedding makes sense.
    # Let's stick to strict disjointness for Items to match 'SynRec' protocol, only sharing Users.
    
    # Force disjoint item IDs for Games
    for idx, item in enumerate(games_items):
        item_map[f"GAMES_{item}"] = beauty_max_id + idx + 1
        
    print(f"Total Users: {len(user_map)}")
    print(f"Total Items: {len(item_map)}")
    
    # 4. Remap and Save
    def process_and_save(df, name, is_games=False):
        # Map Users
        df['uid'] = df['uid'].map(user_map)
        
        # Map Items
        if is_games:
            # Add prefix to ensure lookup hits the Games part of item_map
            df['sid'] = df['sid'].apply(lambda x: f"GAMES_{x}").map(item_map)
        else:
            df['sid'] = df['sid'].map(item_map)
            
        # Drop rows with unmapped IDs (shouldn't happen if map is built from these dfs)
        df = df.dropna()
        df['uid'] = df['uid'].astype(int)
        df['sid'] = df['sid'].astype(int)
        df['rating'] = df['rating'].astype(int)
        
        # Sort
        df = df.sort_values(['uid', 'timestamp'])
        
        # Save
        out_path = f'data/{name}_linked.txt'
        df[['uid', 'sid', 'rating']].to_csv(out_path, sep=' ', header=False, index=False)
        print(f"Saved {name} to {out_path}")

    process_and_save(df_beauty, 'beauty')
    process_and_save(df_games, 'games', is_games=True)
    
    print("Done.")

if __name__ == "__main__":
    main()
