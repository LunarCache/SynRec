#!/usr/bin/env python3
"""
通用数据集处理脚本，生成 userid itemid rating 格式的txt文件
支持: Beauty, Games (Amazon数据集), ML-100K, ML-1M (MovieLens数据集)
"""

import pandas as pd
import gzip
import json
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm

def load_ratings_df(data_folder, csv_file, dataset_name):
    """加载评分数据"""
    file_path = data_folder / csv_file
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    if dataset_name == 'ml-100k':
        # ml-100k有标题行
        df = pd.read_csv(file_path)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
    elif dataset_name == 'ml-1m':
        # ml-1m使用::分隔符，没有标题行
        df = pd.read_csv(file_path, sep='::', header=None, engine='python')
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
    else:
        # Amazon数据集没有标题行
        df = pd.read_csv(file_path, header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
    
    print(f"原始评分数据大小: {len(df)}")
    return df

def load_meta_dict(data_folder, meta_file, dataset_name):
    """加载元数据"""
    file_path = data_folder / meta_file
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    meta_dict = {}
    
    if dataset_name == 'ml-100k':
        # ml-100k的元数据是CSV格式
        import re
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        for row in df.itertuples():
            title = row[2][:-7]  # remove year (optional)
            year = row[2][-7:]
            
            title = re.sub(r'\(.*?\)', '', title).strip()
            # the rest articles and parentheses are not considered here
            if any(', '+x in title.lower()[-5:] for x in ['a', 'an', 'the']):
                title_pre = title.split(', ')[:-1]
                title_post = title.split(', ')[-1]
                title_pre = ', '.join(title_pre)
                title = title_post + ' ' + title_pre
            
            meta_dict[row[1]] = title + year
    elif dataset_name == 'ml-1m':
        # ml-1m的元数据是dat格式，使用::分隔符
        import re
        df = pd.read_csv(file_path, sep='::', header=None, engine='python', encoding="ISO-8859-1")
        df.columns = ['movieId', 'title', 'genres']
        for row in df.itertuples():
            title = row[2][:-7]  # remove year (optional)
            year = row[2][-7:]
            
            title = re.sub(r'\(.*?\)', '', title).strip()
            # the rest articles and parentheses are not considered here
            if any(', '+x in title.lower()[-5:] for x in ['a', 'an', 'the']):
                title_pre = title.split(', ')[:-1]
                title_post = title.split(', ')[-1]
                title_pre = ', '.join(title_pre)
                title = title_post + ' ' + title_pre
            
            meta_dict[row[1]] = title + year
    else:
        # Amazon数据集的元数据是gzip压缩的JSON格式
        with gzip.open(file_path, 'rb') as f:
            for line in f:
                try:
                    item = eval(line)
                    if 'title' in item and len(item['title']) > 0:
                        meta_dict[item['asin'].strip()] = item['title'].strip()
                except:
                    continue
    
    print(f"元数据字典大小: {len(meta_dict)}")
    return meta_dict

def filter_by_meta(df, meta_dict):
    """步骤1: 筛选有元数据的项目"""
    print("步骤1: 筛选有元数据的项目...")
    df_filtered = df[df['sid'].isin(meta_dict)]
    print(f"筛选有元数据的项目后大小: {len(df_filtered)}")
    return df_filtered

def filter_min_interactions(df, min_uc=5, min_sc=5):
    """步骤2: 确保每个用户最少评分次数>=5，每个物品最少被评分次数>=5"""
    print("步骤2: 过滤最少交互次数...")
    print(f"要求: 每个用户最少评分{min_uc}次，每个物品最少被评分{min_sc}次")
    
    # 迭代过滤，直到满足条件
    iteration = 0
    while True:
        iteration += 1
        print(f"  迭代 {iteration}:")
        
        # 计算当前的用户和物品交互次数
        user_counts = df.groupby('uid').size()
        item_counts = df.groupby('sid').size()
        
        print(f"    当前数据大小: {len(df)}")
        print(f"    用户数: {len(user_counts)}, 物品数: {len(item_counts)}")
        
        # 找出满足条件的用户和物品
        valid_users = user_counts[user_counts >= min_uc].index
        valid_items = item_counts[item_counts >= min_sc].index
        
        # 过滤数据
        df_new = df[df['uid'].isin(valid_users) & df['sid'].isin(valid_items)]
        
        # 检查是否收敛
        if len(df_new) == len(df):
            print(f"    收敛！最终数据大小: {len(df_new)}")
            break
            
        df = df_new
        print(f"    过滤后数据大小: {len(df)}")
    
    return df

def remap_ids(df):
    """步骤3: 将用户id和物品id重新映射为连续整数(从1开始)"""
    print("步骤3: 重新映射用户和物品ID...")
    
    # 创建映射
    unique_users = sorted(df['uid'].unique())
    unique_items = sorted(df['sid'].unique())
    
    user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users, 1)}
    item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items, 1)}
    
    # 应用映射
    df_mapped = df.copy()
    df_mapped['uid'] = df_mapped['uid'].map(user_map)
    df_mapped['sid'] = df_mapped['sid'].map(item_map)
    
    print(f"映射后: {len(user_map)}个用户, {len(item_map)}个物品")
    return df_mapped, user_map, item_map

def sort_by_timestamp(df):
    """步骤4: 按时间顺序排列每个用户的交互物品"""
    print("步骤4: 按时间戳排序...")
    
    # 按用户和时间戳排序
    df_sorted = df.sort_values(['uid', 'timestamp', 'sid']).reset_index(drop=True)
    
    return df_sorted

def save_to_txt(df, output_path):
    """保存为txt文件并生成统计信息JSON"""
    print(f"保存到文件: {output_path}")
    
    # 只保留 userid itemid rating 三列
    output_df = df[['uid', 'sid', 'rating']].copy()
    # # 确保rating为整数（无小数点）
    # output_df['rating'] = output_df['rating'].astype(int)
    
    # 保存为txt文件，用空格分隔
    output_df.to_csv(output_path, sep=' ', header=False, index=False)
    
    print(f"保存完成！文件包含 {len(output_df)} 条记录")
    
    # 计算详细统计信息
    num_users = output_df['uid'].nunique()
    num_items = output_df['sid'].nunique()
    num_interactions = len(output_df)
    
    # 计算稀疏度 (1 - 交互数量 / (用户数 * 物品数))
    sparsity = 1 - (num_interactions / (num_users * num_items))
    
    # 计算序列平均长度 (每个用户的平均交互次数)
    user_interaction_counts = output_df.groupby('uid').size()
    avg_sequence_length = user_interaction_counts.mean()
    
    # 打印详细统计信息
    print("\n" + "="*50)
    print("数据集统计信息:")
    print("="*50)
    print(f"用户数量: {num_users:,}")
    print(f"物品数量: {num_items:,}")
    print(f"交互数量: {num_interactions:,}")
    print(f"稀疏度: {sparsity:.6f} ({sparsity*100:.4f}%)")
    print(f"序列平均长度: {avg_sequence_length:.2f}")
    print(f"评分范围: {output_df['rating'].min()} - {output_df['rating'].max()}")
    
    # 额外的统计信息
    print("\n" + "-"*30)
    print("序列长度分布:")
    print("-"*30)
    print(f"最短序列长度: {user_interaction_counts.min()}")
    print(f"最长序列长度: {user_interaction_counts.max()}")
    print(f"序列长度中位数: {user_interaction_counts.median():.1f}")
    print(f"序列长度标准差: {user_interaction_counts.std():.2f}")
    
    print("\n" + "-"*30)
    print("物品流行度分布:")
    print("-"*30)
    item_interaction_counts = output_df.groupby('sid').size()
    print(f"最少被评分次数: {item_interaction_counts.min()}")
    print(f"最多被评分次数: {item_interaction_counts.max()}")
    print(f"平均被评分次数: {item_interaction_counts.mean():.2f}")
    print(f"被评分次数中位数: {item_interaction_counts.median():.1f}")
    print("="*50)
    
    # 创建统计信息字典
    stats = {
        "dataset_info": {
            "name": output_path.stem.replace('_5_5', ''),
            "processed_time": pd.Timestamp.now().isoformat(),
            "output_file": str(output_path)
        },
        "basic_stats": {
            "num_users": int(num_users),
            "num_items": int(num_items),
            "num_interactions": int(num_interactions),
            "sparsity": float(sparsity),
            "sparsity_percent": float(sparsity * 100),
            "avg_sequence_length": float(avg_sequence_length),
            "rating_range": {
                "min": int(output_df['rating'].min()),
                "max": int(output_df['rating'].max())
            }
        },
        "sequence_length_distribution": {
            "min": int(user_interaction_counts.min()),
            "max": int(user_interaction_counts.max()),
            "median": float(user_interaction_counts.median()),
            "std": float(user_interaction_counts.std())
        },
        "item_popularity_distribution": {
            "min": int(item_interaction_counts.min()),
            "max": int(item_interaction_counts.max()),
            "mean": float(item_interaction_counts.mean()),
            "median": float(item_interaction_counts.median())
        }
    }
    
    # 保存统计信息到JSON文件
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计信息已保存到: {json_path}")

def main():
    """主函数"""
    import sys
    import re
    
    # 从命令行参数获取数据集名称，默认为beauty
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'beauty'
    
    # 设置路径
    data_folder = Path(f'new_data/{dataset_name}')
    output_path = Path(f'data/{dataset_name}_5_5.txt')
    
    # 设置文件名
    if dataset_name == 'beauty':
        csv_file = 'beauty.csv'
        meta_file = 'beauty_meta.json.gz'
    elif dataset_name == 'games':
        csv_file = 'games.csv'
        meta_file = 'games_meta.json.gz'
    elif dataset_name == 'ml-100k':
        csv_file = 'ratings.csv'
        meta_file = 'movies.csv'
    elif dataset_name == 'ml-1m':
        csv_file = 'ratings.dat'
        meta_file = 'movies.dat'
    else:
        print(f"不支持的数据集: {dataset_name}")
        print("支持的数据集: beauty, games, ml-100k, ml-1m")
        return 1
    
    try:
        print(f"处理 {dataset_name} 数据集...")
        print(f"数据文件夹: {data_folder}")
        print(f"输出文件: {output_path}")
        print()
        
        # 加载原始数据
        df = load_ratings_df(data_folder, csv_file, dataset_name)
        meta_dict = load_meta_dict(data_folder, meta_file, dataset_name)
        
        # 步骤1: 筛选有元数据的项目
        df = filter_by_meta(df, meta_dict)
        
        # 步骤2: 过滤最少交互次数
        df = filter_min_interactions(df, min_uc=5, min_sc=5)
        
        # 步骤3: 重新映射ID
        df, user_map, item_map = remap_ids(df)
        
        # 步骤4: 按时间排序
        df = sort_by_timestamp(df)
        
        # 保存结果
        save_to_txt(df, output_path)
        
        print("\n处理完成！")
        print(f"输出文件: {output_path}")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
