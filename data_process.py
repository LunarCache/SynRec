import gzip
import os
import json
import csv
from collections import defaultdict


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def process_movielens_dataset(dataset_name, input_path, output_path):
    """处理MovieLens数据集，输入格式为UserID::MovieID::Rating::Timestamp"""
    print(f"正在处理MovieLens数据集: {dataset_name}")
    
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    # 第一遍扫描，统计用户和物品的交互次数
    print("第一遍扫描，统计交互次数...")
    with open(input_path, 'r', encoding='latin-1') as f:
        for line_data in f:
            line += 1
            parts = line_data.strip().split('::')
            if len(parts) >= 4:
                user_id, item_id, rating, timestamp = parts[:4]
                countU[user_id] += 1
                countP[item_id] += 1
    
    print(f"扫描了 {line} 条记录")

    # 第二遍扫描，过滤并处理数据
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    
    print("第二遍扫描，处理有效数据...")
    with open(input_path, 'r', encoding='latin-1') as f:
        for line_data in f:
            parts = line_data.strip().split('::')
            if len(parts) >= 4:
                user_id, item_id, rating, timestamp = parts[:4]
                
                # 过滤掉交互次数少于5的用户和物品
                if countU[user_id] < 5 or countP[item_id] < 5:
                    continue

                if user_id in usermap:
                    userid = usermap[user_id]
                else:
                    usernum += 1
                    userid = usernum
                    usermap[user_id] = userid
                    User[userid] = []
                    
                if item_id in itemmap:
                    itemid = itemmap[item_id]
                else:
                    itemnum += 1
                    itemid = itemnum
                    itemmap[item_id] = itemid
                    
                User[userid].append([int(timestamp), itemid, float(rating)])

    # 按时间排序
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(f'过滤后用户数: {usernum}')
    print(f'过滤后物品数: {itemnum}')

    # 输出为user item rating格式
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for user in User.keys():
            for interaction in User[user]:
                timestamp, item, rating = interaction
                f.write(f'{user} {item} {rating}\n')
    
    print(f"数据已保存到: {output_path}")
    
    # 计算统计信息
    total_interactions = sum(len(User[user]) for user in User.keys())
    avg_sequence_length = total_interactions / usernum if usernum > 0 else 0
    sparsity = 1 - (total_interactions / (usernum * itemnum)) if usernum > 0 and itemnum > 0 else 0
    
    stats = {
        'num_users': usernum,
        'num_items': itemnum,
        'num_interactions': total_interactions,
        'avg_sequence_length': round(avg_sequence_length, 2),
        'sparsity': round(sparsity, 6)
    }
    
    return usernum, itemnum, stats
    """处理单个数据集，输出格式为user item rating"""
    print(f"正在处理数据集: {dataset_name}")
    
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    # 第一遍扫描，统计用户和物品的交互次数
    print("第一遍扫描，统计交互次数...")
    for l in parse(input_path):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        countU[rev] += 1
        countP[asin] += 1
    
    print(f"扫描了 {line} 条记录")

    # 第二遍扫描，过滤并处理数据
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    
    print("第二遍扫描，处理有效数据...")
    for l in parse(input_path):
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        rating = l['overall']
        
        # 过滤掉交互次数少于5的用户和物品
        if countU[rev] < 5 or countP[asin] < 5:
            continue

        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
            
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
            
        User[userid].append([time, itemid, rating])

    # 按时间排序
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(f'过滤后用户数: {usernum}')
    print(f'过滤后物品数: {itemnum}')

    # 输出为user item rating格式
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for user in User.keys():
            for interaction in User[user]:
                time, item, rating = interaction
                f.write(f'{user} {item} {rating}\n')
    
    print(f"数据已保存到: {output_path}")
    
    # 计算统计信息
    total_interactions = sum(len(User[user]) for user in User.keys())
    avg_sequence_length = total_interactions / usernum if usernum > 0 else 0
    sparsity = 1 - (total_interactions / (usernum * itemnum)) if usernum > 0 and itemnum > 0 else 0
    
    stats = {
        'num_users': usernum,
        'num_items': itemnum,
        'num_interactions': total_interactions,
        'avg_sequence_length': round(avg_sequence_length, 2),
        'sparsity': round(sparsity, 6)
    }
    
    return usernum, itemnum, stats


def process_csv_dataset(dataset_name, input_path, output_path):
    """处理CSV格式数据集，输入格式为user_id,item_id,rating,timestamp"""
    print(f"正在处理CSV数据集: {dataset_name}")
    
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    # 第一遍扫描，统计用户和物品的交互次数
    print("第一遍扫描，统计交互次数...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_data in f:
            line += 1
            parts = line_data.strip().split(',')
            if len(parts) >= 4:
                user_id, item_id, rating, timestamp = parts[:4]
                countU[user_id] += 1
                countP[item_id] += 1
    
    print(f"扫描了 {line} 条记录")

    # 第二遍扫描，过滤并处理数据
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    
    print("第二遍扫描，处理有效数据...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_data in f:
            parts = line_data.strip().split(',')
            if len(parts) >= 4:
                user_id, item_id, rating, timestamp = parts[:4]
                
                # 过滤掉交互次数少于5的用户和物品
                if countU[user_id] < 5 or countP[item_id] < 5:
                    continue

                if user_id in usermap:
                    userid = usermap[user_id]
                else:
                    usernum += 1
                    userid = usernum
                    usermap[user_id] = userid
                    User[userid] = []
                    
                if item_id in itemmap:
                    itemid = itemmap[item_id]
                else:
                    itemnum += 1
                    itemid = itemnum
                    itemmap[item_id] = itemid
                    
                User[userid].append([int(timestamp), itemid, float(rating)])

    # 按时间排序
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(f'过滤后用户数: {usernum}')
    print(f'过滤后物品数: {itemnum}')

    # 输出为user item rating格式
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for user in User.keys():
            for interaction in User[user]:
                timestamp, item, rating = interaction
                f.write(f'{user} {item} {rating}\n')
    
    print(f"数据已保存到: {output_path}")
    
    # 计算统计信息
    total_interactions = sum(len(User[user]) for user in User.keys())
    avg_sequence_length = total_interactions / usernum if usernum > 0 else 0
    sparsity = 1 - (total_interactions / (usernum * itemnum)) if usernum > 0 and itemnum > 0 else 0
    
    stats = {
        'num_users': usernum,
        'num_items': itemnum,
        'num_interactions': total_interactions,
        'avg_sequence_length': round(avg_sequence_length, 2),
        'sparsity': round(sparsity, 6)
    }
    
    return usernum, itemnum, stats


def process_dataset(dataset_name, input_path, output_path):
    """处理单个数据集，输出格式为user item rating"""
    print(f"正在处理数据集: {dataset_name}")
    
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    # 第一遍扫描，统计用户和物品的交互次数
    print("第一遍扫描，统计交互次数...")
    for l in parse(input_path):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        countU[rev] += 1
        countP[asin] += 1
    
    print(f"扫描了 {line} 条记录")

    # 第二遍扫描，过滤并处理数据
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    
    print("第二遍扫描，处理有效数据...")
    for l in parse(input_path):
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        rating = l['overall']
        
        # 过滤掉交互次数少于5的用户和物品
        if countU[rev] < 5 or countP[asin] < 5:
            continue

        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
            
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
            
        User[userid].append([time, itemid, rating])

    # 按时间排序
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(f'过滤后用户数: {usernum}')
    print(f'过滤后物品数: {itemnum}')

    # 输出为user item rating格式
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for user in User.keys():
            for interaction in User[user]:
                time, item, rating = interaction
                f.write(f'{user} {item} {rating}\n')
    
    print(f"数据已保存到: {output_path}")
    
    # 计算统计信息
    total_interactions = sum(len(User[user]) for user in User.keys())
    avg_sequence_length = total_interactions / usernum if usernum > 0 else 0
    sparsity = 1 - (total_interactions / (usernum * itemnum)) if usernum > 0 and itemnum > 0 else 0
    
    stats = {
        'num_users': usernum,
        'num_items': itemnum,
        'num_interactions': total_interactions,
        'avg_sequence_length': round(avg_sequence_length, 2),
        'sparsity': round(sparsity, 6)
    }
    
    return usernum, itemnum, stats


# 存储所有数据集的统计信息
dataset_stats = {}

# 处理beauty数据集
beauty_input = 'new_data_copy/beauty/reviews_Beauty.json.gz'
beauty_output = 'new_data_copy/beauty/beauty_rated.txt'
beauty_users, beauty_items, beauty_stats = process_dataset('Beauty', beauty_input, beauty_output)
dataset_stats['beauty'] = beauty_stats

print("\n" + "="*50 + "\n")

# 处理games数据集  
games_input = 'new_data_copy/games/reviews_Video_Games.json.gz'
games_output = 'new_data_copy/games/games_rated.txt'
games_users, games_items, games_stats = process_dataset('Video_Games', games_input, games_output)
dataset_stats['games'] = games_stats

print("\n" + "="*50 + "\n")

# 处理ml-1m数据集
ml1m_input = 'new_data_copy/ml-1m/ratings.dat'
ml1m_output = 'new_data_copy/ml-1m/ml-1m_rated.txt'
ml1m_users, ml1m_items, ml1m_stats = process_movielens_dataset('MovieLens-1M', ml1m_input, ml1m_output)
dataset_stats['ml-1m'] = ml1m_stats

# 保存统计信息到JSON文件
with open('dataset_statistics.json', 'w', encoding='utf-8') as f:
    json.dump(dataset_stats, f, indent=2, ensure_ascii=False)

print("\n所有数据集处理完成!")
print(f"Beauty数据集 - 用户数: {beauty_users}, 物品数: {beauty_items}")
print(f"Games数据集 - 用户数: {games_users}, 物品数: {games_items}")
print(f"MovieLens-1M数据集 - 用户数: {ml1m_users}, 物品数: {ml1m_items}")
print("\n统计信息已保存到 dataset_statistics.json")

print("\n=== 数据集统计信息 ===")
for dataset, stats in dataset_stats.items():
    print(f"\n{dataset.upper()}数据集:")
    print(f"  用户数量: {stats['num_users']:,}")
    print(f"  物品数量: {stats['num_items']:,}")
    print(f"  交互数量: {stats['num_interactions']:,}")
    print(f"  平均序列长度: {stats['avg_sequence_length']}")
    print(f"  稀疏度: {stats['sparsity']:.6f}")