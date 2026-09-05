import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn

# 构建异构图
def build_heterograph(users, items, attributes, topics):
    hg = dgl.heterograph({
        ('user', 'buys', 'item'): (users, items),
        ('item', 'has', 'attribute'): (items, attributes),
        ('item', 'related_to', 'topic'): (items, topics),
        ('topic', 'influences', 'item'): (topics, items)
    })
    return hg

def extract_paths(hg, user_id):
    intra_paths = []
    inter_paths = []
    coref_paths = []

    # 提取内属性推理路径
    for path in hg.find_paths(user_id, 'item', 'item', ['buys', 'has', 'related_to', 'has', 'buys']):
        intra_paths.append(path)

    # 提取跨属性推理路径
    for path in hg.find_paths(user_id, 'item', 'item', ['buys', 'has', 'related_to', 'related_to', 'has', 'buys']):
        inter_paths.append(path)

    # 提取共指推理路径
    for path in hg.find_paths(user_id, 'item', 'item', ['buys', 'has', 'related_to', 'related_to', 'related_to', 'has', 'buys']):
        coref_paths.append(path)

    return intra_paths, inter_paths, coref_paths

class EdgeAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EdgeAttention, self).__init__()
        self.attention = nn.Linear(in_dim, out_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, node_embeddings, relation_type):
        attention_scores = self.attention(node_embeddings)
        return self.activation(attention_scores)

def calculate_path_weight(edges, attention_module):
    weights = []
    for edge in edges:
        node_embeddings = torch.cat([edge.src_embeddings, edge.dst_embeddings])
        weight = attention_module(node_embeddings, edge.relation_type)
        weights.append(weight)
    return torch.mean(torch.tensor(weights))


class RGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels):
        super(RGCN, self).__init__()
        self.conv = dglnn.RelGraphConv(in_dim, out_dim, num_rels, activation=nn.ReLU())

    def forward(self, hg, node_features, edge_types):
        return self.conv(hg, node_features, edge_types)

class PathClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(PathClassifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, global_user_embedding, path_representation, attribute_distance):
        features = torch.cat([global_user_embedding, path_representation, attribute_distance])
        return self.mlp(features)