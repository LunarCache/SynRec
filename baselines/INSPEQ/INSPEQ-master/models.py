import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, paraGRU

class PathReasoningModule(nn.Module):
    def __init__(self, args):
        super(PathReasoningModule, self).__init__()
        self.args = args
        self.gcn_layers = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.num_gcn_layers)])
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.super_node = nn.Parameter(torch.randn(args.hidden_size))

    def forward(self, sequence_emb, paths):
        """
        sequence_emb: [batch_size, seq_length, hidden_size]
        paths: [batch_size, num_paths, path_length, hidden_size]
        """
        batch_size, seq_length, hidden_size = sequence_emb.size()
        num_paths, path_length = paths.size(1), paths.size(2)

        # Flatten paths for GCN
        paths = paths.view(-1, path_length, hidden_size)  # [batch_size * num_paths, path_length, hidden_size]
        path_embeddings = paths[:, 0, :]  # Initialize with the first node [batch_size * num_paths, hidden_size]

        for layer in self.gcn_layers:
            path_embeddings = layer(path_embeddings)
            path_embeddings = self.dropout(path_embeddings)
            path_embeddings = torch.relu(path_embeddings)

        # Max pooling over paths
        path_embeddings = path_embeddings.view(batch_size, num_paths, hidden_size)
        path_embeddings = torch.max(path_embeddings, dim=1)[0]  # [batch_size, hidden_size]

        # Add super node
        super_node_expanded = self.super_node.expand(batch_size, hidden_size)
        path_embeddings = path_embeddings + super_node_expanded

        return path_embeddings


class INSPEQModel(nn.Module):
    def __init__(self, args):
        super(INSPEQModel, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)
        self.PATR_GRU = paraGRU(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1,
                            bidirectional=False, dropout=args.hidden_dropout_prob)

        # 添加 MLP 对齐层
        self.align = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size)
        )

        # 添加路径推理模块
        self.path_reasoning = PathReasoningModule(args)

        self.apply(self.init_weights)

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def forward(self, input_ids, paths=None):
        # 短期偏好表示且缓解错误信息影响
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        sequence_output = item_encoded_layers[-1]

        # 长期偏好表示
        gru_output, _ = self.PATR_GRU(sequence_emb)

        combined_output = torch.cat((sequence_output, gru_output), dim=-1)
        aligned_output = self.align(combined_output)

        # 路径推理模块
        if paths is not None:
            path_embeddings = self.path_reasoning(sequence_emb, paths)
            aligned_output = aligned_output + path_embeddings.unsqueeze(1)

        return aligned_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf <https://github.com/pytorch/pytorch/pull/5617>
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()