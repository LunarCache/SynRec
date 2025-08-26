import numpy as np
import torch
# --- MoE Integration: Import the new SASRec-specific MoE FFN ---

from keys.c_moe import PointWiseFeedForward
from keys.temporal_rating_modules import TemporalEnhancedRatingModule

# --- End MoE Integration ---


class SynRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SynRec, self).__init__()
        self.args = args
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        
        # Enhanced Rating Module - uses optimized temporal-frequency encoder
        if getattr(args, 'use_rating_emb', False):
            rating_strategy = getattr(args, 'rating_strategy', 'temporal_fourier')  # default to temporal_fourier
            self.rating_strategy = rating_strategy  # Store for forward pass logic
            
            if rating_strategy == 'temporal_fourier':
                # Use the optimized temporal-frequency rating encoder
                self.enhanced_rating_module = TemporalEnhancedRatingModule(
                    args.hidden_units, 
                    rating_strategy=rating_strategy, 
                    dropout_rate=args.dropout_rate,
                    args=args
                )
            else:
                # Fallback for simple/legacy strategies
                self.rating_emb = torch.nn.Embedding(6, args.hidden_units, padding_idx=0)
        
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # Note: Removed context_attention as we now use simple mean pooling for efficiency

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate, args=args)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs, rating_seqs=None, domain_ids=None):
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        
        # Create position sequence on the fly, on the correct device
        position_ids = torch.arange(1, log_seqs.size(1) + 1, device=self.dev).unsqueeze(0)
        # Create a mask for non-zero log_seqs entries
        pos_mask = (log_seqs != 0).long()
        # Apply the mask to the position_ids
        poss = position_ids * pos_mask
        
        pos_embedding = self.pos_emb(poss)
        seqs += pos_embedding
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        total_moe_loss_dict = {'load_balancing': 0.0}
        total_viz_data = {} # Initialize viz_data dict

        for i in range(len(self.attention_layers)):
            # Standard Pre-norm attention with proper residual connection
            original_seqs = seqs  # Save original input for residual connection
            normalized_seqs = self.attention_layernorms[i](seqs)
            # Use normalized input for Q, K, V to ensure symmetry
            mha_outputs, _ = self.attention_layers[i](normalized_seqs.transpose(0, 1), 
                                                     normalized_seqs.transpose(0, 1), 
                                                     normalized_seqs.transpose(0, 1),
                                                     attn_mask=attention_mask)
            # Proper residual connection: original_input + sublayer_output
            seqs = original_seqs + mha_outputs.transpose(0, 1)

            # FFN part with proper residual connection
            original_seqs = seqs  # Save original input for residual connection
            normalized_seqs = self.forward_layernorms[i](seqs)
            
            # --- Enhanced Rating-Aware Gating ---
            rating_embedding = None
            rating_extra_info = {}
            if getattr(self.args, 'use_rating_emb', False) and rating_seqs is not None:
                rating_strategy = getattr(self, 'rating_strategy', 'temporal_fourier')
                
                if rating_strategy == 'temporal_fourier':
                    # Use optimized temporal-frequency rating encoder
                    enhanced_rating_repr, rating_extra_info = self.enhanced_rating_module(rating_seqs, domain_ids)
                    
                    # Add positional embedding if enabled (for compatibility)
                    if getattr(self.args, 'rating_pos_emb', False):
                        enhanced_rating_repr += pos_embedding
                    
                    rating_embedding = enhanced_rating_repr
                    
                    # Store frequency analysis info for visualization
                    if 'frequency_analysis' in rating_extra_info:
                        if 'temporal_frequency_analysis' not in total_viz_data:
                            total_viz_data['temporal_frequency_analysis'] = [None] * len(self.attention_layers)
                        
                        freq_analysis = rating_extra_info['frequency_analysis']
                        total_viz_data['temporal_frequency_analysis'][i] = freq_analysis
                        
                        # 生成fourier_rating_attention_detailed数据结构
                        if 'fourier_rating_attention_detailed' not in total_viz_data:
                            total_viz_data['fourier_rating_attention_detailed'] = [None] * len(self.attention_layers)
                        
                        # 按domain_id组织注意力数据
                        layer_fourier_data = {}
                        for domain_key, domain_analysis in freq_analysis.items():
                            if 'visualization_data' in domain_analysis:
                                viz_data = domain_analysis['visualization_data']
                                # 提取domain_id（例如 "domain_0" -> 0）
                                if domain_key.startswith('domain_'):
                                    domain_id = int(domain_key.split('_')[1])
                                    layer_fourier_data[domain_id] = {
                                        'adaptive_weights': viz_data['adaptive_weights']  # 只保留自适应权重
                                    }
                        
                        total_viz_data['fourier_rating_attention_detailed'][i] = layer_fourier_data
                else:
                    # Use traditional simple rating embedding for backward compatibility
                    rating_embedding = self.rating_emb(rating_seqs)
                    # Add the same positional embedding to rating embedding for alignment
                    if getattr(self.args, 'rating_pos_emb', False):
                        rating_embedding += pos_embedding

            # Pass all relevant information to the FFN/MoE layer
            ffn_output, moe_loss_dict_layer, viz_data_layer = self.forward_layers[i](
                normalized_seqs, seqs, domain_ids, rating_embedding,
            )
            
            if hasattr(self.forward_layers[i], 'use_moe') and self.forward_layers[i].use_moe:
                # 合并viz_data_layer到total_viz_data
                for key, value in viz_data_layer.items():
                    total_viz_data[key] = value
            
            # Handle residual connection based on MoE vs non-MoE
            if hasattr(self.forward_layers[i], 'use_moe') and self.forward_layers[i].use_moe:
                # MoE path: apply residual connection externally
                seqs = original_seqs + ffn_output
            else:
                # Non-MoE path: FFN layer already includes residual connection internally
                seqs = ffn_output

            # Accumulate losses from the dictionary
            if isinstance(moe_loss_dict_layer, dict):
                for key, value in moe_loss_dict_layer.items():
                    if torch.is_tensor(value):
                        # Ensure the key exists in total_moe_loss_dict
                        if key not in total_moe_loss_dict:
                            total_moe_loss_dict[key] = 0.0
                        total_moe_loss_dict[key] += value

        log_feats = self.last_layernorm(seqs)
        return log_feats, total_moe_loss_dict, total_viz_data

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, rating_seqs=None, domain_ids=None): # for training
        log_feats, total_moe_loss_dict, viz_data = self.log2feats(log_seqs, rating_seqs, domain_ids)

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, total_moe_loss_dict, viz_data

    def predict(self, user_ids, log_seqs, item_indices, domain_ids=None): # for inference
        # Note: rating_seqs is not used in prediction, so we pass None.
        log_feats, _, _ = self.log2feats(log_seqs, rating_seqs=None, domain_ids=domain_ids)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(item_indices) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
