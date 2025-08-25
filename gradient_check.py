#!/usr/bin/env python3
"""
梯度流检查脚本
检查训练过程中各个损失项对模型参数的梯度贡献
"""

import torch
import numpy as np
from keys.model import SASRec
import argparse

def check_gradient_flow(args):
    """检查梯度流是否正常"""
    
    # 创建模型
    usernum, itemnum = 1000, 1000
    model = SASRec(usernum, itemnum, args)
    model.train()
    
    # 模拟输入数据
    batch_size = 4
    seq_len = args.maxlen
    u = np.random.randint(1, usernum, size=batch_size)
    seq = np.random.randint(1, itemnum, size=(batch_size, seq_len))
    pos = np.random.randint(1, itemnum, size=(batch_size, seq_len))
    neg = np.random.randint(1, itemnum, size=(batch_size, seq_len))
    domain_ids = torch.LongTensor([0, 1, 2, 0])  # 3个域
    
    # 前向传播
    pos_logits, neg_logits, moe_loss_dict, _ = model(u, seq, pos, neg, domain_ids=domain_ids)
    
    print("=== 损失项检查 ===")
    for key, value in moe_loss_dict.items():
        if torch.is_tensor(value):
            print(f"{key}: {value.item():.6f} (requires_grad: {value.requires_grad})")
        else:
            print(f"{key}: {value} (not tensor)")
    
    # 计算总损失
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    pos_labels = torch.ones(pos_logits.shape)
    neg_labels = torch.zeros(neg_logits.shape)
    
    indices = np.where(pos != 0)
    bpr_loss = bce_criterion(pos_logits[indices], pos_labels[indices]) + \
               bce_criterion(neg_logits[indices], neg_labels[indices])
    
    total_loss = bpr_loss
    for key, value in moe_loss_dict.items():
        if torch.is_tensor(value) and value.requires_grad:
            total_loss = total_loss + value
    
    print(f"\n=== 总损失 ===")
    print(f"BPR Loss: {bpr_loss.item():.6f}")
    print(f"Total Loss: {total_loss.item():.6f}")
    print(f"Total Loss requires_grad: {total_loss.requires_grad}")
    
    # 反向传播前清零梯度
    model.zero_grad()
    
    # 启用异常检测
    torch.autograd.set_detect_anomaly(True)
    
    try:
        # 反向传播
        total_loss.backward()
    except RuntimeError as e:
        print(f"\n梯度计算出错: {e}")
        torch.autograd.set_detect_anomaly(False)
        return {}
    
    print(f"\n=== 梯度检查 ===")
    
    # 检查各个组件的梯度
    gradient_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_stats[name] = grad_norm
            if grad_norm > 0:
                print(f"✓ {name}: grad_norm = {grad_norm:.6f}")
            else:
                print(f"✗ {name}: grad_norm = 0 (可能没有梯度)")
        else:
            print(f"✗ {name}: grad is None")
    
    # 统计有梯度的参数
    params_with_grad = sum(1 for stats in gradient_stats.values() if stats > 0)
    total_params = len(gradient_stats)
    
    print(f"\n=== 梯度统计 ===")
    print(f"有梯度的参数: {params_with_grad}/{total_params}")
    print(f"梯度覆盖率: {params_with_grad/total_params*100:.1f}%")
    
    # 检查MoE特定的梯度
    print(f"\n=== MoE组件梯度检查 ===")
    moe_components = ['gate', 'shared_experts', 'domain_experts', 'projectors', 'domain_embedding','domain_attention','domain_transform']
    
    for component in moe_components:
        found_grads = [name for name, grad_norm in gradient_stats.items() 
                      if component in name and grad_norm > 0]
        if found_grads:
            print(f"✓ {component}: {len(found_grads)} 参数有梯度")
            for name in found_grads[:3]:  # 显示前3个
                print(f"  - {name}: {gradient_stats[name]:.6f}")
            if len(found_grads) > 3:
                print(f"  - ... 还有 {len(found_grads)-3} 个参数")
        else:
            print(f"✗ {component}: 没有找到梯度")
    
    return gradient_stats

if __name__ == "__main__":
    # 模拟训练参数
    class MockArgs:
        def __init__(self):
            self.hidden_units = 64
            self.maxlen = 10
            self.dropout_rate = 0.1
            self.num_blocks = 2
            self.num_heads = 2
            self.device = 'cpu'
            self.use_moe = True
            self.num_domains = 3
            self.moe_num_experts = 4
            self.moe_k = 2
            self.use_domain_info = True
            self.moe_load_balancing = True
            self.moe_balance_loss_weight = 0.01
            self.contrastive_learning = True
            self.contrastive_projection_size = 128
            self.contrastive_temperature = 0.07
            self.use_spec_loss = True
            self.use_cohe_loss = True
            self.spec_loss_weight = 0.05
            self.cohe_loss_weight = 0.05
            self.use_dynamic_domain_emb= True
            self.visualize = False
    args = MockArgs()
    
    print("开始梯度流检查...")
    gradient_stats = check_gradient_flow(args)
    print("梯度流检查完成！")
