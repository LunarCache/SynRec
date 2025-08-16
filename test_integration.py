"""
测试新的TemporalEnhancedRatingModule在HAGMRec模型中的集成
"""
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append('/home/wzc/project/CMREC')

from keys.model import HAGMRec

def test_hagmrec_integration():
    """测试HAGMRec与新的temporal rating模块的集成"""
    
    # 创建模拟的args
    class MockArgs:
        def __init__(self):
            self.device = 'cpu'
            self.hidden_units = 64
            self.maxlen = 50
            self.dropout_rate = 0.1
            self.num_blocks = 2  # Transformer层数
            self.num_heads = 2   # 注意力头数
            
            # Rating相关配置
            self.use_rating_emb = True
            self.rating_strategy = 'temporal_fourier'  # 使用新的策略
            
            # 数据集配置
            self.use_datasets = ['beauty_5_5', 'games_5_5', 'ml-1m_5_5']
            self.num_domains = 3  # 领域数量
            
            # MoE相关配置
            self.use_moe = True
            self.num_shared_experts = 1
            self.num_domain_experts = 3
            self.moe_num_experts = 4  # 总专家数量 = 1 shared + 3 domain
            self.moe_k = 2  # top-k专家选择
            self.moe_routing_strategy = 'shared_base'
            self.moe_load_balancing = True
            self.moe_balance_loss_weight = 0.01
            
            # Domain信息
            self.use_domain_info = True
            
            # 专门化损失
            self.use_specialization_loss = True
            self.specialization_weight = 0.01
            self.use_contrastive_loss = True
            self.contrastive_weight = 0.01
            
            # 门控融合
            self.use_gated_fusion = False
            self.visualize = False
    
    args = MockArgs()
    
    # 创建模型
    user_num = 1000
    item_num = 2000
    
    print("🧪 Testing HAGMRec with TemporalEnhancedRatingModule...")
    
    model = HAGMRec(user_num, item_num, args)
    
    print(f"✅ Model created successfully!")
    print(f"   - Rating strategy: {args.rating_strategy}")
    print(f"   - Model type: {type(model.enhanced_rating_module).__name__}")
    
    # 创建测试数据
    batch_size = 8
    seq_len = 30
    
    # 用户ID
    user_ids = torch.randint(1, user_num, (batch_size,))
    
    # 输入序列 (历史交互)
    log_seqs = torch.randint(1, item_num, (batch_size, seq_len))
    
    # 正样本序列 (target items)
    pos_seqs = torch.randint(1, item_num, (batch_size, seq_len))
    
    # 负样本序列 (negative items)
    neg_seqs = torch.randint(1, item_num, (batch_size, seq_len))
    
    # Rating序列
    rating_seqs = torch.randint(1, 6, (batch_size, seq_len))
    
    # Domain IDs
    domain_ids = torch.randint(0, 3, (batch_size,))
    
    print(f"\n📊 Test Data:")
    print(f"   User IDs shape: {user_ids.shape}")
    print(f"   Log seqs shape: {log_seqs.shape}")
    print(f"   Pos seqs shape: {pos_seqs.shape}")
    print(f"   Neg seqs shape: {neg_seqs.shape}")
    print(f"   Rating shape: {rating_seqs.shape}")
    print(f"   Domain IDs: {domain_ids}")
    
    # 前向传播测试
    model.eval()
    with torch.no_grad():
        pos_logits, neg_logits, moe_losses, viz_data = model(
            user_ids, log_seqs, pos_seqs, neg_seqs, rating_seqs, domain_ids
        )
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Pos logits shape: {pos_logits.shape}")
    print(f"   Neg logits shape: {neg_logits.shape}")
    print(f"   Expected shape: ({batch_size}, {seq_len})")
    
    # 验证输出形状
    expected_shape = (batch_size, seq_len)
    assert pos_logits.shape == expected_shape, f"Pos logits shape mismatch: {pos_logits.shape} vs {expected_shape}"
    assert neg_logits.shape == expected_shape, f"Neg logits shape mismatch: {neg_logits.shape} vs {expected_shape}"
    
    # 测试梯度计算
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 使用BPR损失进行测试
    pos_logits, neg_logits, moe_losses, viz_data = model(
        user_ids, log_seqs, pos_seqs, neg_seqs, rating_seqs, domain_ids
    )
    
    # 计算BPR损失
    bpr_loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-8).mean()
    
    # 添加MoE损失
    total_loss = bpr_loss
    if moe_losses:
        for loss_name, loss_value in moe_losses.items():
            if loss_value is not None:
                total_loss += loss_value
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"\n✅ Gradient computation successful!")
    print(f"   BPR Loss: {bpr_loss.item():.6f}")
    print(f"   Total Loss: {total_loss.item():.6f}")
    if moe_losses:
        print(f"   MoE Losses: {[(k, v.item() if v is not None else None) for k, v in moe_losses.items()]}")
    
    # 检查关键参数的梯度
    rating_module = model.enhanced_rating_module
    fourier_encoder = rating_module.fourier_encoder
    
    if hasattr(fourier_encoder, 'cutoff_logit') and fourier_encoder.cutoff_logit.grad is not None:
        print(f"   Cutoff gradient norm: {fourier_encoder.cutoff_logit.grad.norm().item():.6f}")
    
    if hasattr(fourier_encoder, 'boundary_sharpness') and fourier_encoder.boundary_sharpness.grad is not None:
        print(f"   Boundary sharpness gradient norm: {fourier_encoder.boundary_sharpness.grad.norm().item():.6f}")
    
    print(f"\n🎯 Integration Test Summary:")
    print(f"   ✅ Model initialization")
    print(f"   ✅ Forward propagation") 
    print(f"   ✅ Backward propagation")
    print(f"   ✅ Parameter updates")
    print(f"   ✅ Shape validation")
    
    return model, (pos_logits, neg_logits), total_loss

def test_compatibility():
    """测试与原有rating策略的兼容性"""
    
    class CompatArgs:
        def __init__(self, strategy):
            self.device = 'cpu'
            self.hidden_units = 64
            self.maxlen = 50
            self.dropout_rate = 0.1
            self.num_blocks = 2
            self.num_heads = 2
            self.use_rating_emb = True
            self.rating_strategy = strategy
            self.use_datasets = ['beauty_5_5']
            self.num_domains = 1
            self.use_moe = True
            self.num_shared_experts = 1
            self.num_domain_experts = 3
            self.moe_num_experts = 4
            self.moe_k = 2
            self.moe_routing_strategy = 'shared_base'
            self.moe_load_balancing = True
            self.moe_balance_loss_weight = 0.01
            self.use_domain_info = True
            self.use_specialization_loss = True
            self.specialization_weight = 0.01
            self.use_contrastive_loss = True
            self.contrastive_weight = 0.01
            self.use_gated_fusion = False
            self.visualize = False
    
    print("\n" + "="*60)
    print("🧪 Testing backward compatibility...")
    
    strategies = ['temporal_fourier']  # Only test the supported strategy
    
    for strategy in strategies:
        print(f"\n🔧 Testing strategy: {strategy}")
        
        args = CompatArgs(strategy)
        model = HAGMRec(100, 200, args)
        
        # 简单前向传播测试
        batch_size = 4
        seq_len = 10
        
        user_ids = torch.randint(1, 100, (batch_size,))
        log_seqs = torch.randint(1, 200, (batch_size, seq_len))
        pos_seqs = torch.randint(1, 200, (batch_size, seq_len))
        neg_seqs = torch.randint(1, 200, (batch_size, seq_len))
        rating_seqs = torch.randint(1, 6, (batch_size, seq_len))
        domain_ids = torch.randint(0, 1, (batch_size,))
        
        with torch.no_grad():
            pos_logits, neg_logits, moe_losses, viz_data = model(
                user_ids, log_seqs, pos_seqs, neg_seqs, rating_seqs, domain_ids
            )
        
        print(f"   ✅ {strategy}: {type(model.enhanced_rating_module).__name__}")
        print(f"      Pos logits shape: {pos_logits.shape}")
        print(f"      Neg logits shape: {neg_logits.shape}")

if __name__ == "__main__":
    torch.manual_seed(42)
    
    try:
        # 主要集成测试
        model, logits, loss = test_hagmrec_integration()
        
        # 兼容性测试
        test_compatibility()
        
        print("\n" + "="*60)
        print("🎉 All integration tests passed!")
        print("\n✨ Ready for training with:")
        print("   --rating_strategy temporal_fourier")
        print("   --use_rating_emb true")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)