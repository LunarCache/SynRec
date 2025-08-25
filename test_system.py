#!/usr/bin/env python3
"""
CMREC系统测试脚本
快速验证模型的基本功能和性能
"""

import torch
import numpy as np
import argparse
import time
from keys.model import HAGMRec
from keys.c_moe import HAGMoEFFN
from keys.utils import partition_multi_domain, MoerecStyleSampler

def test_model_creation():
    """测试模型创建"""
    print("1. Testing model creation...")
    
    class MockArgs:
        def __init__(self):
            self.hidden_units = 64
            self.maxlen = 50
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
            self.moe_noisy_gating = True
            self.moe_routing_strategy = 'shared_base'
            
            # Rating embedding parameters
            self.use_rating_emb = True
            self.rating_strategy = 'temporal_fourier'  # Updated to use new strategy
            self.use_datasets = ['beauty_rated', 'games', 'ml-1m_rated']
            
            # Fusion parameters  
            self.use_gated_fusion = True
            
            # Loss parameters
            self.use_specialization_loss = True
            self.specialization_weight = 0.01
            self.use_contrastive_loss = True
            self.contrastive_weight = 0.01
            
            # Legacy parameters for compatibility
            self.contrastive_learning = True
            self.contrastive_projection_size = 128
            self.contrastive_temperature = 0.07
            self.spec_loss_weight = 0.05
            self.cohe_loss_weight = 0.05
            self.use_spec_loss = True
            self.use_cohe_loss = True
            self.visualize = False    
    args = MockArgs()
    usernum, itemnum = 1000, 1000
    
    try:
        model = HAGMRec(usernum, itemnum, args)
        print("   ✓ Model created successfully")
        
        # 检查参数数量
        total_params = sum(p.numel() for p in model.parameters())
        moe_params = sum(p.numel() for name, p in model.named_parameters() if 'moe_ffn' in name)
        
        print(f"   ✓ Total parameters: {total_params:,}")
        print(f"   ✓ MoE parameters: {moe_params:,}")
        
        return model, args
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        return None, None

def test_forward_pass(model, args):
    """测试前向传播"""
    print("\n2. Testing forward pass...")
    
    try:
        # 模拟输入数据
        batch_size = 4
        seq_len = args.maxlen
        
        u = np.random.randint(1, 1000, size=batch_size)
        seq = np.random.randint(1, 1000, size=(batch_size, seq_len))
        pos = np.random.randint(1, 1000, size=(batch_size, seq_len))
        neg = np.random.randint(1, 1000, size=(batch_size, seq_len))
        rating_seqs = torch.randint(1, 6, (batch_size, seq_len))  # 评分1-5
        domain_ids = torch.LongTensor([0, 1, 2, 0])
        
        model.eval()
        with torch.no_grad():
            pos_logits, neg_logits, moe_loss_dict, _ = model(u, seq, pos, neg, rating_seqs, domain_ids=domain_ids)

        print(f"   ✓ Forward pass successful")
        print(f"   ✓ Pos logits shape: {pos_logits.shape}")
        print(f"   ✓ Neg logits shape: {neg_logits.shape}")
        print(f"   ✓ MoE losses: {list(moe_loss_dict.keys())}")
        
        return True
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return False

def test_gradient_flow(model, args):
    """测试梯度流"""
    print("\n3. Testing gradient flow...")
    
    try:
        model.train()
        
        # 模拟训练数据
        batch_size = 4
        seq_len = args.maxlen
        
        u = np.random.randint(1, 1000, size=batch_size)
        seq = np.random.randint(1, 1000, size=(batch_size, seq_len))
        pos = np.random.randint(1, 1000, size=(batch_size, seq_len))
        neg = np.random.randint(1, 1000, size=(batch_size, seq_len))
        rating_seqs = torch.randint(1, 6, (batch_size, seq_len))  # 评分1-5
        domain_ids = torch.LongTensor([0, 1, 2, 0])
        
        # 前向传播
        pos_logits, neg_logits, moe_loss_dict,_ = model(u, seq, pos, neg, rating_seqs, domain_ids=domain_ids)
        
        # 计算损失
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
        
        # 反向传播
        model.zero_grad()
        total_loss.backward()
        
        # 检查梯度
        params_with_grad = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None and param.grad.norm() > 0:
                params_with_grad += 1
        
        grad_coverage = params_with_grad / total_params * 100
        
        print(f"   ✓ Gradient computation successful")
        print(f"   ✓ Parameters with gradients: {params_with_grad}/{total_params}")
        print(f"   ✓ Gradient coverage: {grad_coverage:.1f}%")
        
        if grad_coverage < 90:
            print("   ⚠ Warning: Low gradient coverage, some parameters may not be learning")
        
        return True
    except Exception as e:
        print(f"   ✗ Gradient flow test failed: {e}")
        return False

def test_data_loading():
    """测试数据加载（如果数据存在）"""
    print("\n4. Testing data loading...")
    
    try:
        # 尝试加载测试数据集
        test_datasets = ['beauty', 'games']  # 更可能存在的数据集
        
        # 检查数据文件是否存在
        import os
        available_datasets = []
        for dataset in test_datasets:
            if os.path.exists(f'data/{dataset}.txt'):
                available_datasets.append(dataset)
        
        if not available_datasets:
            print("   ⚠ No data files found, skipping data loading test")
            print("   ℹ To test data loading, place dataset files in data/ directory")
            return True
        
        print(f"   ✓ Found datasets: {available_datasets}")
        
        # 加载数据
        dataset = partition_multi_domain(available_datasets)
        user_train, user_valid, user_test, user_to_domain, usernum, itemnum, domain_to_item_range = dataset
        
        print(f"   ✓ Data loaded successfully")
        print(f"   ✓ Users: {usernum}, Items: {itemnum}")
        print(f"   ✓ Domains: {len(set(user_to_domain.values()))}")
        
        # 测试采样器
        class MockArgs:
            maxlen = 50
        
        args = MockArgs()
        sampler = MoerecStyleSampler(user_train, user_to_domain, usernum, itemnum, 
                                   batch_size=4, maxlen=args.maxlen, args=args, 
                                   domain_to_item_range=domain_to_item_range)
        
        # 测试一个批次
        for batch in sampler:
            u, seq, pos, neg, domain_id = batch
            print(f"   ✓ Sampler working, batch size: {len(u)}")
            break
        
        sampler.close()
        return True
        
    except Exception as e:
        print(f"   ✗ Data loading test failed: {e}")
        return False


def test_moe_component():
    """测试MoE组件"""
    print("\n6. Testing MoE component...")
    
    try:
        class MockArgs:
            def __init__(self):
                self.num_domains = 3
                self.moe_num_experts = 4
                self.moe_k = 2
                self.use_domain_info = True
                self.moe_load_balancing = True
                self.moe_balance_loss_weight = 0.01
                self.moe_noisy_gating = True
                self.moe_routing_strategy = 'shared_base'
                self.dropout_rate = 0.1
                
                # New parameters for current implementation
                self.use_specialization_loss = True
                self.specialization_weight = 0.01
                self.use_contrastive_loss = True
                self.contrastive_weight = 0.01
                self.use_gated_fusion = True
                
                # Legacy parameters for compatibility
                self.contrastive_learning = True
                self.contrastive_projection_size = 128
                self.contrastive_temperature = 0.07
                self.spec_loss_weight = 0.05
                self.cohe_loss_weight = 0.05
                self.use_spec_loss = True
                self.use_cohe_loss = True
                self.visualize = False
        
        args = MockArgs()
        hidden_units = 64
        
        moe = HAGMoEFFN(hidden_units, args)
        moe.train()
        
        # 测试输入
        batch_size, seq_len = 4, 10
        inputs = torch.randn(batch_size, seq_len, hidden_units)
        domain_ids = torch.LongTensor([0, 1, 2, 0])
        
        output, loss_dict, _ = moe(inputs, log_feats=None, domain_ids=domain_ids)
        
        print(f"   ✓ MoE forward pass successful")
        print(f"   ✓ Input shape: {inputs.shape}")
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Loss components: {list(loss_dict.keys())}")
        
        # 检查损失值
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                print(f"   ✓ {key}: {value.item():.6f}")
        
        return True
    except Exception as e:
        print(f"   ✗ MoE component test failed: {e}")
        return False

def performance_benchmark():
    """简单的性能基准测试"""
    print("\n6. Performance benchmark...")
    
    try:
        class MockArgs:
            def __init__(self):
                self.hidden_units = 64
                self.maxlen = 100
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
                self.spec_loss_weight = 0.05
                self.cohe_loss_weight = 0.05
                self.use_spec_loss = True
                self.use_cohe_loss = True
                self.visualize = False
        
        args = MockArgs()
        usernum, itemnum = 10000, 10000
        model = HAGMRec(usernum, itemnum, args)
        model.eval()
        
        # 测试批次
        batch_size = 32
        seq_len = args.maxlen
        
        u = np.random.randint(1, usernum, size=batch_size)
        seq = np.random.randint(1, itemnum, size=(batch_size, seq_len))
        pos = np.random.randint(1, itemnum, size=(batch_size, seq_len))
        neg = np.random.randint(1, itemnum, size=(batch_size, seq_len))
        domain_ids = torch.LongTensor(np.random.randint(0, 3, size=batch_size))
        
        # 计时测试
        num_iterations = 10
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                pos_logits, neg_logits, moe_loss_dict, _ = model(u, seq, pos, neg, domain_ids=domain_ids)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        print(f"   ✓ Average forward pass time: {avg_time:.4f}s")
        print(f"   ✓ Throughput: {batch_size / avg_time:.1f} samples/sec")
        
        return True
    except Exception as e:
        print(f"   ✗ Performance benchmark failed: {e}")
        return False

def main():
    print("="*60)
    print("CMREC SYSTEM TEST")
    print("="*60)
    
    # 运行所有测试
    test_results = []
    
    # 1. 模型创建测试
    model, args = test_model_creation()
    test_results.append(model is not None)
    
    if model is not None:
        # 2. 前向传播测试
        test_results.append(test_forward_pass(model, args))
        
        # 3. 梯度流测试
        test_results.append(test_gradient_flow(model, args))
    else:
        test_results.extend([False, False])
    
    # 4. 数据加载测试
    test_results.append(test_data_loading())
    
    # Note: Domain adaptive rating test removed - now using unified temporal_fourier strategy
    
    # 6. MoE组件测试
    test_results.append(test_moe_component())
    
    # 7. 性能基准测试
    test_results.append(performance_benchmark())
    
    # 总结结果
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    test_names = [
        "Model Creation",
        "Forward Pass", 
        "Gradient Flow",
        "Data Loading",
        "MoE Component",
        "Performance Benchmark"
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results), 1):
        status = "PASS" if result else "FAIL"
        print(f"{i}. {name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! System is ready for training.")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. Minor issues detected, but system should work.")
    else:
        print("❌ Multiple test failures. Please check the system configuration.")
    
    print("\nTo start training, run:")
    print("python main.py --train_dir=test_exp --use_datasets beauty games --num_epochs=10")

if __name__ == "__main__":
    main()
