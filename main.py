import os
import time
import torch
import argparse
from tqdm import tqdm
from keys.model import SynRec
from keys.utils import *
import random
import swanlab
import numpy as np
from collections import OrderedDict, defaultdict

# Visualization modules removed - using only SwanLab for training monitoring
# Enhanced visualization available in scripts/inference_visualization.py for inference analysis
ENHANCED_VIZ_AVAILABLE = False

# ANSI color codes for terminal output
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m" 
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def check_rating_strategy_compatibility(args):
    """
    检查增强Rating策略与其他功能的兼容性，并提供优化建议
    """
    rating_strategy = getattr(args, 'rating_strategy', 'temporal_fourier')
    
    # 当前实现只支持temporal_fourier策略
    if rating_strategy == 'temporal_fourier':
        print(f"✅ Using optimized temporal-frequency rating strategy.")
        print(f"   This strategy uses learnable frequency cutoff and unified domain processing.")
    elif rating_strategy in ['simple', 'legacy']:
        print(f"✅ Using simple rating strategy for backward compatibility.")
    else:
        print(f"⚠️  WARNING: Rating strategy '{rating_strategy}' is not supported in current implementation.")
        print(f"   Supported strategies: 'simple', 'legacy', 'temporal_fourier'")
        print(f"   Falling back to 'temporal_fourier' strategy.")
        args.rating_strategy = 'temporal_fourier'
    
    return args
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=100, type=int, help='Maximum sequence length.')
    parser.add_argument('--hidden_units', default=64, type=int, help='Size of hidden vectors.')
    parser.add_argument('--num_blocks', default=2, type=int, help='Number of transformer blocks.')
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--num_heads', default=2, type=int, help='Number of attention heads.')
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--seed', default=42, type=int, help='Seed for reproducibility.')
    parser.add_argument('--use_domain_sampling', default=False, type=str2bool, help='Enable domain-specific sampling,if false use global sampling')
    parser.add_argument('--use_domain_sampling_for_evaluation', default=False, type=str2bool, help='Enable domain-specific candidate pool for evaluation (domain-limited)')
    parser.add_argument('--full_ranking_eval', default=False, type=str2bool, help='Enable full-ranking evaluation (scores against all items)')
    parser.add_argument('--eval_item_batch_size', default=4096, type=int, help='Batch size for item scoring during full-ranking evaluation')
    parser.add_argument('--eval_negative_sample_size', default=100, type=int, help='Number of negatives per test instance in sampled evaluation')
    parser.add_argument('--use_moe', default=True, type=str2bool, help='Enable/Disable MoE')
    parser.add_argument('--use_datasets', nargs='+', default=['baby_5_5', 'tools_5_5', 'toys_5_5'], help='Datasets to use for multi-domain training')
    parser.add_argument('--use_domain_info', default=True, type=str2bool, help='Use domain info in MoE gating')
    parser.add_argument('--use_rating_emb', default=True, type=str2bool, help='Use rating embedding to inform gating')
    parser.add_argument('--use_gated_fusion', default=True, type=str2bool, help='Use a gated mechanism to fuse rating embedding')
    parser.add_argument('--rating_pos_emb', default=False, type=str2bool, help='Add positional embedding to rating embeddings')
    parser.add_argument('--rating_strategy', default='temporal_fourier', type=str, 
                       choices=['simple', 'legacy', 'temporal_fourier'],
                       help='Strategy for rating information modeling: simple/legacy (backward compatibility), temporal_fourier (Optimized temporal-frequency domain feature extraction)')
    # temporal_fourier strategy uses learnable parameters, no manual config needed
    parser.add_argument('--moe_num_experts', default=4, type=int, help='Number of experts in MoE')
    parser.add_argument('--moe_k', default=2, type=int, help='Number of experts to use for each token')
    parser.add_argument('--moe_noisy_gating', default=True, type=str2bool, help='Use noisy gating in MoE')
    parser.add_argument('--moe_routing_strategy', default='shared_base', type=str, choices=['vanilla', 'shared_base'], help='MoE routing strategy')
    parser.add_argument('--moe_load_balancing', default=True, type=str2bool, help='Use load balancing in MoE')
    parser.add_argument('--moe_balance_loss_weight', default=0.01, type=float, help='Weight for MoE load balancing loss')
    parser.add_argument('--use_specialization_loss', default=True, type=str2bool, help='Enable specialization loss for expert specialization')
    parser.add_argument('--specialization_weight', default=0.01, type=float, help='Weight for specialization loss')
    parser.add_argument('--use_contrastive_loss', default=True, type=str2bool, help='Enable contrastive learning for expert specialization')
    parser.add_argument('--contrastive_weight', default=0.01, type=float, help='Weight for contrastive loss')
    parser.add_argument('--visualize', default=False, type=str2bool, help='Enable data collection for visualization (used by scripts)')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loading.')
    parser.add_argument('--swanlab_project', type=str, default='HAGMRec', help='SwanLab project name')
    parser.add_argument('--use_swanlab', default=True, type=str2bool, help='Enable/Disable SwanLab')
    parser.add_argument('--patience', default=10, type=int, help='Early stopping patience')
    parser.add_argument('--baseline_preset', default=None, type=str, 
                        choices=[None, 'shared_bottom', 'vanilla_moe', 'sharedbase_wo_rating', 'synrec_full'],
                        help='Baseline preset to ensure fair, reproducible configurations')
    parser.add_argument('--shared_user_ids', default=False, type=str2bool, help='If True, assumes input datasets share a global user ID space (no user offsets applied).')
    args = parser.parse_args()
    
    # Check compatibility between rating strategy and other options
    args = check_rating_strategy_compatibility(args)
    
    return args

def main():
    args = parse_args()
    # Apply baseline preset overrides for fair comparisons
    if getattr(args, 'baseline_preset', None) is not None:
        preset = args.baseline_preset
        if preset == 'shared_bottom':
            # No MoE; shared Transformer backbone
            args.use_moe = False
            args.use_rating_emb = False
            args.use_gated_fusion = False
            args.use_specialization_loss = False
            args.use_contrastive_loss = False
            # keep load balancing irrelevant when no MoE
        elif preset == 'vanilla_moe':
            # Plain MoE: top-k across all experts; remove rating/domain enhancements and MoE-specific specialization losses
            args.use_moe = True
            args.moe_routing_strategy = 'vanilla'
            args.use_rating_emb = False
            args.use_gated_fusion = False
            args.use_domain_info = False
            args.use_specialization_loss = False
            args.use_contrastive_loss = False
            # keep load balancing True for stability
        elif preset == 'sharedbase_wo_rating':
            # Our shared-base MoE without rating signals, to isolate frequency-guided routing effects
            args.use_moe = True
            args.moe_routing_strategy = 'shared_base'
            args.use_rating_emb = False
            args.use_gated_fusion = False
            args.use_domain_info = True
            # keep specialization/contrastive/lb as in full model
        elif preset == 'synrec_full':
            # Full SynRec
            args.use_moe = True
            args.moe_routing_strategy = 'shared_base'
            args.use_rating_emb = True
            args.use_gated_fusion = True
            args.use_domain_info = True
            args.use_specialization_loss = True
            args.use_contrastive_loss = True
            args.moe_load_balancing = True
        else:
            pass
    # Set the seed for the entire environment
    if args.seed is not None:
        set_seed(args.seed)
    
    # Initialize SwanLab
    if args.use_swanlab:
        swanlab.init(
            project=args.swanlab_project,
            experiment_name='-'.join(args.use_datasets),
            config=vars(args)
        )
        print("📊 SwanLab上传已启用")

    dataset_name_str = '-'.join(args.use_datasets)
    experiment_dir = os.path.join('exp', dataset_name_str + '_' + args.train_dir)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
    with open(os.path.join(experiment_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    # global dataset
    dataset = partition_multi_domain(args.use_datasets, shared_user_ids=args.shared_user_ids)
    [user_train, user_valid, user_test, user_to_domain, usernum, itemnum, domain_to_item_range] = dataset
    args.num_domains = len(args.use_datasets) # Save number of domains
    domain_map = {i: name for i, name in enumerate(args.use_datasets)}
    

    asl = {f'domain_{i}': 0 for i in range(args.num_domains)} # total sequence length for each domain
    usrnum_of_domain = {f'domain_{i}': 0 for i in range(args.num_domains)} # user number of each domain

    print('\n{:-^100}'.format('Average sequence length of each domain'))
    for u in user_train:
        asl[f'domain_{user_to_domain[u]}'] += len(user_train[u])
        usrnum_of_domain[f'domain_{user_to_domain[u]}'] += 1
    
    for i in range(args.num_domains):
        print(f'{domain_map[i]}: {asl[f"domain_{i}"] / usrnum_of_domain[f"domain_{i}"]:.2f}')

    f = open(os.path.join(experiment_dir, 'log.txt'), 'w')
    f.write('epoch\tvalid_metrics\ttest_metrics\n')
    
    train_dataset = MoerecDataset(user_train, user_to_domain, usernum, itemnum, args.maxlen, args, domain_to_item_range)
    train_sampler = StratifiedSampler(train_dataset)
    train_collator = MoerecCollator(maxlen=args.maxlen)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler, # Use our custom sampler
        shuffle=False, # Shuffle must be False when using a custom sampler
        num_workers=args.num_workers,
        collate_fn=train_collator,
        pin_memory=True # Speeds up data transfer to GPU
    )
    model = SynRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    total_params = 0
    original_params = 0
    moe_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if 'moe_ffn' in name:
            moe_params += num_params
        else:
            original_params += num_params

    print('\n{:-^100}'.format("Model Parameters"))
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Transformer Block Parameters: {original_params:,}")
    print(f"  MoE-related Parameters: {moe_params:,}")
    print("-" * 100,"\n")

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        print('Running inference-only evaluation...')
        
        # 创建日志文件（如果不存在）
        f = open(os.path.join(experiment_dir, 'log.txt'), 'w')
        f.write('epoch\tvalid_metrics\ttest_metrics\n')
        
        # 进行评估
        t_valid = evaluate_batched(model, dataset, args, 'valid')
        t_test = evaluate_batched(model, dataset, args, 'test')

        # 获取epoch信息（从权重文件名提取，如果可能的话）
        epoch_num = 'inference'
        if args.state_dict_path and 'epoch=' in args.state_dict_path:
            try:
                tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
                epoch_num = int(tail[:tail.find('.')])
            except:
                epoch_num = 'inference'
        
        # 打印结果
        print(f'Inference Results - Valid: NDCG@10={t_valid["overall_NDCG@10"]:.4f}, HR@10={t_valid["overall_HT@10"]:.4f} | Test: NDCG@10={t_test["overall_NDCG@10"]:.4f}, HR@10={t_test["overall_HT@10"]:.4f}')
        print(f"Valid metrics: {t_valid}")
        print(f"Test metrics: {t_test}")
        
        # 保存结果到log.txt文件（与训练模式相同格式）
        valid_metrics_str = ",".join([f"{k}:{v:.4f}" for k, v in sorted(t_valid.items())])
        test_metrics_str = ",".join([f"{k}:{v:.4f}" for k, v in sorted(t_test.items())])
        f.write(f'{epoch_num}\t{valid_metrics_str}\t{test_metrics_str}\n')
        f.close()
        
        print(f"Results saved to: {os.path.join(experiment_dir, 'log.txt')}")
        print("You can now use analyze_results.py to visualize and compare these results.")
        return

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_test_ndcg = 0.0
    best_val_ndcg = 0.0
    patience_counter = 0
    prev_best_model_path = None # Keep track of the previous best model to delete it
    T = 0.0
    
    # Simplified visualization setup - data collection preserved for scripts
    print(f"🔧 Training mode: basic SwanLab monitoring only")
    print(f"   Enhanced visualization available via scripts/inference_visualization.py")
    
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        
        pbar = tqdm(train_loader, desc=f"{BOLD}{BLUE}Epoch {epoch}/{args.num_epochs}{RESET}", colour='blue')
        for step, (u, seq, rating_seq, pos, neg, domain_id) in enumerate(pbar):
            # Move batch to device
            u, seq, rating_seq, pos, neg, domain_id = u.to(args.device), seq.to(args.device), rating_seq.to(args.device), pos.to(args.device), neg.to(args.device), domain_id.to(args.device)

            pos_logits, neg_logits, moe_loss_dict, viz_data = model(u, seq, pos, neg, rating_seqs=rating_seq, domain_ids=domain_id)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            
            adam_optimizer.zero_grad()
            indices = torch.where(pos != 0)
            
            postfix_data = OrderedDict()
            bpr_loss = bce_criterion(pos_logits[indices], pos_labels[indices]) + bce_criterion(neg_logits[indices], neg_labels[indices])
            postfix_data['bpr_loss'] = f"{BOLD}{BLUE}{bpr_loss.item():.4f}{RESET}"

            loss = bpr_loss
            
            for i in moe_loss_dict.keys():
                if torch.is_tensor(moe_loss_dict[i]):
                    loss = loss + moe_loss_dict[i]
                    postfix_data[i] = f"{BOLD}{BLUE}{moe_loss_dict[i].item():.4f}{RESET}"

                    if args.l2_emb > 0:
                        for param in model.item_emb.parameters(): 
                            loss = loss + args.l2_emb * torch.norm(param)
            
            loss.backward()
            adam_optimizer.step()

            postfix_data['loss'] = f"{BOLD}{BLUE}{loss.item():.4f}{RESET}"
            pbar.set_postfix(postfix_data)

            # SwanLab basic training monitoring
            if args.use_swanlab:
                log_data = {
                    'train/loss': loss.item(),
                    'train/bpr_loss': bpr_loss.item(),
                    'learning_rate': adam_optimizer.param_groups[0]['lr']
                }
                for k, v_val in moe_loss_dict.items():
                    if torch.is_tensor(v_val):
                        log_data[f'train/{k}'] = v_val.item()
                        
                # Expert load monitoring for SwanLab (basic metrics only)
                if 'expert_load' in viz_data:
                    expert_load = viz_data['expert_load']
                    for i, val in enumerate(expert_load):
                        load_key = f'train_expert_load/Domain_{domain_map[i]}' if args.moe_routing_strategy == 'shared_base' else f'train_expert_load/Expert_{i}'
                        log_data[load_key] = val.item()
                
                swanlab_global_step = (epoch - 1) * len(train_loader) + step
                if swanlab.get_run() is not None:
                    swanlab.log(log_data, step=swanlab_global_step)


        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            t_valid = evaluate_batched(model, dataset, args, 'valid')
            t_test = evaluate_batched(model, dataset, args, 'test')
            print('epoch:%d, time: %f(s)' % (epoch, T))

            def pretty_print_metrics(metrics_dict, title, color_code):
                print(f"\n  {BOLD}{color_code}[{title}]{RESET}")
                
                domain_metrics = defaultdict(dict)
                overall_metrics = {}
                weighted_overall_metrics = {}
                
                for k, v in sorted(metrics_dict.items()):
                    if k.startswith('domain_'):
                        parts = k.split('_')
                        domain_id = int(parts[1])
                        metric_name = '_'.join(parts[2:])
                        domain_metrics[domain_id][metric_name] = v
                    elif k.startswith('overall_weighted_'):
                        metric_name = k.replace('overall_weighted_', '')
                        weighted_overall_metrics[metric_name] = v
                    elif k.startswith('overall_') and not k.startswith('overall_eval_'):
                        metric_name = k.replace('overall_', '')
                        overall_metrics[metric_name] = v

                for domain_id, d_metrics in sorted(domain_metrics.items()):
                    raw_domain_name = domain_map.get(domain_id, f"Unknown Domain {domain_id}")
                    # 规范化领域名称显示
                    if not raw_domain_name.startswith("Unknown Domain"):
                        try:
                            from visualization.enhanced_plots import _normalize_domain_name
                            domain_name = _normalize_domain_name(raw_domain_name)
                        except ImportError:
                            domain_name = raw_domain_name
                    else:
                        domain_name = raw_domain_name
                    print(f"    - {BOLD}Domain: {domain_name}{RESET}")
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in sorted(d_metrics.items())])
                    print(f"        {metrics_str}")
                
                if overall_metrics or weighted_overall_metrics:
                    print("    " + "-"*50)
                    if overall_metrics:
                        print(f"    - {BOLD}Overall{RESET}")
                        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in sorted(overall_metrics.items())])
                        print(f"        {metrics_str}")
                    if weighted_overall_metrics:
                        print(f"    - {BOLD}Weighted Overall{RESET}")
                        metrics_str = ", ".join([f"weighted_{k}: {v:.4f}" for k, v in sorted(weighted_overall_metrics.items())])
                        print(f"        {metrics_str}")

            pretty_print_metrics(t_valid, "Full Valid Metrics", GREEN)
            pretty_print_metrics(t_test, "Full Test Metrics", CYAN)
            
            # Print inference performance summary after all metrics
            print(f"\n  {BOLD}Inference Performance Summary{RESET}")
            valid_eval_seconds = t_valid.get('overall_eval_seconds', 0)
            valid_eval_throughput = t_valid.get('overall_eval_throughput_users_s', 0)
            test_eval_seconds = t_test.get('overall_eval_seconds', 0)
            test_eval_throughput = t_test.get('overall_eval_throughput_users_s', 0)
            
            print(f"    - {BOLD}Valid Set{RESET}")
            print(f"        Eval time (s): {valid_eval_seconds:.2f}")
            print(f"        Eval throughput (users/s): {valid_eval_throughput:.2f}")
            print(f"    - {BOLD}Test Set{RESET}")
            print(f"        Eval time (s): {test_eval_seconds:.2f}")
            print(f"        Eval throughput (users/s): {test_eval_throughput:.2f}")

            if args.use_swanlab:
                eval_log_dict = {"epoch": epoch}
                # Consolidate all validation and test metrics into a single dictionary
                for key, value in t_valid.items():
                    metric_name = f'eval/valid_{key}'
                    eval_log_dict[metric_name] = value
                for key, value in t_test.items():
                    metric_name = f'eval/test_{key}'
                    eval_log_dict[metric_name] = value
                
                if args.use_swanlab:
                    if swanlab.get_run() is not None:
                        swanlab.log(eval_log_dict)

            # Early stopping logic based on Validation NDCG
            performance_improved = t_valid['overall_NDCG@10'] > best_val_ndcg
            
            if performance_improved:
                best_val_ndcg = t_valid['overall_NDCG@10']
                best_test_ndcg = t_test['overall_NDCG@10'] # Update best test score for reference
                patience_counter = 0 # Reset patience
                
                print(f"✨ New best Valid NDCG@10: {best_val_ndcg:.4f} (Test: {best_test_ndcg:.4f})")
                folder = experiment_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                model_path = os.path.join(folder, fname)
                torch.save(model.state_dict(), model_path)
                
                # Delete previous best model to save space
                if prev_best_model_path and os.path.exists(prev_best_model_path) and prev_best_model_path != model_path:
                    try:
                        os.remove(prev_best_model_path)
                    except OSError as e:
                        print(f"Warning: Could not delete previous model file {prev_best_model_path}: {e}")
                
                prev_best_model_path = model_path
            else:
                patience_counter += 1
                print(f"⚠️ Performance did not improve. Patience: {patience_counter}/{args.patience}")
                if patience_counter >= args.patience:
                    print(f"🛑 Early stopping triggered after {epoch} epochs.")
                    break

    
            # Format the metrics string for log file
            valid_metrics_str = ",".join([f"{k}:{v:.4f}" for k, v in sorted(t_valid.items())])
            test_metrics_str = ",".join([f"{k}:{v:.4f}" for k, v in sorted(t_test.items())])
            f.write(f'{epoch}\t{valid_metrics_str}\t{test_metrics_str}\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = experiment_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    
    f.close()
    if args.use_swanlab:
        swanlab.finish()
    print("Done")

if __name__ == '__main__':
    main()
