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
    parser.add_argument('--use_domain_sampling_for_evaluation', default=False, type=str2bool, help='Enable domain-specific sampling for evaluation, if false use global sampling')
    parser.add_argument('--use_moe', default=True, type=str2bool, help='Enable/Disable MoE')
    parser.add_argument('--use_datasets', nargs='+', default=['beauty_5_5', 'games_5_5', 'ml-1m_5_5'], help='Datasets to use for multi-domain training')
    parser.add_argument('--use_domain_info', default=True, type=str2bool, help='Use domain info in MoE gating')
    parser.add_argument('--use_context', default=False, type=str2bool, help='Use context information in MoE gating')
    parser.add_argument('--use_rating_emb', default=True, type=str2bool, help='Use rating embedding to inform gating')
    parser.add_argument('--use_gated_fusion', default=True, type=str2bool, help='Use a gated mechanism to fuse rating embedding')
    parser.add_argument('--rating_pos_emb', default=False, type=str2bool, help='Add positional embedding to rating embeddings')
    parser.add_argument('--rating_strategy', default='fourier', type=str, 
                       choices=['simple', 'legacy', 'fourier'],
                       help='Strategy for rating information modeling: simple/legacy (backward compatibility), fourier (Fourier-based long/short-term feature extraction)')
    
    # 自适应配置参数
    parser.add_argument('--use_adaptive_rating_config', default=True, type=str2bool,
                       help='Enable adaptive rating configuration based on dataset characteristics')
    
    # 手动配置参数（当策略为fourier且自适应配置关闭时使用）
    parser.add_argument('--rating_num_frequencies', default=12, type=int, 
                       help='Number of frequency components for Fourier rating encoding (used when adaptive config disabled)')
    parser.add_argument('--rating_branch1_heads', default=1, type=int,
                       help='Number of attention heads for rating branch 1 (used when adaptive config disabled)')
    parser.add_argument('--rating_branch2_heads', default=1, type=int,
                       help='Number of attention heads for rating branch 2 (used when adaptive config disabled)')
    parser.add_argument('--moe_num_experts', default=4, type=int, help='Number of experts in MoE')
    parser.add_argument('--moe_k', default=2, type=int, help='Number of experts to use for each token')
    parser.add_argument('--moe_noisy_gating', default=True, type=str2bool, help='Use noisy gating in MoE')
    parser.add_argument('--moe_routing_strategy', default='shared_base', type=str, choices=['vanilla', 'shared_base'], help='MoE routing strategy')
    parser.add_argument('--moe_load_balancing', default=True, type=str2bool, help='Use load balancing in MoE')
    parser.add_argument('--moe_balance_loss_weight', default=0.01, type=float, help='Weight for MoE load balancing loss')
    parser.add_argument('--gate_temperature', default=1.0, type=float, help='Initial temperature for gate softmax')
    parser.add_argument('--min_gate_temperature', default=0.1, type=float, help='Minimum temperature for gate softmax')
    parser.add_argument('--temperature_decay', default=0.995, type=float, help='Temperature decay rate per step')
    parser.add_argument('--use_specialization_loss', default=True, type=str2bool, help='Enable specialization loss for expert specialization')
    parser.add_argument('--specialization_weight', default=0.01, type=float, help='Weight for specialization loss')
    parser.add_argument('--use_contrastive_loss', default=True, type=str2bool, help='Enable contrastive learning for expert specialization')
    parser.add_argument('--contrastive_weight', default=0.01, type=float, help='Weight for contrastive loss')
    parser.add_argument('--use_adaptive_balance', default=False, type=str2bool, help='Use adaptive load balancing based on specialization')
    parser.add_argument('--visualize', default=True, type=str2bool, help='Enable visualization of expert usage')
    parser.add_argument('--log_freq', default=100, type=int, help='Frequency of logging visualizations (in steps)')
    parser.add_argument('--tsne_log_freq', default=1, type=int, help='Frequency of logging t-SNE plots (in epochs) - DEPRECATED: now using performance-based triggering')
    parser.add_argument('--viz_on_improvement', default=True, type=str2bool, help='Only generate visualizations when model performance improves')
    parser.add_argument('--viz_force_epochs', default=[], type=lambda x: [int(i) for i in x.split(',')] if x else [], help='Force visualization on specific epochs (comma-separated), regardless of performance')
    parser.add_argument('--tsne_sample_size', default=512, type=int, help='Number of points to sample for t-SNE plot')
    # Enhanced visualization parameters
    parser.add_argument('--journal_style', default='science', type=str, 
                       choices=['nature', 'science', 'cell', 'high_quality'],
                       help='Journal style for enhanced visualizations')
    parser.add_argument('--viz_dpi', default=300, type=int, help='DPI for visualization outputs')
    parser.add_argument('--viz_format', default='png', type=str, 
                       choices=['pdf', 'png', 'svg', 'eps'],
                       help='Primary format for visualization exports')
    parser.add_argument('--save_publication_figs', default=False, type=str2bool,
                       help='Save publication-quality figures using enhanced visualization')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loading.')
    parser.add_argument('--swanlab_project', type=str, default='HAGMRec', help='SwanLab project name')
    parser.add_argument('--use_swanlab', default=True, type=str2bool, help='Enable/Disable SwanLab')
    args = parser.parse_args()
    
    # Check compatibility between rating strategy and other options
    args = check_rating_strategy_compatibility(args)
    
    return args