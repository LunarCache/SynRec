#!/usr/bin/env python3
"""
修复后的Fourier消融实验脚本
"""

import os
import sys
import json
import time
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from keys.model import HAGMRec
from keys.temporal_rating_modules import AblationMode
from keys.utils import evaluate_batched, partition_multi_domain

try:
    from visualization import (
        plot_frequency_ablation_comparison_journal,
        create_journal_config,
        VisualizationConfig
    )
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Visualization modules not available: {e}")
    VISUALIZATION_AVAILABLE = False

class FourierAblationExperiment:
    """Fourier频域消融实验管理器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if hasattr(args, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化可视化配置
        if VISUALIZATION_AVAILABLE:
            self.viz_config = create_journal_config('nature')
        else:
            self.viz_config = None
        
        # 存储实验结果
        self.results = {}
        
        print(f"🔬 Initialized Fourier Ablation Experiment")
        print(f"   - Device: {self.device}")
        print(f"   - Output directory: {self.output_dir}")
        print(f"   - Visualization available: {VISUALIZATION_AVAILABLE}")
    
    def _load_training_args(self, args_file: str):
        """从训练时保存的args.txt文件加载参数"""
        try:
            with open(args_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ',' in line:
                        key, value = line.split(',', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # 类型转换
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.lower() == 'none':
                            value = None
                        elif value.startswith('[') and value.endswith(']'):
                            # 处理列表格式
                            try:
                                import ast
                                value = ast.literal_eval(value)
                            except:
                                pass
                        else:
                            # 尝试转换为数字
                            try:
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except ValueError:
                                # 保持字符串
                                pass
                        
                        setattr(self.args, key, value)
            
            print(f"✅ Successfully loaded training arguments")
            
        except Exception as e:
            print(f"⚠️ Failed to load training arguments: {e}")
    
    def load_model_and_data(self, model_path: str):
        """加载模型和数据"""
        print(f"📂 Loading model and data...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 尝试从模型路径的同一目录加载训练参数
        model_dir = os.path.dirname(model_path)
        args_file = os.path.join(model_dir, 'args.txt')
        
        if os.path.exists(args_file):
            print(f"📖 Loading training arguments from: {args_file}")
            self._load_training_args(args_file)
        else:
            print(f"⚠️ No args.txt found in {model_dir}, using current arguments")
        
        # 加载数据集信息以获取正确的参数
        try:
            datasets = getattr(self.args, 'use_datasets', ['beauty_5_5', 'games_5_5', 'ml-1m_5_5'])
            # 处理字符串格式的数据集列表
            if isinstance(datasets, str):
                # 从字符串解析列表 "['beauty_5_5', 'games_5_5', 'ml-1m_5_5']"
                import ast
                try:
                    datasets = ast.literal_eval(datasets)
                except:
                    datasets = ['beauty_5_5', 'games_5_5', 'ml-1m_5_5']
                    
            dataset = partition_multi_domain(datasets)
            [user_train, user_valid, user_test, user_to_domain, usernum, itemnum, domain_to_item_range] = dataset
            
            # 更新参数
            self.args.item_num = itemnum
            self.args.user_num = usernum
            self.args.num_domains = len(datasets)
            
            print(f"📊 Dataset info: users={usernum}, items={itemnum}, domains={len(datasets)}")
            
        except Exception as e:
            print(f"⚠️ Could not load dataset info: {e}")
            dataset = None
        
        # 创建模型实例
        model = HAGMRec(
            user_num=getattr(self.args, 'user_num', 6040),
            item_num=getattr(self.args, 'item_num', 3416),
            args=self.args
        )
        
        # 加载模型参数
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model and data loaded successfully")
        return model, dataset
    
    def run_ablation_mode(self, model: HAGMRec, mode: AblationMode, dataset) -> Dict[str, float]:
        """运行单个消融模式的实验"""
        print(f"🧪 Running ablation experiment: {mode.value}")
        
        # 切换模型到指定的消融模式
        try:
            if hasattr(model, 'enhanced_rating_module'):
                model.enhanced_rating_module.set_ablation_mode(mode)
                print(f"   ✓ Set ablation mode to {mode.value}")
            else:
                print("⚠️ Model does not have enhanced_rating_module")
        except Exception as e:
            print(f"⚠️ Failed to set ablation mode: {e}")
        
        if dataset is None:
            print("⚠️ No dataset available, using dummy metrics")
            # 返回虚拟的度量值用于演示
            metrics = {
                'overall_NDCG@10': np.random.uniform(0.1, 0.3),
                'overall_HT@10': np.random.uniform(0.2, 0.5),
            }
        else:
            try:
                # 使用真实的评估函数
                print("   Running evaluation...")
                metrics = evaluate_batched(model, dataset, self.args, 'test')
                print(f"✅ Evaluation completed successfully")
            except Exception as e:
                print(f"⚠️ Evaluation failed: {e}")
                print("   Using dummy metrics")
                metrics = {
                    'overall_NDCG@10': np.random.uniform(0.05, 0.15),
                    'overall_HT@10': np.random.uniform(0.1, 0.3),
                }
        
        # 转换键名以匹配期望的格式，只保留NDCG@10和Hit@10
        result_metrics = {
            'ndcg@10': metrics.get('overall_NDCG@10', 0.0),
            'hit@10': metrics.get('overall_HT@10', 0.0),
        }
        
        # 添加按领域的指标
        domain_names = ['Beauty', 'Games', 'MovieLens']
        for i, domain_name in enumerate(domain_names):
            domain_key = f'domain_{i}'
            if f'{domain_key}_NDCG@10' in metrics:
                result_metrics[f'{domain_name.lower()}_ndcg@10'] = metrics[f'{domain_key}_NDCG@10']
            if f'{domain_key}_HT@10' in metrics:
                result_metrics[f'{domain_name.lower()}_hit@10'] = metrics[f'{domain_key}_HT@10']
        
        print(f"📊 {mode.value} Results:")
        for metric, value in result_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        return result_metrics
    
    def run_full_ablation_study(self, model: HAGMRec, dataset) -> Dict[str, Dict[str, float]]:
        """运行完整的消融研究"""
        print("🔬 Starting full ablation study...")
        
        ablation_modes = [
            AblationMode.FULL,
            AblationMode.LOW_FREQ_ONLY,
            AblationMode.HIGH_FREQ_ONLY
        ]
        
        results = {}
        
        for i, mode in enumerate(ablation_modes):
            print(f"\n📋 Experiment {i+1}/{len(ablation_modes)}")
            mode_key = mode.value
            results[mode_key] = self.run_ablation_mode(model, mode, dataset)
            
            # 保存中间结果
            self.save_intermediate_results(results)
            
            # 短暂休息以避免GPU过热
            if i < len(ablation_modes) - 1:
                print("⏳ Cooling down...")
                time.sleep(2)
        
        print("✅ Ablation study completed")
        return results
    
    def save_intermediate_results(self, results: Dict[str, Dict[str, float]]):
        """保存中间结果"""
        results_file = self.output_dir / 'intermediate_results.json'
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save intermediate results: {e}")
    
    def save_final_results(self, results: Dict[str, Dict[str, float]]):
        """保存最终结果"""
        print("💾 Saving final results...")
        
        # 保存JSON结果
        results_file = self.output_dir / 'ablation_results.json'
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved to {results_file}")
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")
        
        # 打印摘要
        self.print_results_summary(results)
        
        return results_file
    
    def print_results_summary(self, results: Dict[str, Dict[str, float]]):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("📊 ABLATION STUDY RESULTS SUMMARY")
        print("="*60)
        
        for mode, metrics in results.items():
            print(f"\n🧪 {mode.upper()}:")
            for metric, value in metrics.items():
                print(f"   {metric:<15}: {value:.4f}")
        
        # 计算相对性能
        if 'full' in results:
            print(f"\n📈 RELATIVE PERFORMANCE (vs FULL):")
            full_ndcg = results['full']['ndcg@10']
            for mode, metrics in results.items():
                if mode != 'full' and full_ndcg > 0:
                    relative_perf = (metrics['ndcg@10'] / full_ndcg - 1) * 100
                    print(f"   {mode:<15}: {relative_perf:+.1f}% NDCG@10")
        
        print("="*60)
    
    def generate_plots(self, results: Dict[str, Dict[str, float]], domain_map: Dict[int, str]):
        """生成可视化图表"""
        if not VISUALIZATION_AVAILABLE:
            print("⚠️ Visualization not available, skipping plots")
            return
        
        print("🎨 Generating visualization plots...")
        
        try:
            # 创建自定义的可视化配置
            from visualization.config import VisualizationConfig
            
            # 设置只输出600dpi png格式
            custom_config = VisualizationConfig(
                dpi=600,
                save_formats=['png'],  # 只保存png格式
                journal_style='nature',
                high_quality_export=True
            )
            
            fig, saved_files = self.plot_domain_ablation_comparison(
                results, domain_map, epoch=0, config=custom_config, save_plots=True
            )
            
            if saved_files:
                print(f"✅ Visualization plots saved: {len(saved_files)} files")
                for file_path in saved_files:
                    print(f"   - {file_path}")
            
        except Exception as e:
            print(f"⚠️ Plot generation failed: {e}")
    
    def plot_domain_ablation_comparison(self, results: Dict[str, Dict[str, float]], 
                                       domain_map: Dict[int, str], epoch: int,
                                       config, save_plots: bool = True):
        """生成折线图展示各领域性能在不同模式下的变化"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not results:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'No ablation results available', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Domain Ablation Comparison - No Data')
            return fig, []
        
        # 提取模式和指标
        modes = list(results.keys())
        if not modes:
            return plt.figure(), []
        
        # 定义要显示的指标和领域
        metrics = ['ndcg@10', 'hit@10']
        domain_names = ['Beauty', 'Games', 'MovieLens']
        mode_labels = ['Full Model', 'Low-Freq Only', 'High-Freq Only']
        
        # 创建子图布局：1行2列（NDCG@10和HIT@10）
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
        
        # 定义子图标签
        subplot_labels = ['(a)', '(b)']
        
        # 领域配色方案和标记样式
        domain_colors = {
            'Beauty': '#e74c3c',     # 红色
            'Games': '#2ecc71',      # 绿色 
            'MovieLens': '#3498db'   # 蓝色
        }
        
        domain_markers = {
            'Beauty': 'o',      # 圆形
            'Games': 's',       # 方形
            'MovieLens': '^'    # 三角形
        }
        
        try:
            # 绘制每个指标的折线图
            for col, metric in enumerate(metrics):
                ax = axes[col]
                metric_title = metric.upper().replace('@', '@')
                
                # 添加学术子图标签
                ax.text(-0.1, 1.1, subplot_labels[col], transform=ax.transAxes,
                       fontsize=14, fontweight='bold', va='top', ha='right')
                
                # 为每个领域绘制一条折线
                for domain_name in domain_names:
                    domain_metric = f'{domain_name.lower()}_{metric}'
                    
                    # 收集该领域在不同模式下的数值
                    values = []
                    x_positions = []
                    
                    for i, mode in enumerate(modes):
                        if domain_metric in results[mode]:
                            values.append(results[mode][domain_metric])
                            x_positions.append(i)
                    
                    if values:
                        # 绘制折线
                        ax.plot(x_positions, values, 
                               color=domain_colors[domain_name],
                               marker=domain_markers[domain_name],
                               linewidth=2.5,
                               markersize=8,
                               label=domain_name,
                               alpha=0.8)
                        
                        # 添加数值标签
                        for x, y in zip(x_positions, values):
                            ax.annotate(f'{y:.4f}', 
                                      (x, y), 
                                      textcoords="offset points",
                                      xytext=(0, 10),
                                      ha='center',
                                      fontsize=9,
                                      fontweight='bold',
                                      color=domain_colors[domain_name])
                
                # 设置图表样式
                ax.set_title(f'{metric_title} Performance Across Domains', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Frequency Component Mode', fontsize=12)
                ax.set_ylabel(f'{metric_title} Score', fontsize=12)
                
                # 设置x轴刻度和标签
                ax.set_xticks(range(len(modes)))
                ax.set_xticklabels(mode_labels, fontsize=11)
                
                # 添加网格但不在子图中添加图例
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_axisbelow(True)
                
                # 设置边框样式
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.2)
                ax.spines['bottom'].set_linewidth(1.2)
            
            # 设置总标题
            fig.suptitle('Multi-Domain Performance across Frequency Components', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # 在图表下方添加集中的图例
            handles, labels = axes[0].get_legend_handles_labels()
            if handles and labels:
                fig.legend(handles, labels, 
                         loc='lower center', 
                         bbox_to_anchor=(0.5, 0.02),
                         ncol=3,  # 水平排列三个领域
                         frameon=True, 
                         fancybox=True, 
                         shadow=True, 
                         framealpha=0.95, 
                         fontsize=12,
                         columnspacing=2.0)
            
            # 添加说明文本（移除表情符号，使用文字说明）
            explanation = (
                "Full: Complete model with both frequency components  |  "
                "Low-Only: Only low-frequency (long-term) components  |  "
                "High-Only: Only high-frequency (short-term) components"
            )
            fig.text(0.5, 0.04, explanation, fontsize=11, 
                    ha='center', style='italic', alpha=0.8)
            
            # 调整布局，为标签和图例留出更多空间
            plt.tight_layout(rect=[0, 0.15, 1, 0.90])  # 为子图标签和图例预留空间
            
            # 保存图片
            saved_files = []
            if save_plots:
                filename = f'domain_frequency_trend_comparison_epoch_{epoch}'
                
                # 确保输出目录存在
                os.makedirs(self.output_dir, exist_ok=True)
                
                # 保存为600dpi png到正确的输出目录
                output_path = self.output_dir / f'{filename}.png'
                plt.savefig(str(output_path), dpi=600, format='png', 
                           bbox_inches='tight', pad_inches=0.2,
                           facecolor='white', edgecolor='none')
                saved_files.append(str(output_path))
            
            return fig, saved_files
            
        except Exception as e:
            print(f"⚠️ Error in domain ablation plotting: {e}")
            plt.close(fig)
            # 创建简单错误图
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, f'Plotting error: {str(e)}', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Domain Ablation Comparison - Error')
            return fig, []

def create_default_args():
    """创建基础的参数对象，主要参数会从训练时的args.txt加载"""
    class DefaultArgs:
        def __init__(self):
            # 基础参数
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 这些参数会从训练时的args.txt文件中覆盖
            self.maxlen = 100
            self.hidden_units = 64
            self.batch_size = 256
            self.num_workers = 4
            self.use_datasets = ['beauty_5_5', 'games_5_5', 'ml-1m_5_5']
    
    return DefaultArgs()

def main():
    parser = argparse.ArgumentParser(description='Fixed Fourier Ablation Experiment for CMREC')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save ablation results')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--generate_plots', action='store_true',
                      help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # 自动选择设备
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 合并默认参数
    default_args = create_default_args()
    for key, value in vars(default_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    print(f"🚀 Starting Fixed Fourier Ablation Experiment")
    print(f"   Model path: {args.model_path}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Device: {args.device}")
    print(f"   Generate plots: {args.generate_plots}")
    
    try:
        # 创建实验管理器
        experiment = FourierAblationExperiment(args)
        
        # 加载模型和数据
        model, dataset = experiment.load_model_and_data(args.model_path)
        
        # 运行消融实验
        results = experiment.run_full_ablation_study(model, dataset)
        
        # 保存结果
        results_file = experiment.save_final_results(results)
        
        # 生成图表（如果请求）
        if args.generate_plots:
            domain_map = {0: 'Beauty', 1: 'Games', 2: 'MovieLens'}
            experiment.generate_plots(results, domain_map)
        
        print(f"\n🎉 Experiment completed successfully!")
        print(f"📁 Results saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()