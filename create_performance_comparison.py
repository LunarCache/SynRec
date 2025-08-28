#!/usr/bin/env python3
"""
期刊级别性能对比可视化脚本

从LaTeX表格数据生成专业性能对比图，支持：
1. 多数据集NDCG@10和HR@10指标对比
2. 期刊级别的视觉标准
3. 多种输出格式(PDF, PNG)
4. 自动化的最佳性能突出显示
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
from dataclasses import dataclass

# 导入项目可视化模块
from visualization.journal_styles import apply_journal_style
from visualization.config import VisualizationConfig, create_journal_config
from visualization.enhanced_plots import EnhancedVisualization
from visualization.color_schemes import JournalColorSchemes

@dataclass
class PerformanceData:
    """性能数据容器"""
    methods: List[str]
    datasets: List[str]
    metrics: List[str]
    data: Dict[str, Dict[str, Dict[str, float]]]  # method -> dataset -> metric -> value
    best_values: Dict[str, Dict[str, str]]  # dataset -> metric -> best_method

class PerformanceVisualizer:
    """性能对比可视化器"""

    def __init__(self, journal_style: str = 'nature'):
        self.journal_style = journal_style
        self.config = create_journal_config(journal_style)
        self.visualizer = EnhancedVisualization(self.config)
        self.color_schemes = JournalColorSchemes()

        # 设置输出目录
        self.output_dir = Path('/home/wzc/workspace/github/SynRec/mypaper/images')
        self.output_dir.mkdir(exist_ok=True)

    def extract_table_data(self, latex_content: str) -> PerformanceData:
        """
        从LaTeX表格内容中提取性能数据

        Args:
            latex_content: LaTeX表格字符串

        Returns:
            PerformanceData对象
        """
        # 手动定义数据（基于提供的表格内容）
        datasets = ['Amazon Beauty', 'Games', 'MovieLens-1M']
        metrics = ['NDCG@10', 'HR@10']

        # 方法名称列表（按表格顺序）
        methods = [
            'PopRec', 'BPR', 'FMC', 'FPMC', 'TransRec', 'GRU4Rec', 'GRU4Rec+', 'Caser',
            'SASRec', 'π-Net', 'MMoE', 'BERT4Rec', 'Transformers4Rec', 'FairSR', 'MIA-SR', 'INSPEQ', 'SynRec'
        ]

        # 性能数据（按方法、数据集、指标的顺序）
        raw_data = {
            'PopRec': {
                'Amazon Beauty': {'NDCG@10': 0.2277, 'HR@10': 0.4003},
                'Games': {'NDCG@10': 0.2779, 'HR@10': 0.4724},
                'MovieLens-1M': {'NDCG@10': 0.2377, 'HR@10': 0.4329}
            },
            'BPR': {
                'Amazon Beauty': {'NDCG@10': 0.2183, 'HR@10': 0.3775},
                'Games': {'NDCG@10': 0.2875, 'HR@10': 0.4853},
                'MovieLens-1M': {'NDCG@10': 0.3287, 'HR@10': 0.5781}
            },
            'FMC': {
                'Amazon Beauty': {'NDCG@10': 0.2477, 'HR@10': 0.3771},
                'Games': {'NDCG@10': 0.4456, 'HR@10': 0.6358},
                'MovieLens-1M': {'NDCG@10': 0.4676, 'HR@10': 0.6986}
            },
            'FPMC': {
                'Amazon Beauty': {'NDCG@10': 0.2891, 'HR@10': 0.4310},
                'Games': {'NDCG@10': 0.4680, 'HR@10': 0.6802},
                'MovieLens-1M': {'NDCG@10': 0.5176, 'HR@10': 0.7599}
            },
            'TransRec': {
                'Amazon Beauty': {'NDCG@10': 0.3020, 'HR@10': 0.4607},
                'Games': {'NDCG@10': 0.4557, 'HR@10': 0.6838},
                'MovieLens-1M': {'NDCG@10': 0.3969, 'HR@10': 0.6413}
            },
            'GRU4Rec': {
                'Amazon Beauty': {'NDCG@10': 0.1203, 'HR@10': 0.2125},
                'Games': {'NDCG@10': 0.1837, 'HR@10': 0.2938},
                'MovieLens-1M': {'NDCG@10': 0.3381, 'HR@10': 0.5581}
            },
            'GRU4Rec+': {
                'Amazon Beauty': {'NDCG@10': 0.2556, 'HR@10': 0.3949},
                'Games': {'NDCG@10': 0.4759, 'HR@10': 0.6599},
                'MovieLens-1M': {'NDCG@10': 0.5513, 'HR@10': 0.7501}
            },
            'Caser': {
                'Amazon Beauty': {'NDCG@10': 0.2547, 'HR@10': 0.4264},
                'Games': {'NDCG@10': 0.3214, 'HR@10': 0.5282},
                'MovieLens-1M': {'NDCG@10': 0.5538, 'HR@10': 0.7886}
            },
            'SASRec': {
                'Amazon Beauty': {'NDCG@10': 0.3476, 'HR@10': 0.5205},
                'Games': {'NDCG@10': 0.4952, 'HR@10': 0.7144},
                'MovieLens-1M': {'NDCG@10': 0.5786, 'HR@10': 0.8127}
            },
            'π-Net': {
                'Amazon Beauty': {'NDCG@10': 0.2956, 'HR@10': 0.4521},
                'Games': {'NDCG@10': 0.4234, 'HR@10': 0.6234},
                'MovieLens-1M': {'NDCG@10': 0.4876, 'HR@10': 0.7123}
            },
            'MMoE': {
                'Amazon Beauty': {'NDCG@10': 0.3156, 'HR@10': 0.4678},
                'Games': {'NDCG@10': 0.4892, 'HR@10': 0.7056},
                'MovieLens-1M': {'NDCG@10': 0.5234, 'HR@10': 0.7634}
            },
            'BERT4Rec': {
                'Amazon Beauty': {'NDCG@10': 0.3089, 'HR@10': 0.4732},
                'Games': {'NDCG@10': 0.5124, 'HR@10': 0.7289},
                'MovieLens-1M': {'NDCG@10': 0.5767, 'HR@10': 0.8089}
            },
            'Transformers4Rec': {
                'Amazon Beauty': {'NDCG@10': 0.2533, 'HR@10': 0.4344},
                'Games': {'NDCG@10': 0.3312, 'HR@10': 0.5311},
                'MovieLens-1M': {'NDCG@10': 0.3921, 'HR@10': 0.4723}
            },
            'FairSR': {
                'Amazon Beauty': {'NDCG@10': 0.2653, 'HR@10': 0.4723},
                'Games': {'NDCG@10': 0.4684, 'HR@10': 0.7312},
                'MovieLens-1M': {'NDCG@10': 0.5104, 'HR@10': 0.8137}
            },
            'MIA-SR': {
                'Amazon Beauty': {'NDCG@10': 0.3421, 'HR@10': 0.5236},
                'Games': {'NDCG@10': 0.5571, 'HR@10': 0.7823},
                'MovieLens-1M': {'NDCG@10': 0.6082, 'HR@10': 0.8277}
            },
            'INSPEQ': {
                'Amazon Beauty': {'NDCG@10': 0.3477, 'HR@10': 0.5322},
                'Games': {'NDCG@10': 0.5594, 'HR@10': 0.7951},
                'MovieLens-1M': {'NDCG@10': 0.6253, 'HR@10': 0.8316}
            },
            'SynRec': {
                'Amazon Beauty': {'NDCG@10': 0.3742, 'HR@10': 0.5704},
                'Games': {'NDCG@10': 0.5846, 'HR@10': 0.8210},
                'MovieLens-1M': {'NDCG@10': 0.7846, 'HR@10': 0.9794}
            }
        }

        # 找出每个数据集和指标的最佳方法
        best_values = {dataset: {metric: '' for metric in metrics} for dataset in datasets}

        for dataset in datasets:
            for metric in metrics:
                best_method = ''
                best_value = -1
                for method in methods:
                    value = raw_data[method][dataset][metric]
                    if value > best_value:
                        best_value = value
                        best_method = method
                best_values[dataset][metric] = best_method

        return PerformanceData(
            methods=methods,
            datasets=datasets,
            metrics=metrics,
            data=raw_data,
            best_values=best_values
        )

    def create_comparison_plot(self, perf_data: PerformanceData) -> Tuple[plt.Figure, List[str]]:
        """
        创建性能对比图

        Args:
            perf_data: 性能数据

        Returns:
            (figure, saved_files) 元组
        """
        # 设置布局：3个数据集 x 2个指标 = 6个子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=600)
        fig.patch.set_facecolor('white')

        # 获取配色方案
        colors = self.color_schemes.get_journal_palette(self.journal_style, len(perf_data.methods))

        # 为每个数据集和指标创建子图
        for row, metric in enumerate(perf_data.metrics):
            for col, dataset in enumerate(perf_data.datasets):
                ax = axes[row, col]

                # 准备数据
                methods = perf_data.methods
                values = [perf_data.data[method][dataset][metric] for method in methods]

                # 找出最佳性能
                best_method = perf_data.best_values[dataset][metric]
                best_index = methods.index(best_method)

                # 绘制柱状图
                bars = ax.bar(range(len(methods)), values, color=colors[:len(methods)],
                             alpha=0.8, edgecolor='white', linewidth=0.5)

                # 高亮最佳性能
                bars[best_index].set_color('#FF6B6B')  # 醒目的红色
                bars[best_index].set_edgecolor('#CC5555')
                bars[best_index].set_linewidth(1.5)

                # 设置标签
                ax.set_title(f'{dataset}\n{metric}', fontsize=11, fontweight='bold', pad=10)
                ax.set_ylabel('Score', fontsize=9)

                # 设置x轴
                ax.set_xticks(range(len(methods)))
                ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)

                # 添加网格
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_axisbelow(True)

                # 添加数值标签
                for i, (bar, value) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=7,
                           fontweight='bold' if i == best_index else 'normal')

        # 设置整体标题
        fig.suptitle('Performance Comparison Across Datasets',
                    fontsize=14, fontweight='bold', y=0.98)

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # 保存图片
        saved_files = []
        for fmt in ['pdf', 'png']:
            filename = f'performance_comparison.{fmt}'
            filepath = self.output_dir / filename
            fig.savefig(filepath, format=fmt, dpi=600, bbox_inches='tight', pad_inches=0.1)
            saved_files.append(str(filepath))
            print(f"已保存: {filepath}")

        return fig, saved_files

    def create_summary_table_plot(self, perf_data: PerformanceData) -> Tuple[plt.Figure, List[str]]:
        """
        创建汇总表格图

        Args:
            perf_data: 性能数据

        Returns:
            (figure, saved_files) 元组
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=600)
        fig.patch.set_facecolor('white')

        # 准备数据用于表格
        methods = perf_data.methods
        datasets = perf_data.datasets
        metrics = perf_data.metrics

        # 创建表格数据
        table_data = []
        for method in methods:
            row = [method]
            for dataset in datasets:
                for metric in metrics:
                    value = perf_data.data[method][dataset][metric]
                    is_best = perf_data.best_values[dataset][metric] == method
                    row.append('.4f')
            table_data.append(row)

        # 列名
        columns = ['Method']
        for dataset in datasets:
            for metric in metrics:
                columns.append(f'{dataset}\n{metric}')

        # 创建表格
        table = ax.table(cellText=table_data,
                        colLabels=columns,
                        cellLoc='center',
                        loc='center',
                        colColours=['lightgray'] * len(columns))

        # 样式化表格
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # 设置单元格颜色
        for i, method in enumerate(methods):
            for j, dataset in enumerate(datasets):
                for k, metric in enumerate(metrics):
                    cell_idx = i + 1  # +1 因为第一行是标题
                    col_idx = j * 2 + k + 1  # +1 因为第一列是方法名

                    is_best = perf_data.best_values[dataset][metric] == method
                    if is_best:
                        table[(cell_idx, col_idx)].set_facecolor('#FFE6E6')  # 浅红色背景
                        table[(cell_idx, col_idx)].set_text_props(weight='bold')

        ax.axis('off')
        ax.set_title('Performance Summary Table\n(Best results highlighted in red)',
                    fontsize=12, fontweight='bold', pad=20)

        # 保存图片
        saved_files = []
        for fmt in ['pdf', 'png']:
            filename = f'performance_summary_table.{fmt}'
            filepath = self.output_dir / filename
            fig.savefig(filepath, format=fmt, dpi=600, bbox_inches='tight', pad_inches=0.1)
            saved_files.append(str(filepath))
            print(f"已保存: {filepath}")

        return fig, saved_files

def main():
    """主函数"""

    # LaTeX表格内容（从用户提供的表格中提取）
    latex_content = r"""
\begin{table*}[!htbp]
\centering
\caption{Performance Comparison Across Individual Datasets. Best results in \textbf{bold}, second-best \underline{underlined}.}
\label{tab:overall_performance}
\resizebox{\textwidth}{!}{
\begin{tabular}{l|cc|cc|cc}
\hline
\multirow{2}{*}{\textbf{Method}} & \multicolumn{2}{c|}{\textbf{Amazon Beauty}} & \multicolumn{2}{c|}{\textbf{Games}} & \multicolumn{2}{c}{\textbf{MovieLens-1M}} \\
\cline{2-7}
 & NDCG@10 & HR@10 & NDCG@10 & HR@10 & NDCG@10 & HR@10 \\
\hline
PopRec & 0.2277 & 0.4003 & 0.2779 & 0.4724 & 0.2377 & 0.4329 \\
BPR & 0.2183 & 0.3775 & 0.2875 & 0.4853 & 0.3287 & 0.5781 \\
FMC & 0.2477 & 0.3771 & 0.4456 & 0.6358 & 0.4676 & 0.6986 \\
FPMC & 0.2891 & 0.4310 & 0.4680 & 0.6802 & 0.5176 & 0.7599 \\
TransRec & 0.3020 & 0.4607 & 0.4557 & 0.6838 & 0.3969 & 0.6413 \\
GRU4Rec & 0.1203 & 0.2125 & 0.1837 & 0.2938 & 0.3381 & 0.5581 \\
GRU4Rec$^+$ & 0.2556 & 0.3949 & 0.4759 & 0.6599 & 0.5513 & 0.7501 \\
Caser & 0.2547 & 0.4264 & 0.3214 & 0.5282 & 0.5538 & 0.7886 \\
SASRec & 0.3476 & 0.5205 & 0.4952 & 0.7144 & 0.5786 & 0.8127 \\
$\pi$-Net & 0.2956 & 0.4521 & 0.4234 & 0.6234 & 0.4876 & 0.7123 \\
MMoE & 0.3156 & 0.4678 & 0.4892 & 0.7056 & 0.5234 & 0.7634 \\
BERT4Rec & 0.3089 & 0.4732 & 0.5124 & 0.7289 & 0.5767 & 0.8089 \\
Transformers4Rec & 0.2533 & 0.4344 & 0.3312 & 0.5311 & 0.3921 & 0.4723 \\
FairSR & 0.2653 & 0.4723 & 0.4684 & 0.7312 & 0.5104 & 0.8137 \\
MIA-SR & 0.3421 & 0.5236 & 0.5571 & 0.7823 & 0.6082 & 0.8277 \\
INSPEQ & \underline{0.3477} & \underline{0.5322} & \underline{0.5594} & \underline{0.7951} & \underline{0.6253} & \underline{0.8316} \\
\hline
\textbf{SynRec} & \textbf{0.3742} & \textbf{0.5704} & \textbf{0.5846} & \textbf{0.8210} & \textbf{0.7846} & \textbf{0.9794} \\
\hline
\end{tabular}
}
\end{table*}
"""

    # 创建可视化器
    visualizer = PerformanceVisualizer(journal_style='nature')

    # 提取数据
    perf_data = visualizer.extract_table_data(latex_content)

    print("提取到的数据:")
    print(f"方法数量: {len(perf_data.methods)}")
    print(f"数据集: {perf_data.datasets}")
    print(f"指标: {perf_data.metrics}")
    print(f"方法列表: {perf_data.methods}")

    # 生成对比图
    print("\n正在生成性能对比图...")
    fig1, files1 = visualizer.create_comparison_plot(perf_data)

    # 生成汇总表格
    print("\n正在生成汇总表格...")
    fig2, files2 = visualizer.create_summary_table_plot(perf_data)

    print("\n生成完成！输出文件:")
    print("\n对比图:")
    for f in files1:
        print(f"  {f}")

    print("\n汇总表格:")
    for f in files2:
        print(f"  {f}")

    print(f"\n所有文件保存在: {visualizer.output_dir}")

    return perf_data

if __name__ == "__main__":
    perf_data = main()
