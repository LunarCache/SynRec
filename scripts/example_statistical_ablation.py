#!/usr/bin/env python3
"""
Fourier消融实验统计分析示例
Example Usage of Statistical Fourier Ablation Experiment

此示例展示如何使用 run_fourier_ablation_statistical.py 进行统计分析
"""

import os
import subprocess
from pathlib import Path

def main():
    print("🔬 Fourier消融实验统计分析示例")
    print("=" * 50)
    
    # 示例配置
    # 注意：请根据实际情况修改这些路径
    model_path = "exp/beauty_fourier_temporal/model_best.pth"  # 示例模型路径
    output_dir = "exp/statistical_ablation_results"
    
    print(f"📁 模型路径: {model_path}")
    print(f"📁 输出目录: {output_dir}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"⚠️ 模型文件不存在: {model_path}")
        print("请修改 model_path 变量为实际的模型文件路径")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🚀 开始统计消融实验...")
    
    # 示例1: 快速测试（少量运行）
    print("\n📋 示例1: 快速测试（10次运行）")
    cmd_quick = [
        "python", "run_fourier_ablation_statistical.py",
        "--model_path", model_path,
        "--output_dir", f"{output_dir}/quick_test",
        "--num_runs", "10",
        "--generate_plots"
    ]
    
    print("命令:", " ".join(cmd_quick))
    print("运行时间预计: 5-10分钟")
    
    # 示例2: 标准分析（推荐配置）
    print("\n📋 示例2: 标准分析（30次运行，推荐）")
    cmd_standard = [
        "python", "run_fourier_ablation_statistical.py",
        "--model_path", model_path,
        "--output_dir", f"{output_dir}/standard_analysis",
        "--num_runs", "30",
        "--alpha", "0.05",
        "--effect_size_threshold", "0.5",
        "--correction_method", "fdr_bh",
        "--generate_plots"
    ]
    
    print("命令:", " ".join(cmd_standard))
    print("运行时间预计: 15-30分钟")
    
    # 示例3: 严格分析（高标准）
    print("\n📋 示例3: 严格分析（50次运行，严格标准）")
    cmd_strict = [
        "python", "run_fourier_ablation_statistical.py",
        "--model_path", model_path,
        "--output_dir", f"{output_dir}/strict_analysis",
        "--num_runs", "50",
        "--alpha", "0.01",
        "--effect_size_threshold", "0.8",
        "--correction_method", "bonferroni",
        "--generate_plots"
    ]
    
    print("命令:", " ".join(cmd_strict))
    print("运行时间预计: 30-60分钟")
    
    print("\n" + "=" * 50)
    print("💡 使用建议:")
    print("1. 首次使用建议运行示例1进行快速测试")
    print("2. 正式分析建议使用示例2的标准配置")
    print("3. 发表论文时建议使用示例3的严格配置")
    print("4. 根据计算资源调整 --num_runs 参数")
    print("5. 使用 --device cpu 如果GPU内存不足")
    
    print("\n📊 输出文件说明:")
    print("- statistical_ablation_results.json: 完整统计结果")
    print("- statistical_report.md: 可读性报告")
    print("- descriptive_statistics.csv: 描述性统计")
    print("- significance_tests.csv: 显著性检验结果")
    print("- *.png: 统计可视化图表")
    
    # 交互式选择运行
    print("\n🎯 是否要运行某个示例？")
    print("1. 快速测试")
    print("2. 标准分析")
    print("3. 严格分析")
    print("0. 仅显示命令（不运行）")
    
    try:
        choice = input("请选择 (0-3): ").strip()
        
        if choice == "1":
            print("🚀 运行快速测试...")
            subprocess.run(cmd_quick)
        elif choice == "2":
            print("🚀 运行标准分析...")
            subprocess.run(cmd_standard)
        elif choice == "3":
            print("🚀 运行严格分析...")
            subprocess.run(cmd_strict)
        elif choice == "0":
            print("✅ 命令已显示，您可以手动运行")
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n❌ 用户取消")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
    
    print("\n🎉 示例演示完成！")

if __name__ == "__main__":
    main()