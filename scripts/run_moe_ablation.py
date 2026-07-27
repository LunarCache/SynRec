import subprocess
import os
import re
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
from datetime import datetime

# 复用项目统一的期刊样式，保证与其它重绘图风格一致（仅排版，不改数据）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from visualization.journal_styles import apply_journal_style as _apply_journal_style
except Exception:
    _apply_journal_style = None

# --- Configuration ---
SUMMARY_DIR = "exp/moe_ablation_summary"
STATE_FILE = os.path.join(SUMMARY_DIR, "ablation_state.json")
RESULTS_CSV = os.path.join(SUMMARY_DIR, "ablation_results.csv")

DATASETS_LIST = ["beauty_5_5", "games_5_5", "ml-1m_5_5"]
DATASETS_STR = " ".join(DATASETS_LIST)
DATASETS_JOINED = "-".join(DATASETS_LIST) 

BASE_CMD = f"python main.py --use_datasets {DATASETS_STR} --num_epochs 100 --batch_size 1024 --rating_strategy temporal_fourier"

experiments = [
    {"id": "Base", "moe_num_experts": 4, "hidden_units": 64},
    {"id": "Exp5", "moe_num_experts": 5, "hidden_units": 64},
    {"id": "Exp6", "moe_num_experts": 6, "hidden_units": 64},
    {"id": "Exp7", "moe_num_experts": 7, "hidden_units": 64},
    {"id": "Exp8", "moe_num_experts": 8, "hidden_units": 64},
    {"id": "Dim32", "moe_num_experts": 4, "hidden_units": 32},
    {"id": "Dim48", "moe_num_experts": 4, "hidden_units": 48},
    {"id": "Dim96", "moe_num_experts": 4, "hidden_units": 96},
    {"id": "Dim128", "moe_num_experts": 4, "hidden_units": 128}
]

# --- State Management ---
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ State file corrupted, starting fresh.")
    return {}

def save_state(state):
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def update_results_csv(state):
    rows = []
    for exp_id, data in state.items():
        if data['status'] == 'done' and 'metrics' in data:
            row = {'id': exp_id}
            config = next((e for e in experiments if e['id'] == exp_id), None)
            if config:
                row.update(config)
                row.update(data['metrics'])
                rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        cols = ['id', 'moe_num_experts', 'hidden_units', 'NDCG@10', 'HR@10']
        for c in df.columns:
            if c not in cols: cols.append(c)
        df = df[cols] if set(cols).issubset(df.columns) else df
        
        df.to_csv(RESULTS_CSV, index=False)
        print(f"📊 Results updated: {RESULTS_CSV}")
        return df
    return None

# --- Parsing Logic ---
def parse_best_metrics_from_log(train_dir):
    log_file = os.path.join(train_dir, "log.txt")
    
    if not os.path.exists(log_file):
        print(f"   (Debug: Log file not found at {log_file})")
        return None, None

    best_ndcg = 0.0
    best_hr = 0.0
    found_data = False
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                
                found_data = True
                test_str = parts[2]
                metrics = {}
                for item in test_str.split(','):
                    if ':' in item:
                        k, v = item.split(':')
                        try: metrics[k] = float(v)
                        except ValueError: pass
                
                ndcg = metrics.get('overall_NDCG@10', 0.0)
                hr = metrics.get('overall_HT@10', 0.0)
                
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_hr = hr
                    
    except Exception as e:
        print(f"❌ Error parsing log {log_file}: {e}")
        return None, None
        
    if not found_data:
        print(f"   (Debug: Log file found but contained no valid data rows)")
        return None, None
        
    return best_ndcg, best_hr

# --- Execution Logic ---
def run_experiment(exp_config, state):
    exp_id = exp_config["id"]
    experts = exp_config["moe_num_experts"]
    hidden = exp_config["hidden_units"]
    
    exp_suffix = f"ablation_{exp_id}_E{experts}_H{hidden}"
    real_train_dir = os.path.join("exp", f"{DATASETS_JOINED}_{exp_suffix}")
    
    if exp_id in state:
        if state[exp_id]['status'] == 'done':
            print(f"⏭️  Skipping {exp_id} (Completed)")
            return state[exp_id]
            
    print(f"\n{'='*60}")
    print(f"🚀 Starting Experiment: {exp_id}")
    print(f"   Experts: {experts} | Hidden: {hidden}")
    print(f"   Log: {SUMMARY_DIR}/{exp_id}_stdout.log")
    print(f"{'='*60}\n")
    
    state[exp_id] = {
        "status": "running",
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": exp_config
    }
    save_state(state)
    
    # PYTHONUNBUFFERED is still good for the log file
    cmd = f"PYTHONUNBUFFERED=1 {BASE_CMD} --train_dir {exp_suffix} --moe_num_experts {experts} --hidden_units {hidden}"
    
    debug_log = os.path.join(SUMMARY_DIR, f"{exp_id}_stdout.log")
    start_time = time.time()
    
    try:
        # Using subprocess.run directly to file, cleaner than Popen loop if we don't print
        with open(debug_log, "w") as log_file:
            process = subprocess.run(
                cmd, 
                shell=True, 
                stdout=log_file, 
                stderr=subprocess.STDOUT
            )
        
        if process.returncode != 0:
            print(f"\n❌ Experiment {exp_id} Failed! Return Code: {process.returncode}")
            state[exp_id]['status'] = 'failed'
            save_state(state)
            return None
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted!")
        state[exp_id]['status'] = 'interrupted'
        save_state(state)
        raise
        
    duration = time.time() - start_time
    
    ndcg, hr = parse_best_metrics_from_log(real_train_dir)
    
    if ndcg is not None:
        print(f"\n✅ Finished {exp_id}. Best NDCG: {ndcg:.4f}, HR: {hr:.4f}")
        state[exp_id] = {
            "status": "done",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "metrics": {
                "NDCG@10": ndcg,
                "HR@10": hr
            },
            "config": exp_config
        }
        save_state(state)
        update_results_csv(state)
        return state[exp_id]
    else:
        print(f"\n❌ Verification Failed for {exp_id}. Log file not found or empty at: {real_train_dir}")
        state[exp_id]['status'] = 'failed_verification'
        save_state(state)
        return None

# --- Plotting Helper ---
def plot_results(df, output_dir=SUMMARY_DIR):
    if df is None or df.empty: return

    # 统一期刊风格（与其它重绘图一致：science 样式 + 放大字体）。仅排版，不改数据。
    if _apply_journal_style is not None:
        _apply_journal_style('science')
    else:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            plt.style.use('ggplot')
    plt.rcParams.update({
        'font.size': 15, 'axes.titlesize': 18, 'axes.labelsize': 16,
        'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 14,
        'lines.linewidth': 2.2, 'lines.markersize': 9,
        'figure.dpi': 600, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'axes.grid': True, 'grid.alpha': 0.3,
    })

    # 画布宽高（11 x 5.0，宽高比≈2.2，子图更窄、与正文宽度配合 0.85\textwidth）
    fig, (ax_exp, ax_dim) = plt.subplots(1, 2, figsize=(11, 5.0))
    
    def add_dual_axis_plot(ax1, df_sub, x_col, x_label, title):
        if df_sub.empty:
            ax1.text(0.5, 0.5, "No Data", ha='center')
            return None, None
        
        # NDCG (Left Axis)
        color_ndcg = 'tab:blue'
        ax1.set_xlabel(x_label, fontsize=18)
        ax1.set_ylabel('NDCG@10', color=color_ndcg, fontsize=16)
        l1 = ax1.plot(df_sub[x_col], df_sub["NDCG@10"], marker='o', color=color_ndcg, label='NDCG@10', linewidth=2.2)
        ax1.tick_params(axis='y', labelcolor=color_ndcg)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # HR (Right Axis)
        ax2 = ax1.twinx()
        # science 样式默认隐藏右脊柱，但双轴图的右轴需要显示
        ax2.spines['right'].set_visible(True)
        ax2.spines['top'].set_visible(False)
        color_hr = 'tab:orange'
        ax2.set_ylabel('Hit Rate@10', color=color_hr, fontsize=16)
        l2 = ax2.plot(df_sub[x_col], df_sub["HR@10"], marker='s', color=color_hr, label='HR@10', linewidth=2.2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color_hr)
        ax2.grid(False)
        
        # Collect handles for a single shared legend placed at the figure bottom
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax1.set_title(title, fontsize=18, fontweight='bold')

        # Explicitly set x-ticks to match data points for clarity
        unique_x = sorted(df_sub[x_col].unique())
        ax1.set_xticks(unique_x)
        ax1.set_xticklabels(unique_x)
        return lns, labs

    # 1. Plot Experts (Left Subplot)
    df_experts = df[df["hidden_units"] == 64].copy().sort_values("moe_num_experts")
    leg_handles, leg_labels = None, None
    if not df_experts.empty:
        # Map Total M to Shared Ns (Ns = M - 3)
        df_experts["num_shared"] = df_experts["moe_num_experts"] - 3
        leg_handles, leg_labels = add_dual_axis_plot(ax_exp, df_experts, "num_shared", "Number of Shared Experts ($N_s$)", "Impact of Shared Expert Capacity")

    # 2. Plot Hidden Units (Right Subplot)
    df_hidden = df[df["moe_num_experts"] == 4].sort_values("hidden_units")
    h2, l2_ = add_dual_axis_plot(ax_dim, df_hidden, "hidden_units", "Hidden Dimension ($d$)", "Impact of Expert Dimensionality")
    if leg_handles is None:
        leg_handles, leg_labels = h2, l2_

    # Reserve bottom space and place one shared legend below both subplots
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    if leg_handles:
        fig.legend(leg_handles, leg_labels, loc='lower center', ncol=2,
                   frameon=False, bbox_to_anchor=(0.5, 0.01))
    os.makedirs(output_dir, exist_ok=True)
    combined_filename = "moe_configuration_ablation.png"
    out_path = os.path.join(output_dir, combined_filename)
    plt.savefig(out_path, dpi=600)
    print(f"📈 Saved combined plot: {out_path}")
    plt.close()

def _parse_cli():
    p = argparse.ArgumentParser(description="MoE configuration ablation runner / plotter")
    p.add_argument('--plot_only', action='store_true',
                   help='不训练，直接从已有 CSV 读取结果并绘图')
    p.add_argument('--from_csv', type=str, default=RESULTS_CSV,
                   help='结果 CSV 路径 (列: id,moe_num_experts,hidden_units,NDCG@10,HR@10)')
    p.add_argument('--output_dir', type=str, default=SUMMARY_DIR,
                   help='图片输出目录')
    return p.parse_args()

if __name__ == "__main__":
    cli = _parse_cli()

    # 仅绘图模式：直接读取已有 CSV，复用现有实验数据，不重新训练
    if cli.plot_only:
        if not os.path.exists(cli.from_csv):
            print(f"❌ CSV not found: {cli.from_csv}")
            sys.exit(1)
        df = pd.read_csv(cli.from_csv)
        print(f"📥 Loaded results from {cli.from_csv}:")
        print(df)
        plot_results(df, output_dir=cli.output_dir)
        print("\nDone (plot-only).")
        sys.exit(0)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    print(f"State file: {STATE_FILE}")
    state = load_state()

    try:
        for exp in experiments:
            run_experiment(exp, state)
    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user.")
    finally:
        print("\nGenerating final summary...")
        df = update_results_csv(state)
        if df is not None:
            print("\nLatest Results:")
            print(df)
            plot_results(df, output_dir=cli.output_dir)
        print("\nDone.")