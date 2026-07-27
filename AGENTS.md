# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Required Conda Environment:** Use the `rec` environment with conda:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rec
```

## Common Commands

### Training
```bash
# Basic training with default datasets (beauty_5_5, games_5_5, ml-1m_5_5)
python main.py --train_dir <experiment_name>

# Training with specific datasets
python main.py --train_dir <experiment_name> --use_datasets beauty_5_5 games_5_5

# Training with optimized temporal-frequency rating strategy
python main.py --train_dir <experiment_name> --rating_strategy temporal_fourier
```

### Inference/Evaluation
```bash
# Run inference only on trained model
python main.py --inference_only true --state_dict_path <path_to_model.pth> --train_dir <experiment_name>
```

### Visualization and Analysis
```bash
# Enhanced inference visualization (post-training analysis)
python scripts/inference_visualization.py --state_dict_path <model.pth> --train_dir <experiment_name>

# Statistical analysis and domain comparison
python scripts/multi_domain_user_comparison.py

# Fourier ablation studies
python scripts/run_fourier_ablation_statistical.py

# Batch visualization and statistics
python scripts/batch_viz_stats.py
```

### Data Processing
```bash
# Process raw datasets
python data_process.py

# Process new datasets
python process_datasets.py
```

## High-Level Architecture

### Core Model: SynRec (Synergistic Multi-Domain Recommendation with Frequency-Guided Expert Specialization)

**Main Components:**
- **HAGMRec Model** (`keys/model.py`): Core transformer-based recommendation model with:
  - Multi-head self-attention layers
  - Enhanced rating module with Fourier-based encoding
  - Mixture of Experts (MoE) feed-forward networks
  - Domain-adaptive configurations

- **MoE System** (`keys/c_moe.py`): Sophisticated mixture of experts with:
  - Domain-aware expert routing
  - Load balancing mechanisms
  - Specialization and contrastive losses
  - Adaptive gating with temperature control

- **Rating Module** (`keys/temporal_rating_modules.py`): Optimized temporal-frequency rating modeling:
  - OptimizedFourierRatingEncoder: Advanced FFT-based time-frequency analysis
  - Z-score normalization for numerical stability
  - Learnable frequency cutoff with soft boundaries
  - Spectral leakage prevention using windowing
  - Dual-branch attention for low/high frequency patterns
  - Unified domain processing with adaptive parameters

### Unified Architecture
- **Simplified Design**: Single optimized rating encoder handles all domains
- **Learnable Parameters**: Automatic frequency cutoff optimization eliminates manual tuning
- **Multi-domain Training**: Supports beauty, games, and MovieLens datasets with unified processing

### Visualization System
- **Enhanced Visualization** (`visualization/`): Publication-quality plots with multiple journal styles
- **Real-time Monitoring**: SwanLab integration for experiment tracking
- **Expert Analysis**: t-SNE plots, attention heatmaps, routing visualizations

### Key Features
- **Temporal-Fourier Rating Strategy**: Optimized time-frequency domain analysis with learnable parameters
- **Unified Processing**: Single encoder adaptively handles all domains without manual configuration
- **Performance-based Visualization**: Generates plots only when model improves
- **Multi-format Export**: PDF, PNG, SVG support for publications

### Data Pipeline
- **Multi-domain Dataset Loading**: Automatic partitioning and domain assignment
- **Rating-aware Sequences**: Incorporates both item and rating information
- **Stratified Sampling**: Ensures balanced representation across domains

## Model Configuration

### Key Hyperparameters
- `--rating_strategy temporal_fourier`: Use optimized temporal-frequency rating encoding (recommended)
- `--moe_routing_strategy shared_base`: Use shared base expert routing
- `--viz_on_improvement true`: Generate visualizations only on performance gains

### Performance Optimization
- Models automatically save only when test performance improves
- Visualization triggered by performance gains to reduce overhead
- Unified architecture minimizes computational cost while maximizing accuracy
- Learnable parameters eliminate the need for manual domain-specific tuning

## Important Files
- `main.py`: Primary training and evaluation script
- `keys/model.py`: Core HAGMRec model implementation
- `keys/temporal_rating_modules.py`: Optimized temporal-frequency rating encoder
- `keys/c_moe.py`: Mixture of Experts implementation
- `keys/utils.py`: Data loading and evaluation utilities
- `scripts/`: Advanced visualization and analysis tools
- `exp/`: Experiment outputs (models, logs, visualizations)
- `data/`: Preprocessed datasets
- `md/BEST_ARGS.md`: Optimal hyperparameter configurations
- `swanlog/`: SwanLab experiment tracking logs