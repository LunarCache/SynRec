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

# Training with enhanced visualization
python main.py --train_dir <experiment_name> --save_publication_figs true --journal_style science
```

### Inference/Evaluation
```bash
# Run inference only on trained model
python main.py --inference_only true --state_dict_path <path_to_model.pth> --train_dir <experiment_name>
```

### Data Processing
```bash
# Process raw datasets
python data_process.py

# Process new datasets
python process_datasets.py
```

### Testing
```bash
# Run system tests
python test_system.py

# Gradient checking
python gradient_check.py
```

## High-Level Architecture

### Core Model: HAGMRec (Hierarchical Adaptive Gating Multi-domain Recommender)

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

- **Rating Modules** (`keys/rating_modules.py`): Advanced rating information modeling:
  - FourierRatingEncoder: FFT-based multi-scale rating analysis
  - Dual-branch attention for temporal pattern learning
  - Adaptive fusion mechanisms

### Domain Management
- **Domain Config** (`keys/domain_config.py`): Automatic parameter tuning based on dataset characteristics
- **Multi-domain Training**: Supports beauty, games, and MovieLens datasets simultaneously
- **Domain-aware Sampling**: Stratified sampling for balanced multi-domain learning

### Visualization System
- **Enhanced Visualization** (`visualization/`): Publication-quality plots with multiple journal styles
- **Real-time Monitoring**: SwanLab integration for experiment tracking
- **Expert Analysis**: t-SNE plots, attention heatmaps, routing visualizations

### Key Features
- **Fourier Rating Strategy**: Default and recommended approach using FFT for rating sequence analysis
- **Adaptive Configuration**: Automatic parameter optimization per domain
- **Performance-based Visualization**: Generates plots only when model improves
- **Multi-format Export**: PDF, PNG, SVG support for publications

### Data Pipeline
- **Multi-domain Dataset Loading**: Automatic partitioning and domain assignment
- **Rating-aware Sequences**: Incorporates both item and rating information
- **Stratified Sampling**: Ensures balanced representation across domains

## Model Configuration

### Key Hyperparameters
- `--rating_strategy fourier`: Use Fourier-based rating encoding (recommended)
- `--use_adaptive_rating_config true`: Enable domain-specific parameter tuning
- `--moe_routing_strategy shared_base`: Use shared base expert routing
- `--viz_on_improvement true`: Generate visualizations only on performance gains

### Performance Optimization
- Models automatically save only when test performance improves
- Visualization triggered by performance gains to reduce overhead
- Domain-adaptive configurations minimize computational cost while maximizing accuracy

## Important Files
- `main.py`: Primary training and evaluation script
- `keys/model.py`: Core HAGMRec model implementation
- `keys/utils.py`: Data loading and evaluation utilities
- `exp/`: Experiment outputs (models, logs, visualizations)
- `data/`: Preprocessed datasets
- `md/BEST_ARGS.md`: Optimal hyperparameter configurations