# Environment Setup

Create and activate the Conda environment:

```bash
conda env create -f env.yaml
conda activate rec
```

CUDA is used by default. To run on CPU, add `--device cpu` to the commands below.

## Data Format

Preprocessed datasets are stored in `data/`. Each line follows this format:

```text
user_id item_id rating
```

Example:

```text
1 2794 4.0
1 5788 5.0
```

The repository includes the `baby_5_5`, `tools_5_5`, `toys_5_5`, `beauty_5_5`, `games_5_5`, and `ml-1m_5_5` datasets.

## Training

Train with the default datasets:

```bash
python main.py --train_dir demo
```

Train with selected datasets and the temporal-frequency rating strategy:

```bash
python main.py \
  --train_dir beauty_games_ml1m \
  --use_datasets beauty_5_5 games_5_5 ml-1m_5_5 \
  --rating_strategy temporal_fourier
```

Arguments, logs, and the best checkpoint are saved to:

```text
exp/<dataset_names>_<train_dir>/
```

## Inference and Evaluation

```bash
python main.py \
  --inference_only true \
  --state_dict_path exp/<experiment_dir>/<checkpoint>.pth \
  --train_dir <experiment_name> \
  --use_datasets baby_5_5 tools_5_5 toys_5_5
```

Use the same datasets and model configuration as the training run.

## Visualization

```bash
python scripts/inference_visualization.py \
  --experiment_dir exp/<experiment_dir> \
  --state_dict_path exp/<experiment_dir>/<checkpoint>.pth \
  --output_dir exp/inference_visualization
```

See [`scripts/README.md`](scripts/README.md) and [`scripts/README_inference_visualization.md`](scripts/README_inference_visualization.md) for additional analysis tools and options.

## Project Structure

```text
SynRec/
├── main.py                         # Training and evaluation entry point
├── keys/
│   ├── model.py                    # SynRec model
│   ├── c_moe.py                    # Mixture-of-Experts modules
│   ├── temporal_rating_modules.py  # Temporal-frequency rating encoder
│   └── utils.py                    # Data loading and evaluation utilities
├── data/                           # Preprocessed datasets
├── scripts/                        # Analysis, ablation, and visualization scripts
├── visualization/                  # Plotting utilities
├── baselines/                      # Baseline models
└── exp/                            # Experiment outputs
```
