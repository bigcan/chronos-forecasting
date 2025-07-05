# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Package Installation and Environment Setup
- `pip install chronos-forecasting` - Install the package from PyPI
- `pip install --editable ".[training]"` - Install in development mode with training dependencies
- `pip install ".[test]"` - Install with test dependencies
- `pip install ".[typecheck]"` - Install with type checking dependencies
- `pip install ".[evaluation]"` - Install with evaluation dependencies

### Testing
- `pytest` - Run all tests
- `mypy src test` - Run type checking on source and test directories

### Model Training and Evaluation
- `python scripts/training/train.py --config /path/to/config.yaml` - Train a model
- `CUDA_VISIBLE_DEVICES=0 python scripts/training/train.py --config /path/to/config.yaml` - Train on specific GPU
- `torchrun --nproc-per-node=8 scripts/training/train.py --config /path/to/config.yaml` - Multi-GPU training
- `python scripts/evaluation/evaluate.py evaluation/configs/in-domain.yaml results.csv` - Run evaluation
- `python scripts/kernel-synth.py --num-series 1000 --max-kernels 5` - Generate synthetic time series data

## Architecture Overview

Chronos is a family of pretrained time series forecasting models that treat time series as sequences of tokens, similar to language models. The architecture is based on transformer models (primarily T5) adapted for time series forecasting.

### Core Components

#### Pipeline Classes
- **BaseChronosPipeline**: Abstract base class for all Chronos pipelines
- **ChronosPipeline**: Main pipeline for original Chronos models (T5-based)
- **ChronosBoltPipeline**: Pipeline for Chronos-Bolt models (faster, more efficient versions)

#### Model Architecture
- **ChronosModel**: Core model implementation wrapping T5 architecture
- **ChronosTokenizer**: Handles conversion between time series values and discrete tokens
- **MeanScaleUniformBins**: Tokenization strategy using uniform binning with mean scaling

#### Key Files
- `src/chronos/base.py` - Base pipeline functionality and interfaces
- `src/chronos/chronos.py` - Original Chronos model implementation
- `src/chronos/chronos_bolt.py` - Chronos-Bolt model implementation
- `src/chronos/utils.py` - Utility functions for data processing

### Data Processing Pipeline
1. Time series are normalized using mean scaling
2. Values are quantized into discrete tokens using uniform binning
3. Tokens are fed into a transformer model for training/inference
4. During prediction, tokens are sampled and converted back to numerical values

### Model Variants
- **Chronos-T5**: Original models (tiny, mini, small, base, large) - 8M to 710M parameters
- **Chronos-Bolt**: Optimized models (tiny, mini, small, base) - 9M to 205M parameters, up to 250x faster

### Training and Evaluation Structure
- `scripts/training/` - Training scripts and configuration files
- `scripts/evaluation/` - Evaluation scripts and benchmark configurations
- `scripts/kernel-synth.py` - Synthetic data generation using Gaussian processes
- `test/` - Test suite with dummy models for testing

### Configuration Management
- Training configs in `scripts/training/configs/` specify model architecture, data paths, hyperparameters
- Evaluation configs in `scripts/evaluation/configs/` define benchmark datasets and metrics
- Models can be loaded from HuggingFace Hub using model IDs like "amazon/chronos-t5-small"

### Key Features
- Zero-shot forecasting capability
- Support for both probabilistic and point forecasts
- Encoder embedding extraction for downstream tasks
- Multi-GPU training support
- Integration with GluonTS ecosystem
- Arrow format dataset support

## Development Workflow

1. **Model Development**: Modify pipeline classes in `src/chronos/`
2. **Training**: Use configs in `scripts/training/configs/` and run training scripts
3. **Evaluation**: Use benchmark configs in `scripts/evaluation/configs/`
4. **Testing**: Run pytest for unit tests, mypy for type checking
5. **Data Generation**: Use `kernel-synth.py` for synthetic time series data

## Dependencies and Requirements
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.48+
- Accelerate 0.32+
- GluonTS for training and evaluation workflows
- HuggingFace Hub for model distribution

## Gold Futures Analysis Project

### Project Overview
The `gold_futures_analysis/` directory contains a comprehensive analysis of Chronos models for gold futures forecasting, including systematic optimization and performance evaluation.

### Key Analysis Files
- `gold_futures_chronos_fev_interactive.ipynb` - Main interactive analysis notebook with FEV framework integration
- `gold_futures_forecast_dashboard.html` - Interactive Plotly dashboard (4.9MB) with comprehensive visualizations
- `PHASE1_OPTIMIZATION_REPORT.md` - Complete Phase 1 optimization analysis and results
- `FORECASTING_GUIDE.md` - Comprehensive guide to forecasting methodologies

### Data Files
- `GCUSD_MAX_FROM_PERPLEXITY.csv` - Gold futures OHLCV data (2020-2021 primary analysis period)
- `gold_futures_forecast_predictions.csv` - Complete prediction results with all models
- `gold_futures_forecast_metrics.csv` - Performance metrics comparison

### Optimization Results
- `phase1_context_window_results.csv` - Context window optimization (30, 63, 126, 252 days)
- `phase1_model_size_results.csv` - Model size comparison (Tiny, Small, Base)
- `phase1_horizon_results.csv` - Prediction horizon analysis (1, 3, 7, 14 days)
- `phase1_final_comparison_results.csv` - Comprehensive model comparison
- `phase1_optimization_summary.csv` - Summary statistics and optimal configuration

### Analysis Scripts
- `run_phase1_optimization.py` - Context window optimization script
- `run_model_size_comparison.py` - Model size comparison script  
- `run_horizon_analysis.py` - Prediction horizon analysis script
- `run_final_optimization_comparison.py` - Comprehensive final comparison script

### Optimal Configuration Found
```
Context Window: 126 days
Model: amazon/chronos-bolt-base
Prediction Horizon: 1 day
Performance: MASE 1.4259 (vs 0.9953 naive baseline)
Directional Accuracy: 47.2%
```

### Key Findings
1. **Naive baseline dominance** in 2020-2021 trending gold market (MASE: 0.9953)
2. **Chronos optimization** achieved minimal improvement (-0.5%) but provides valuable directional signals
3. **Systematic testing** of 18+ configurations identified optimal settings
4. **Ensemble approach recommended** combining naive accuracy with Chronos directional intelligence

### Running the Analysis
```bash
# Context window optimization
python3 run_phase1_optimization.py

# Model size comparison  
python3 run_model_size_comparison.py

# Prediction horizon analysis
python3 run_horizon_analysis.py

# Final comprehensive comparison
python3 run_final_optimization_comparison.py
```

### Interactive Dashboard
Open `gold_futures_forecast_dashboard.html` in a web browser for interactive exploration of:
- Actual vs predicted prices with zoom capabilities
- Model performance comparison charts
- Error analysis and distribution plots
- Directional accuracy metrics
- Rolling performance analysis

### Future Development
- **Phase 2**: Feature engineering with technical indicators and external data
- **Ensemble methods**: Combining naive + Chronos for production systems
- **Market regime detection**: Volatility-based model switching
- **Extended testing**: 2022-2024 data validation