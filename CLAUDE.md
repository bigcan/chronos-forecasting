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