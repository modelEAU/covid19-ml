# COVID-19-ML

A machine learning project using convolutional neural networks and adversarial training for short-term COVID-19 epidemiological forecasting.

## Description

This project implements a machine learning approach for predicting COVID-19 epidemiological trends using multivariate time series data. The project includes specialized neural network architectures for processing city-specific features and making joint forecasts of multiple public health indicators across multiple time horizons.

## Features

- Data pipeline to transform datasets in the [PHES-ODM v1](https://github.com/Big-Life-Lab/PHES-ODM/releases/tag/v1.0.0) format into ready-to-use tensors.
- Time series forecasting for COVID-19 indicators.
- City-specific model adaptations.
- Visualization of the training process and modelling results.
- Results analysis and comparison tools.
- Qualitative trend analysis (slope, curvature, monotonicity).

## Installation

```bash
# Clone the repository
git clone https://github.com/modelEAU/covid19-ml.git
cd covid19-ml

# Install the package with dev dependencies
pip install -e ".[dev]"
```

## Requirements

- Python 3.10
- PyTorch >= 2.0.0
- PyTorch Lightning == 2.0.8
- Pandas, NumPy, SciPy
- Comet ML (for experiment tracking)
- Additional dependencies listed in pyproject.toml

## Usage

### Basic Usage

The simplest way to train a model with default settings:

```bash
python src/covid19_ml/learning_loop.py
```

This will train a model using the default "basicph_n1_smooth" recipe and the Quebec datasets specified in the configuration file.

### Common Use Cases

#### Train a model with a specific recipe and datasets

```bash
python src/covid19_ml/learning_loop.py --recipename reg_cases_hosp_deaths_smooth --datasets qc1_2021 qc2_2021 montreal_2022
```

#### Enable debugging for quicker iterations

```bash
python src/covid19_ml/learning_loop.py --debug
```

#### Use Comet.ml for experiment tracking

```bash
python src/covid19_ml/learning_loop.py --use_comet
```

Make sure you have the environment variables `COMET_API_KEY`, `COMET_WORKSPACE`, and `COMET_PROJECT` set.

#### Train a model to test generalization to an unseen city

```bash
python src/covid19_ml/learning_loop.py --datasets qc1_2021 qc2_2021 --unseendataset montreal_2022
```

#### Fine-tune hyperparameters

```bash
python src/covid19_ml/learning_loop.py --batch_size 8 --dropout_rate 0.5 --leaking_rate 0.01
```

#### Use a pre-trained model to train a new city head

```bash
python src/covid19_ml/learning_loop.py --model path/to/model.pt --newheaddataset new_city_2022
```

### Advanced Configuration

For more complex scenarios, you can use the hyperparameter table approach:

```bash
python src/covid19_ml/learning_loop.py --hp_table hyperparameters.csv --hp_line 7
```

This will load hyperparameter configuration from line 7 of the specified CSV file.

## Configuration

The system uses YAML configuration files to define:

- Model hyperparameters
- Dataset configurations
- Input/output recipes
- Logging settings

Example configuration structure:

```yaml
models_folder: "models/"
data_folder: "data/"
logs_folder: "logs/"
metrics_folder: "metrics/"
hparams:
  # Hyperparameters
  batch_size: 32
  learning_rate: 0.001
  # ... other hyperparameters
datasets:
  # Dataset configurations
recipes:
  # Input/output recipes
```

## Testing

```bash
pytest
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- Jean-David Therrien [mail](jeandavidt@gmail.com)
- Patrick Dallaire
- Simon Halle
- Thomas Maere
- Peter A. Varolleghem

## Acknowledgments

This project uses [Comet ML](https://www.comet.com/) for experiment tracking and visualization.
