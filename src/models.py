"""
This module implements the CityConvModel class,
which is the main model of the project.
The model is composed of a series of modules,
which are defined in src/modules.py.
"""

import os
import tempfile
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from matplotlib.figure import Figure

import visualizations
from datamodules import get_batch_item_info
from losses import (
    LOSSES_DICO,
    CrossEntropyLoss,
)
from model_results import (
    extract_prediction_from_results_list_by_horizon,
    extract_prediction_from_results_list_mc_by_horizon,
    extract_true_values_series_from_results_list,
)
from modules import (
    CityHeadModule,
    GradientReversal,
    MiddleModule,
    PredictionHead,
    SmallNN,
)
from optimizers import create_optimizer
from results_analysis import (
    compile_city_classification_metrics,
    compile_regression_metrics,
    create_distances_dico,
)
from types_config import (
    CityConvConfig,
    HyperParameters,
    OptimizerConfig,
    SmallNNConfig,
)
from types_ml import (
    HeadTensors,
    Stage,
    StepResult,
    Task,
    TensorConfig,
    TimeSeriesType,
    TrainingMode,
    Variable,
)

# Set device variable
device = (
    torch.device("mps")
    if False
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

plt.rcParams["figure.max_open_warning"] = 100


def average_model_parameters(module_dict, new_module):
    module_key = list(module_dict.keys())[0]
    param_sum = {
        name: torch.zeros_like(param)
        for name, param in module_dict[module_key].named_parameters()
    }
    num_models = len(module_dict)

    for model in module_dict.values():
        for name, param in model.named_parameters():
            param_sum[name] += param.data

    param_avg = {name: param / num_models for name, param in param_sum.items()}

    with torch.no_grad():
        for name, param in new_module.named_parameters():
            param.data.copy_(param_avg[name])

    return new_module


class IdentityModule(nn.Module):
    def __init__(self, output_length: int = 1):
        self.output_length = output_length
        super().__init__()

    def forward(self, x):
        return x


class CityConvModel(pl.LightningModule):
    def __init__(
        self,
        cities: list[str],
        hparams: HyperParameters,
        tensor_config: TensorConfig,
        optimizer_config: OptimizerConfig,
        conv_config: CityConvConfig,
        city_classifier_config: SmallNNConfig,
        trial: Optional[optuna.Trial] = None,
        training_mode: TrainingMode = TrainingMode.PREDICTION_AND_CITY_CLASSIFIER,
    ):
        super().__init__()
        self.simple = {
            "city": hparams.use_identity_for_city_heads,
            "middle": False,
            "prediction": False,
        }
        self.cities = list(set(cities))
        self.current_city = self.cities[0]
        self.city_lookup = {city: i for i, city in enumerate(self.cities)}
        self.training_mode = training_mode
        self.set_optimizer_parameters(optimizer_config)
        self.set_hyperparameters(hparams)
        self.set_recipe(tensor_config)
        self.setup_trial(trial)
        self.create_result_stores()

        self.set_conv_parameters(conv_config)
        self.set_city_classifier_parameters(city_classifier_config)
        self.setup_city_heads()
        self.city_output_size = self.city_modules[self.cities[0]].output_length
        self.setup_middle_module()
        self.middle_output_size = self.middle_module.output_length  # type: ignore
        self.setup_prediction_heads()
        self.setup_city_classifier()
        self.configure_optimizers()
        self.last_validation_accuracy = None
        self.n_tests = 0
        self.new_head_name = None
        self.kde_stats = {}
        self.debug_sizes = {}

    def set_conv_parameters(self, conv_config: CityConvConfig):
        self.dropout_rate = conv_config.dropout_rate

        self.leaking_rate = conv_config.leaking_rate

        self.n_in_channels_city = int(
            len(self.input_variables) * len(self.input_ts_types)
        )
        self.n_middle_channels_city = int(conv_config.n_middle_channels_city)
        self.n_out_channels_city = int(conv_config.n_out_channels_city)

        self.n_layers_city = int(conv_config.n_layers_city)
        self.pooling_type = conv_config.pooling_type

        self.n_middle_channels_middle = int(conv_config.n_middle_channels_middle)
        self.n_out_channels_middle = int(conv_config.n_out_channels_middle)
        self.pool_type = conv_config.pooling_type
        self.entry_nn_middle = int(conv_config.entry_nn_middle)
        self.entry_nn_prediction = int(conv_config.entry_nn_prediction)
        self.middle_kernel_size = int(conv_config.middle_kernel_size)
        self.middle_conv_padding = int(conv_config.middle_conv_padding)
        self.middle_conv_dilation = int(conv_config.middle_conv_dilation)
        self.middle_conv_stride = int(conv_config.middle_conv_stride)
        self.middle_pool_kernel_size = int(conv_config.middle_pooling_kernel_size)
        self.middle_pool_padding = int(conv_config.middle_pooling_padding)
        self.middle_pool_stride = int(conv_config.middle_pooling_stride)
        self.middle_pool_dilation = int(conv_config.middle_pooling_dilation)

        self.city_kernel_size = int(conv_config.city_kernel_size)
        self.city_conv_padding = int(conv_config.city_conv_padding)
        self.city_conv_dilation = int(conv_config.city_conv_dilation)
        self.city_conv_stride = int(conv_config.city_conv_stride)
        self.city_pool_kernel_size = int(conv_config.city_pooling_kernel_size)
        self.city_pool_padding = int(conv_config.city_pooling_padding)
        self.city_pool_stride = int(conv_config.city_pooling_stride)
        self.city_pool_dilation = int(conv_config.city_pooling_dilation)

        self.prediction_kernel_size = int(conv_config.prediction_kernel_size)
        self.prediction_conv_padding = int(conv_config.prediction_conv_padding)
        self.prediction_conv_dilation = int(conv_config.prediction_conv_dilation)
        self.prediction_conv_stride = int(conv_config.prediction_conv_stride)
        self.prediction_pool_kernel_size = int(
            conv_config.prediction_pooling_kernel_size
        )
        self.prediction_pool_padding = int(conv_config.prediction_pooling_padding)
        self.prediction_pool_stride = int(conv_config.prediction_pooling_stride)
        self.prediction_pool_dilation = int(conv_config.prediction_pooling_dilation)

        self.n_layers_middle = int(conv_config.n_layers_middle)

        self.n_middle_channels_prediction = int(
            conv_config.n_middle_channels_prediction
        )
        self.n_out_channels_prediction = int(conv_config.n_out_channels_prediction)

        self.n_layers_prediction = int(conv_config.n_layers_prediction)
        self.activation = conv_config.activation

    def set_city_classifier_parameters(
        self,
        city_classifier_config: SmallNNConfig,
    ):
        self.n_layers_city_classifier = int(city_classifier_config.n_layers)
        self.n_hidden_city_classifier = int(city_classifier_config.n_hidden)

    def setup_city_classifier(self) -> None:
        if self.simple["city"]:
            classifier_inputs = self.n_in_channels_city * self.n_days_back
        else:
            classifier_inputs = self.city_output_size * self.n_out_channels_city  # type: ignore
        self.reverse_grad = GradientReversal(alpha=self.city_classifier_lambda)
        n_out = len(self.cities)
        self.city_classifier = SmallNN(
            n_inputs=classifier_inputs,
            n_hidden=self.n_hidden_city_classifier,
            n_hidden_layers=self.n_layers_city_classifier,
            n_out=n_out,
        )

    def setup_city_heads(self) -> nn.ModuleDict:
        self.city_modules = nn.ModuleDict()
        if self.simple["city"]:
            for city in self.cities:
                self.city_modules[city] = IdentityModule(output_length=self.n_days_back)
        else:
            for city in self.cities:
                city_module = CityHeadModule(
                    city=city,
                    n_days_back=self.n_days_back,
                    n_in_channels=self.n_in_channels_city,
                    n_middle_channels=self.n_middle_channels_city,
                    n_out_channels=self.n_out_channels_city,
                    n_layers=self.n_layers_city,
                    dropout_rate=self.dropout_rate,
                    kernel_size=self.city_kernel_size,
                    leaking_rate=self.leaking_rate,
                    pool_type=self.pool_type,
                    conv_padding=self.city_conv_padding,
                    conv_dilation=self.city_conv_dilation,
                    conv_stride=self.city_conv_stride,
                    pool_kernel_size=self.city_pool_kernel_size,
                    pool_padding=self.city_pool_padding,
                    pool_stride=self.city_pool_stride,
                    pool_dilation=self.city_pool_dilation,
                    activation=self.activation,
                )
                self.city_modules[city] = city_module
        return self.city_modules

    def add_city_head(self, city_name: str, new_head: CityHeadModule) -> None:
        if isinstance(new_head, CityHeadModule):
            new_head = average_model_parameters(self.city_modules, new_head)
        self.city_modules[city_name] = new_head
        self.new_head_name = city_name
        current_n_cities = len(self.cities)
        self.city_lookup[city_name] = current_n_cities
        self.training_mode = TrainingMode.NEW_HEAD

    def setup_middle_module(self) -> MiddleModule:
        if self.simple["middle"]:
            self.middle_module = IdentityModule(output_length=self.n_days_back)
        else:
            if self.simple["city"]:
                self.n_out_channels_city = len(self.input_variables) * len(
                    self.input_ts_types
                )
            self.middle_module = MiddleModule(
                input_length=self.city_output_size,  # type: ignore
                n_in_channels=self.n_out_channels_city,
                n_middle_channels=self.n_middle_channels_middle,
                n_out_channels=self.n_out_channels_middle,
                n_layers=self.n_layers_middle,
                dropout_rate=self.dropout_rate,
                kernel_size=self.middle_kernel_size,
                leaking_rate=self.leaking_rate,
                pool_type=self.pool_type,
                conv_padding=self.middle_conv_padding,
                conv_dilation=self.middle_conv_dilation,
                conv_stride=self.middle_conv_stride,
                pool_kernel_size=self.middle_pool_kernel_size,
                pool_padding=self.middle_pool_padding,
                pool_stride=self.middle_pool_stride,
                pool_dilation=self.middle_pool_dilation,
                activation=self.activation,
                entry_nn=self.entry_nn_middle,
            )
        return self.middle_module  # type: ignore

    def setup_prediction_heads(self) -> None:
        self.prediction_heads = nn.ModuleDict()
        for variable in self.predicted_variables:
            for ts_type in self.predicted_ts_types:
                if self.simple["prediction"]:
                    prediction_head = PredictionHead(
                        n_in_channels=len(self.input_variables)
                        * len(self.input_ts_types),
                        n_per_in_channel=self.n_days_back,
                        n_middle_channels=self.n_middle_channels_prediction,
                        n_out_channels=self.n_out_channels_prediction,
                        n_out_neurons=self.n_prediction_out,
                        variable=variable,
                        ts_type=ts_type,
                        n_layers=self.n_layers_prediction,
                        leaking_rate=self.leaking_rate,
                        dropout_rate=self.dropout_rate,
                        kernel_size=self.prediction_kernel_size,
                        conv_padding=self.prediction_conv_padding,
                        conv_dilation=self.prediction_conv_dilation,
                        conv_stride=self.prediction_conv_stride,
                        pooling_kernel_size=self.pool_kernel_size,
                        pooling_padding=self.prediction_pool_padding,
                        pooling_stride=self.prediction_pool_stride,
                        pooling_dilation=self.prediction_pool_dilation,
                        activation=self.activation,
                        entry_nn=self.entry_nn_prediction,
                    )
                else:
                    prediction_head = PredictionHead(
                        n_in_channels=self.n_out_channels_middle,
                        n_per_in_channel=self.middle_output_size,
                        n_middle_channels=self.n_middle_channels_prediction,
                        n_out_channels=self.n_out_channels_prediction,
                        n_out_neurons=self.n_prediction_out,
                        variable=variable,
                        ts_type=ts_type,
                        n_layers=self.n_layers_prediction,
                        leaking_rate=self.leaking_rate,
                        kernel_size=self.prediction_kernel_size,
                        dropout_rate=self.dropout_rate,
                        conv_padding=self.prediction_conv_padding,
                        conv_dilation=self.prediction_conv_dilation,
                        conv_stride=self.prediction_conv_stride,
                        pool_type=self.pool_type,
                        pool_kernel_size=self.prediction_pool_kernel_size,
                        pool_padding=self.prediction_pool_padding,
                        pool_stride=self.prediction_pool_stride,
                        pool_dilation=self.prediction_pool_dilation,
                        activation=self.activation,
                        entry_nn=self.entry_nn_prediction,
                    )
                self.prediction_heads[f"{variable.value},{ts_type.value}"] = (
                    prediction_head
                )

    def _forward(self, x, city):
        """
        Forward pass of the model. This is used for the training loop.
        """
        self.current_city = city
        city_out = self._ensure_3d(self.city_modules[city](x))

        classifier_prediction = self.city_classifier(self.reverse_grad(city_out))

        # 3. To the middle module to be used for the prediction heads
        middle_out = self._ensure_3d(self.middle_module(city_out))
        prediction_heads_out = {}
        for variable in self.predicted_variables:
            for ts_type in self.predicted_ts_types:
                prediction_head_out = self.prediction_heads[
                    self._prediction_key(variable, ts_type)
                ](middle_out)
                prediction_heads_out[self._prediction_key(variable, ts_type)] = (
                    prediction_head_out
                )

        return (
            prediction_heads_out,  # for predictions
            classifier_prediction,  # to improve embedding of city heads
            city_out.clone().detach(),  # to graphically represent the city embedding
        )

    def _step(self, batch):
        x, y, info = batch
        self.check_model_integrity()
        self._check_for_nan(x, "x")
        self._check_for_nan(y, "y")
        x = x.to(device)
        y = y.to(device)

        dico_y_hat = {}
        head_losses = {}
        predicted_city_labels = torch.zeros(x.shape[0]).to(device)
        batch_city_heads_outputs = torch.zeros(
            x.shape[0],
            self.n_out_channels_city,
            self.city_output_size,  # type: ignore
        ).to(device)
        for i in range(x.shape[0]):
            item_info = get_batch_item_info(info, i)
            city = item_info["city"]
            x_normalized = self._normalize_input(
                x[i],
                item_info["x_column_lookup"],
                self.trainer.datamodule.x_norm[city],
                self.log_transform,  # type: ignore
            )
            y_targets_normalized = self._normalize_output(
                y[i],
                item_info["y_column_lookup"],
                self.trainer.datamodule.y_norm[city],
                self.log_transform,  # type: ignore
            )
            (
                prediction_head_outputs,
                city_classifier_outputs,
                detached_city_encoder_outputs,
            ) = self._forward(x_normalized, city)
            self._check_for_nan(
                torch.stack(list(prediction_head_outputs.values())),
                f"y_hat of item {i}",
            )
            if self.training_mode == TrainingMode.NEW_HEAD:
                city_classifier_loss = torch.tensor([0.0], requires_grad=True)
            else:
                expected_index = torch.tensor([self.city_lookup[city]])
                city_classifier_loss = CrossEntropyLoss()(
                    city_classifier_outputs, expected_index
                ).unsqueeze(0)
            city_decoder_output_label = torch.argmax(city_classifier_outputs, dim=1)
            predicted_city_labels[i] = city_decoder_output_label.clone().detach().cpu()
            batch_city_heads_outputs[i] = detached_city_encoder_outputs.squeeze().cpu()

            head_losses[i], dico_y_hat[i] = self._compute_losses_and_predictions(
                prediction_head_outputs,
                x_normalized,
                y_targets_normalized,
                self.trainer.datamodule.y_norm[city],  # type: ignore
                item_info["x_column_lookup"],
                item_info["y_column_lookup"],
                self.predicted_variables,
                self.predicted_ts_types,
                use_delta=self.use_delta,
                weighted_loss=self.weighted_loss,
                city_classifier_loss=city_classifier_loss,
            )

        return (
            head_losses,
            dico_y_hat,
            batch_city_heads_outputs.cpu(),
            predicted_city_labels.cpu(),
        )

    @staticmethod
    def _ensure_3d(tensor: torch.Tensor) -> torch.Tensor:
        """Ensure the tensor has 3 dimensions."""
        if len(tensor.shape) == 3:
            return tensor
        elif len(tensor.shape) == 2:
            return tensor.unsqueeze(dim=0)
        elif len(tensor.shape) == 1:
            return tensor.unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            raise ValueError(f"Tensor has {len(tensor.shape)} dimensions")

    @staticmethod
    def _prediction_key(variable, ts_type) -> str:
        """Generate the dictionary key for prediction heads."""
        return f"{variable.value},{ts_type.value}"

    @staticmethod
    def _lookup_key(variable, ts_type) -> str:
        """Generate the dictionary key for the column lookup."""
        return f"{variable.value}_{ts_type.value}"

    @staticmethod
    def _check_for_nan(tensor, name) -> None:
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains nan")

    @staticmethod
    def _normalize_input(
        x: torch.Tensor,
        input_indices: dict[str, int],
        x_norm_stats: dict[str, tuple[float, float]],
        log_transform: bool,
    ) -> torch.Tensor:
        x = x.clone()
        x_normalized = torch.zeros_like(x)
        for col_name, col_index in input_indices.items():
            mean, std = x_norm_stats[col_name]
            if log_transform and ("RAW" in col_name or "SMOOTH" in col_name):
                x[col_index, :] = torch.log1p(x[col_index, :])
            x_normalized[col_index, :] = (x[col_index, :] - mean) / (std + 1e-5)
        return x_normalized

    @staticmethod
    def _normalize_output(y, output_indices, y_norm_stats, log_transform):
        y_targets_normalized = torch.zeros_like(y)
        for col_name, col_index in output_indices.items():
            mean, std = y_norm_stats[col_name]
            if log_transform and ("RAW" in col_name or "SMOOTH" in col_name):
                y[col_index, :] = torch.log1p(y[col_index, :])
            y_targets_normalized[col_index, :] = (y[col_index, :] - mean) / (std + 1e-5)
        return y_targets_normalized

    @staticmethod
    def _build_comparison_df(
        city_results: list[StepResult],
        name: str,
        n_days_back: int,
        n_days_ahead: int,
    ):
        true_values = extract_true_values_series_from_results_list(
            city_results, n_days_back, name
        )
        true_values.name = "true"
        predicted_values = {
            i: extract_prediction_from_results_list_by_horizon(city_results, i, name)
            for i in range(1, n_days_ahead + 1)
        }
        for horizon, series in predicted_values.items():
            series.name = f"t+{horizon}"
        predicted_values_df = pd.concat(predicted_values.values(), axis=1)
        return pd.concat([true_values, predicted_values_df], axis=1)

    @staticmethod
    def _build_mc_comparison_df(
        city_results: list[StepResult],
        name: str,
        n_days_back: int,
        n_days_ahead: int,
    ):
        true_values = extract_true_values_series_from_results_list(
            city_results, n_days_back, name
        )
        true_values.name = "true"
        predicted_values_dfs = {
            i: extract_prediction_from_results_list_mc_by_horizon(city_results, i, name)
            for i in range(1, n_days_ahead + 1)
        }
        mc_df = pd.concat(predicted_values_dfs.values(), axis=1)
        return pd.concat([true_values, mc_df], axis=1)

    def set_optimizer_parameters(self, optimizer_config: OptimizerConfig):
        self.optimizer_config = optimizer_config
        self.learning_rate = optimizer_config.learning_rate

    def set_hyperparameters(self, hyperparameters: HyperParameters):
        self.n_days_back = hyperparameters.n_back
        self.n_prediction_out = hyperparameters.n_forward
        self.task = Task(hyperparameters.task)
        self.trend_model_order = hyperparameters.trend_model_order
        self.batch_size = hyperparameters.batch_size
        self.use_delta = "delta" in hyperparameters.target_type
        self.weighted_loss = hyperparameters.weighted_loss
        self.n_montecarlo_dropout_samples = hyperparameters.n_montecarlo_samples
        self.city_classifier_lambda = hyperparameters.classifier_regularization
        self.train_prediction_heads = hyperparameters.train_prediction
        self.train_city_classification = hyperparameters.train_classification
        self.kl_weight = hyperparameters.kl_weight
        self.use_kl = hyperparameters.kl_weight > -1
        self.loss_fn = LOSSES_DICO[hyperparameters.loss_fn]
        self.log_transform = hyperparameters.log_transform

    def set_recipe(self, tensor_config: TensorConfig):
        self.input_variables = [
            Variable(input_) for input_ in tensor_config.input_variables
        ]
        self.input_ts_types = [
            TimeSeriesType(input_) for input_ in tensor_config.input_ts_types
        ]
        self.predicted_variables = [
            Variable(target) for target in tensor_config.target_variables
        ]
        self.predicted_ts_types = [
            TimeSeriesType(target) for target in tensor_config.target_ts_types
        ]

    def create_result_stores(self) -> None:
        # defining data structures to save results in
        self.epoch_test_results: list[StepResult] = []
        self.complete_test_results: list[StepResult] = []

        self.epoch_train_results: list[StepResult] = []
        self.epoch_validation_results: list[StepResult] = []
        self.complete_validation_results: list[StepResult] = []

        self.final_epoch_train_results: list[StepResult] = []
        self.final_epoch_validation_results: list[StepResult] = []
        self.first_middle_module_activations: list[torch.Tensor] = []

    def setup_trial(self, trial: Optional[optuna.Trial]) -> None:
        self.trial = trial

    def configure_optimizers(self):
        training_mode = self.training_mode
        print(f"Training mode is {training_mode}")
        print(
            "setting monitor variable and enabling gradients for specific parts of the model"
        )
        if training_mode == TrainingMode.PREDICTION_AND_CITY_CLASSIFIER:
            monitor = "validation_loss_total"
            for param in self.city_modules.parameters():
                param.requires_grad = True
            for param in self.middle_module.parameters():
                param.requires_grad = True
            for param in self.prediction_heads.parameters():
                param.requires_grad = True
            for param in self.city_classifier.parameters():
                param.requires_grad = True

        elif training_mode == TrainingMode.NEW_HEAD:
            new_head_name = self.new_head_name
            if new_head_name is None or new_head_name not in self.city_modules.keys():
                raise NameError(f"{new_head_name} not in City modules dictionary")
            monitor = "validation_loss_total_prediction"
            for name, module in self.city_modules.items():
                if name == new_head_name:
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    for param in module.parameters():
                        param.requires_grad = False
            for param in self.middle_module.parameters():
                param.requires_grad = False
            for param in self.prediction_heads.parameters():
                param.requires_grad = False
            for param in self.city_classifier.parameters():
                param.requires_grad = False

        else:
            raise ValueError(f"Unknown training mode: {training_mode}")

        trained_parameters = filter(lambda p: p.requires_grad, self.parameters())

        optimizer, scheduler = create_optimizer(
            self, self.optimizer_config, trained_parameters
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": monitor,
        }

    def check_model_integrity(self):
        """
        Checks if the model contains nan values.
        """
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"{name} contains nan")

    def _compute_losses_and_predictions(
        self,
        model_output_dico: dict[str, torch.Tensor],
        x_normalized: torch.Tensor,
        y_targets_normalized: torch.Tensor,
        y_normalization_stats: dict[str, tuple[float, float]],
        input_indices: dict[str, int],
        output_indices: dict[str, int],
        predicted_variables: list[Variable],
        predicted_ts_types: list[TimeSeriesType],
        use_delta: bool,
        weighted_loss: bool,
        city_classifier_loss: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]]:
        losses = {}

        losses["city_classifier"] = city_classifier_loss
        full_predictions_dico: dict[str, list[torch.Tensor]] = {}
        for variable in predicted_variables:
            for ts_type in predicted_ts_types:
                prediction_name = self._prediction_key(variable, ts_type)
                if prediction_name not in full_predictions_dico.keys():
                    full_predictions_dico[prediction_name] = []

                y_lookup_name = self._lookup_key(variable, ts_type)
                y_index = output_indices[y_lookup_name]

                x_lookup_name = self._lookup_key(variable, ts_type)
                input_index = input_indices[x_lookup_name]

                last_normalized_x = x_normalized[input_index, -1].detach()
                # model outputs a delta. I add the last value of the input to get the prediction and then denormalize
                # then, I store it for later
                model_output_normalized = model_output_dico[prediction_name].squeeze()

                y_mean = y_normalization_stats[y_lookup_name][0]
                y_std = y_normalization_stats[y_lookup_name][1]

                prediction_gap_normalized = None
                if use_delta:
                    prediction_gap_normalized = model_output_normalized
                    full_predictions_normalized = (
                        prediction_gap_normalized + last_normalized_x
                    )
                else:
                    full_predictions_normalized = model_output_normalized

                fullscale_predictions = (
                    full_predictions_normalized * (y_std + 1e-5)
                ) + y_mean
                if self.log_transform and ts_type in [
                    TimeSeriesType.RAW,
                    TimeSeriesType.SMOOTH,
                ]:
                    fullscale_predictions = torch.exp(fullscale_predictions) - 1

                full_predictions_dico[prediction_name].append(
                    fullscale_predictions.squeeze().detach().cpu()  # detach!
                )

                losses[prediction_name] = torch.tensor([0.0])

                target_normalized = y_targets_normalized[y_index, :].squeeze()
                target_gap_normalized = target_normalized - last_normalized_x

                loss_function = self.loss_fn(
                    use_horizon_weighting=weighted_loss,
                )
                if use_delta:
                    head_loss = (
                        loss_function(
                            target_gap_normalized,
                            prediction_gap_normalized,
                        )
                    ).mean()
                else:
                    head_loss = (
                        loss_function(
                            target_normalized,
                            full_predictions_normalized,
                        )
                    ).mean()
                losses[prediction_name] += head_loss

        losses["total"] = torch.stack([losses[key] for key in losses.keys()]).sum()
        return losses, full_predictions_dico

    def compute_batch_losses(self, batch_losses, shape):
        losses = {}
        for key in batch_losses[0].keys():
            losses[key] = torch.stack(
                [batch_losses[i][key] for i in range(shape)]
            ).sum()

        prediction_head_names = [key for key in losses.keys() if "," in key]
        losses["total_prediction"] = torch.stack(
            [losses[head] for head in prediction_head_names]
        ).sum()
        return losses

    def log_losses(self, losses, stage: Stage):
        for loss_name, loss_value in losses.items():
            self.log(
                f"{stage.value}_loss_{loss_name}",
                loss_value,
                batch_size=self.batch_size,
                on_step=False,
                on_epoch=True,
            )

    def _compile_and_store_results(
        self,
        x,
        y,
        info,
        head_predictions,
        city_outputs,
        city_predictions,
        stage,
        mc_results=None,
    ):
        batch_step_results = self.compile_step_results(
            x.cpu(),
            y.cpu(),
            info,
            head_predictions,
            city_outputs,
            city_predictions,
            Stage(stage),
            mc_results,
        )
        results_list = getattr(self, f"epoch_{stage.value}_results")
        results_list.extend(batch_step_results)
        return results_list

    def training_step(self, batch, batch_idx):
        head_losses, head_predictions, city_outputs, city_predictions = self._step(
            batch
        )
        x, y, info = batch
        shape = x.shape[0]
        batch_losses = self.compute_batch_losses(head_losses, shape)

        self.log_losses(batch_losses, Stage.TRAIN)
        info = self.bring_dict_to_cpu(info)
        self._compile_and_store_results(
            x,
            y,
            info,
            head_predictions,
            city_outputs,
            city_predictions,
            Stage.TRAIN,
        )
        return batch_losses["total"]

    def bring_dict_to_cpu(self, dico):
        for key in dico.keys():
            if isinstance(dico[key], torch.Tensor):
                dico[key] = dico[key].cpu()
            elif isinstance(dico[key], dict):
                dico[key] = self.bring_dict_to_cpu(dico[key])
        return dico

    def validation_step(self, batch, batch_idx):
        head_losses, head_predictions, city_outputs, city_predictions = self._step(
            batch
        )
        x, y, info = batch
        shape = x.shape[0]
        batch_losses = self.compute_batch_losses(head_losses, shape)
        self.log_losses(batch_losses, Stage.VAL)
        info = self.bring_dict_to_cpu(info)
        self._compile_and_store_results(
            x,
            y,
            info,
            head_predictions,
            city_outputs,
            city_predictions,
            Stage.VAL,
        )
        val_loss = batch_losses["total"]
        if self.trial:
            self.trial.report(val_loss, step=self.trainer.global_step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return val_loss

    def test_step(self, batch, batch_idx):
        if self.new_head_name is not None:
            self.training_mode = TrainingMode.NEW_HEAD
        else:
            self.training_mode = TrainingMode.PREDICTION_AND_CITY_CLASSIFIER
        head_losses, head_predictions, city_outputs, city_predictions = self._step(
            batch
        )

        x, y, info = batch
        batch_size = x.shape[0]
        batch_losses = self.compute_batch_losses(head_losses, batch_size)
        self.log_losses(batch_losses, Stage.TEST)
        info = self.bring_dict_to_cpu(info)
        self.train()
        monte_carlo_results = self.monte_carlo_step(batch)
        self.eval()
        self._compile_and_store_results(
            x,
            y,
            info,
            head_predictions,
            city_outputs,
            city_predictions,
            Stage.TEST,
            monte_carlo_results,
        )
        test_loss = batch_losses["total"]
        # self.log_step_mc_figs(self.epoch_test_results[-batch_size:], batch)
        return test_loss

    def _collect_mc_bunch_for_sample(self, x_, y_, info) -> list:
        bunch_results = []
        for _ in range(self.n_montecarlo_dropout_samples):
            _, y_hat, _, _ = self._step((x_, y_, info))
            bunch_results.append(y_hat)
        return bunch_results

    def _compute_percentiles(self, array):
        return {
            5: np.percentile(array, 5),
            50: np.percentile(array, 50),
            95: np.percentile(array, 95),
        }

    def _compute_pct_errors(self, new_percentiles, previous_percentiles):
        return {
            pct: abs(new_percentiles[pct] - previous_percentiles[pct])
            / (previous_percentiles[pct] + 1e-6)
            for pct in [5, 50, 95]
        }

    def _check_mc_bunches(self, mc_tensor, percentiles):
        pct_errors = {}
        for horizon in range(1, self.n_prediction_out + 1):
            new_horizon_percentiles = self._compute_percentiles(
                mc_tensor[:, horizon - 1].numpy()
            )
            pct_errors[horizon] = self._compute_pct_errors(
                new_horizon_percentiles, percentiles[horizon]
            )
            percentiles[horizon] = new_horizon_percentiles

        return percentiles, pct_errors

    def _mc_algorithm(
        self, x_, y_, info, head_names, max_bunches, max_pct_change
    ) -> list[torch.Tensor]:
        all_bunches = []
        head_percentiles = {
            k: {
                j: {i: 0 for i in [5, 50, 95]}
                for j in range(1, self.n_prediction_out + 1)
            }
            for k in head_names
        }
        pct_changes = {}
        for bunch_no in range(max_bunches):
            bunch_results = self._collect_mc_bunch_for_sample(x_, y_, info)
            all_bunches.extend(bunch_results)
            converged_heads = []
            for name in head_names:
                mc_tensor = torch.stack([y_hat[0][name][0] for y_hat in all_bunches])
                head_percentiles[name], pct_changes[name] = self._check_mc_bunches(
                    mc_tensor, head_percentiles[name]
                )
                head_has_converged = all(
                    all(
                        pct_change < max_pct_change
                        for pct_change in pct_changes[name][horizon].values()
                    )
                    for horizon in range(1, self.n_prediction_out + 1)
                )
                converged_heads.append(head_has_converged)
            if all(converged_heads):
                break
        # print(f"Finished after {bunch_no+1} bunches.")
        return all_bunches

    def monte_carlo_step(
        self, batch, max_bunches=20, max_pct_change=0.05
    ) -> list[list[torch.Tensor]]:
        x, y, info = batch
        batch_size = x.size(0)
        head_names = [
            self._prediction_key(var, ts_type)
            for var in self.predicted_variables
            for ts_type in self.predicted_ts_types
        ]
        all_batch_results = []
        for i in range(batch_size):
            x_ = x[i].unsqueeze(0)
            y_ = y[i].unsqueeze(0)
            sample_mc_results = self._mc_algorithm(
                x_, y_, info, head_names, max_bunches, max_pct_change
            )
            all_batch_results.append(sample_mc_results)
        return all_batch_results

    def log_step_mc_figs(self, step_results, batch):
        _, _, info = batch
        for i, step_result in enumerate(step_results):
            input_indices = info["x_column_lookup"]
            output_indices = info["y_column_lookup"]
            city = step_result.city
            last_input_date = step_result.info_item["last_input_date"]
            first_target_date = step_result.info_item["first_target_date"]
            x_ = step_result.x_item
            y_ = step_result.y_item
            for variable in self.predicted_variables:
                for ts_type in self.predicted_ts_types:
                    prediction_name = self._prediction_key(variable, ts_type)
                    lookup_name = self._lookup_key(variable, ts_type)
                    if not step_result.mc_results:
                        continue
                    y_hat = torch.stack(
                        [
                            mc_result[0][prediction_name][0]
                            for mc_result in step_result.mc_results
                        ]
                    ).squeeze()
                    x_item = x_[input_indices[lookup_name][0], :].squeeze()
                    y_item = y_[output_indices[lookup_name][0], :].squeeze()

                    if i % 10 == 0:
                        fig = visualizations.plot_montecarlo_dropout(
                            predicted_variable=variable,
                            predicted_ts_type=ts_type,
                            var_inputs=x_item.numpy(),
                            var_targets=y_item.squeeze().numpy(),
                            var_mc_preds=y_hat.numpy(),
                            city=city,
                            last_input_date=last_input_date,
                            first_target_date=first_target_date,
                        )

                        # Log the figure
                        self.log_fig(
                            fig,
                            f"Test {city} {prediction_name} montecarlo dropout t0 = {last_input_date} {self.mode_stub()}",
                        )
                        plt.close(fig)

    def mode_stub(self):
        if self.training_mode == TrainingMode.PREDICTION_AND_CITY_CLASSIFIER:
            return "pred_cl"
        return "new_head"

    def log_fig(self, fig, name):
        if self.logger is None:
            return
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            # for matplotlib fig: save the figure to a temporary file
            if isinstance(fig, Figure):
                fig.savefig(tmpfile.name, bbox_inches=None, pad_inches=0.5, dpi=200)
                # for plotly figures: save the figure to a temporary file§
                if hasattr(self.logger.experiment, "log_image"):  # type: ignore
                    self.logger.experiment.log_image(  # type: ignore
                        tmpfile.name,
                        name=name,
                    )
                # remove the temporary file
                os.remove(tmpfile.name)
            else:
                # Convert Plotly figure to HTML string
                fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

                # Log HTML string as an asset in Comet.ml
                if hasattr(self.logger.experiment, "log_asset_data"):  # type: ignore
                    self.logger.experiment.log_html(fig_html)
            # log the temporary file
        return

    def compile_step_results(
        self,
        x,
        y,
        info,
        head_predictions,
        city_outputs,
        city_predictions,
        stage: Stage,
        mc_results: Optional[list[list[dict[str, torch.Tensor]]]] = None,
    ) -> list[StepResult]:
        step_results = []
        batch_size = x.size(0)
        x_col_lookup = info["x_column_lookup"]
        y_col_lookup = info["y_column_lookup"]
        for i in range(batch_size):
            result = {
                "city": info["city"][i],
                "task": self.task.value,
                "date_t": pd.to_datetime(info["last_input_date"][i]),
                "x_item": x[i],
                "y_item": y[i],
                "info_item": get_batch_item_info(info, i),
                "head_results": {},
                "mc_results": mc_results[i] if mc_results else None,
                "stage": stage,
                "city_output": city_outputs[i],
                "city_predicted_label": city_predictions[i],
            }
            for variable in self.predicted_variables:
                for ts_type in self.predicted_ts_types:
                    name = f"{variable.value},{ts_type.value}"
                    col_lookup_name = "_".join([variable.value, ts_type.value])

                    result["head_results"][name] = {}

                    var_input = x[i, x_col_lookup[col_lookup_name], :]
                    head_tensors = HeadTensors(
                        input=var_input[i].squeeze(),
                        target=y[i, y_col_lookup[col_lookup_name], :][i].squeeze(),
                        prediction=head_predictions[i][name][0].squeeze(),
                    )

                    result["head_results"][name] = head_tensors

            step_result = StepResult(**result)
            step_results.append(step_result)
        return step_results

    def _plot_and_log_confusion_matrices(self, epoch_results: list[StepResult], stage):
        figs = visualizations.plot_confusion_matrices(
            self.epoch_test_results, self.n_prediction_out
        )

        for name, fig in figs.items():
            self.log_fig(
                fig,
                f"{stage.value} {name} confusion_matrix, epoch={self.current_epoch} {self.mode_stub()}",
            )
            plt.close(fig)
        figs = visualizations.plot_classification(
            self.epoch_test_results, self.n_prediction_out
        )
        for name, fig in figs.items():
            self.log_fig(
                fig,
                f"{stage.value} {name} classification, epoch={self.current_epoch} {self.mode_stub()}",
            )
            plt.close(fig)

    def _process_city_results_by_horizon(
        self, city_results: list[StepResult], city: str, name: str
    ) -> None:
        for horizon in range(1, self.n_prediction_out + 1):
            predicted_tensor = torch.stack(
                [
                    city_result.head_results[name].prediction[horizon - 1].squeeze()
                    for city_result in city_results
                ]
            )
            true_tensor = torch.vstack(
                [
                    city_result.head_results[name].target[horizon - 1].squeeze()
                    for city_result in city_results
                ]
            )

            metrics = self._compile_regression_metrics(
                city, name, horizon, predicted_tensor, true_tensor
            )

            if not self.logger:
                continue
            self.logger.experiment.log_metrics(metrics)

    def _compile_regression_metrics(
        self, city, name, horizon, predicted_tensor, true_tensor
    ) -> dict:
        predicted_tensor = predicted_tensor.reshape(-1)
        true_tensor = true_tensor.reshape(-1)
        return compile_regression_metrics(
            city, name, horizon, predicted_tensor, true_tensor
        )

    def _process_city_results(self, city_results: list[StepResult], city: str) -> None:
        city_results = sorted(city_results, key=lambda x: x.date_t)

        names = list(city_results[0].head_results.keys())
        for name in names:
            comparison_df = self._build_comparison_df(
                city_results,
                name,
                self.n_days_back,
                self.n_prediction_out,
            )
            mc_comparison_df = self._build_mc_comparison_df(
                city_results,
                name,
                self.n_days_back,
                self.n_prediction_out,
            )

            # fig = visualizations.plot_time_series_with_monte_carlo_bounds_plotly(
            #     mc_comparison_df, name, city, self.n_prediction_out, use_log=True
            # )
            # self.log_fig(
            #     fig,
            #     f" Test ts_{city}_{name}_mc, horizons=: 1 to {self.n_prediction_out} days ahead, log-scale {self.mode_stub()}",
            # )
            fig = visualizations.plot_time_series_with_monte_carlo_bounds_plotly(
                mc_comparison_df, name, city, self.n_prediction_out, use_log=False
            )
            self.log_fig(
                fig,
                f" Test ts_{city}_{name}_mc, horizons=: 1 to {self.n_prediction_out} days ahead, linear scale {self.mode_stub()}",
            )
            fig = visualizations.plot_head_results_by_horizon(
                comparison_df, name, city, self.n_prediction_out
            )
            self.log_fig(
                fig,
                f" Test ts_{city}_{name}_horizons=: 1 to {self.n_prediction_out} days ahead {self.mode_stub()}",
            )
            plt.close(fig)

            self._process_city_results_by_horizon(city_results, city, name)

    def _plot_distance_kde(self, results: list[StepResult], stage: Stage):
        if not results or len(self.cities) < 2:
            return
        results_by_city = {}
        for city in self.cities:
            city_results: list[StepResult] = [
                step_result for step_result in results if step_result.city == city
            ]
            if not city_results:
                continue
            results_by_city[city] = city_results
        data_by_city = {}
        for city, city_results in results_by_city.items():
            n_items = len(city_results)
            prediction_tensors = (
                torch.stack([result.city_output for result in city_results])
                .numpy()
                .reshape(n_items, -1)
            )
            true_city_labels = torch.tensor(
                [self.city_lookup[result.city] for result in city_results]
            ).numpy()
            dates = [result.date_t for result in city_results]
            data_by_city[city] = [
                {
                    "coordinates": prediction_tensor,
                    "label": true_city_label,
                    "date": date,
                }
                for prediction_tensor, true_city_label, date in zip(
                    prediction_tensors, true_city_labels, dates
                )
            ]
        for city in data_by_city.keys():
            data_by_city[city] = sorted(
                data_by_city[city], key=lambda x: x["date"], reverse=False
            )
        if len(data_by_city) < 2:
            return
        distances_between_cities = create_distances_dico(data_by_city)
        all_distances = []
        for _, distances in distances_between_cities.items():
            all_distances.extend(list(distances.flatten()))
        all_distances = np.array(all_distances)
        percentiles = np.percentile(all_distances, [5, 50, 95])
        self.kde_stats[self.current_epoch] = {
            "stage": stage,
            "training_mode": self.training_mode.value,
            "pct_5": percentiles[0],
            "pct_50": percentiles[1],
            "pct_95": percentiles[2],
        }
        # closest_between_cities = find_closest_examples_between_cities(
        #     results_by_city, distances_between_cities
        # )
        # fig = visualizations.plot_closest_examples_between_cities(
        #     results_by_city,
        #     closest_between_cities,
        # )
        # self.log_fig(
        #     fig,
        #     f"{stage.value} closest examples between cities, epoch={self.current_epoch}",
        # )
        # plt.close(fig)
        fig = visualizations.plot_distance_kde_between_cities(distances_between_cities)

        self.log_fig(
            fig,
            f"{stage.value} distance kde, epoch={self.current_epoch}, {self.mode_stub()}",
        )
        plt.close(fig)

    def _plot_city_classifier_cm(self, stage: Stage):
        results: list[StepResult] = getattr(self, f"epoch_{stage.value}_results")
        if not results:
            return
        predicted_city_labels = []
        true_city_labels = []
        for result in results:
            predicted_city_label = result.city_predicted_label
            true_city_label = self.city_lookup[result.city]
            predicted_city_labels.append(predicted_city_label)
            true_city_labels.append(true_city_label)
        predicted_city_labels = torch.tensor(predicted_city_labels)
        true_city_labels = torch.tensor(true_city_labels)

        city_classifier_cm = visualizations.plot_city_classifier_cm(
            predicted_city_labels, true_city_labels, self.city_lookup
        )
        figure = city_classifier_cm
        self.log_fig(
            figure,
            f"{stage.value} city classifier confusion matrix, epoch={self.current_epoch} {self.mode_stub()}",
        )
        plt.close(figure)

    def _plot_city_classifier_distribution(
        self, results: list[StepResult], stage: Stage
    ):
        if not results:
            return
        prediction_tensors = torch.stack(
            [result.city_output for result in results]
        ).numpy()
        true_city_labels = torch.tensor(
            [self.city_lookup[result.city] for result in results]
        ).numpy()
        figure = visualizations.plot_city_classifier_distribution_tsne2(
            prediction_tensors, true_city_labels, self.city_lookup
        )
        self.log_fig(
            figure,
            f"{stage.value} T-SNE, epoch={self.current_epoch} {self.mode_stub()}",
        )
        plt.close(figure)

    def _process_city_classifier_results(self, results: list[StepResult], stage: Stage):
        if not results:
            return
        true_city_labels = []
        predicted_city_labels = []
        for result in results:
            predicted_label = result.city_predicted_label
            true_city_label = self.city_lookup[result.city]
            predicted_city_labels.append(predicted_label)
            true_city_labels.append(true_city_label)
        predicted_city_labels = torch.tensor(predicted_city_labels)
        true_city_labels = torch.tensor(true_city_labels)
        if len(self.cities) > 1:
            metrics = compile_city_classification_metrics(
                true_city_labels, predicted_city_labels, self.city_lookup, stage.value
            )
            if stage == Stage.VAL:
                self.last_validation_accuracy = metrics[
                    f"CityClassifier_ACCURACY_{stage.value}"
                ]
            if not self.logger:
                return
            self.logger.experiment.log_metrics(metrics)

    def on_test_epoch_end(self) -> None:  # sourcery skip: low-code-quality
        # self._plot_and_log_all_predictions(all_results, Stage.TEST)
        self._plot_and_log_all_predictions(self.epoch_test_results, Stage.TEST)

        # self._compute_and_log_naive_comps(self.epoch_test_results, Stage.TEST, 1)
        if len(self.cities) > 1 and self.training_mode != TrainingMode.NEW_HEAD:
            self._plot_city_classifier_cm(Stage.TEST)
            self._plot_city_classifier_distribution(self.epoch_test_results, Stage.TEST)
            self._process_city_classifier_results(self.epoch_test_results, Stage.TEST)

        cities = list({x.city for x in self.epoch_test_results})
        for city in cities:
            city_results: list[StepResult] = [
                step_result
                for step_result in self.epoch_test_results
                if step_result.city == city
            ]
            self._process_city_results(city_results, city)

        self.complete_test_results.extend(self.epoch_test_results)

        self.epoch_test_results.clear()
        self.on_train_epoch_end()

    def _plot_and_log_all_predictions(self, results: list[StepResult], stage: Stage):
        figures = visualizations.plot_all_prediction_time_series(
            results,
            self.n_days_back,
            self.n_prediction_out,
            stage=stage.value,
            only_delta=True,
        )

        figures.update(
            visualizations.plot_all_prediction_45_degrees(
                results,
                self.n_prediction_out,
                stage.value,
                use_delta=True,
            )
        )
        for name, fig in figures.items():
            self.log_fig(
                fig,
                f"{stage.value} - All predictions, {name}, epoch={self.current_epoch}, {self.mode_stub()}",
            )
            # close the figure if it is a matplotlib figure
            if isinstance(fig, Figure):
                plt.close(fig)

    def on_validation_epoch_end(self):
        print(
            "Validation epoch end: we were validating in mode",
            self.training_mode,
        )
        self.complete_validation_results.clear()
        self.final_epoch_train_results.clear()
        self.final_epoch_validation_results.clear()
        self._process_city_classifier_results(
            self.epoch_train_results,
            Stage.TRAIN,
        )
        self._process_city_classifier_results(self.epoch_validation_results, Stage.VAL)
        self._plot_and_log_all_predictions(
            self.epoch_train_results + self.epoch_validation_results, Stage.VAL
        )
        if len(self.cities) > 1:
            self._plot_distance_kde(self.epoch_validation_results, Stage.TRAIN)
            self._plot_city_classifier_cm(Stage.VAL)
            self._plot_city_classifier_distribution(
                self.epoch_validation_results, Stage.VAL
            )
        self.complete_validation_results.extend(self.epoch_validation_results)
        self.final_epoch_train_results.extend(self.epoch_train_results)
        self.final_epoch_validation_results.extend(self.epoch_validation_results)
        self.epoch_train_results.clear()
        self.epoch_validation_results.clear()
        plt.close("all")
