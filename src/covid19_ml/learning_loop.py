import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import comet_ml  # noqa: F401
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from easydict import EasyDict as edict
from pydantic import BaseModel
from pytorch_lightning.loggers import CometLogger

from covid19_ml.augment import augment_datasets
from covid19_ml.datamodules import (
    CityDataModule,
    ClassBalancedDataModule,
    CombinedDataModule,
)
from covid19_ml.datasets import CityDataSet
from covid19_ml.models import CityConvModel, IdentityModule
from covid19_ml.modules import CityHeadModule
from covid19_ml.types_config import Config, DataSetConfig, HyperParameters, Recipe
from covid19_ml.types_ml import (
    City,
    SavedResults,
    Task,
    TensorConfig,
    TimeSeriesType,
    TrainingMode,
    Variable,
)
from covid19_ml.visualizations import plot_city_classifier_distribution_tsne2

import pytorch_lightning as pl  # isort: skip
from pytorch_lightning.callbacks.early_stopping import EarlyStopping  # isort: skip
from pytorch_lightning.callbacks import ModelCheckpoint  # isort: skip
import torch  # isort: skip
import warnings

MAX_EPOCHS = 200
device = (
    torch.device("mps")
    if False
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


def plot_initial_tsne(datamodule: CombinedDataModule | ClassBalancedDataModule):
    val_data = torch.vstack(
        [item[0] for item in iter(datamodule.val_dataloader())]
    ).numpy()

    infos = [item[2] for item in iter(datamodule.val_dataloader())]
    city_labels = [info["city"] for info in infos]
    city_labels = [item for sublist in city_labels for item in sublist]
    city_lookup = {}
    i = 0
    for city in city_labels:
        if city not in city_lookup:
            city_lookup[city] = i
            i += 1
    city_labels = np.array([city_lookup[city] for city in city_labels])  # type: ignore
    fig = plot_city_classifier_distribution_tsne2(val_data, city_labels, city_lookup)  # type: ignore
    return fig


def model_to_dict(obj):
    if isinstance(obj, BaseModel):
        return {
            field: model_to_dict(value) for field, value in obj.model_dump().items()
        }
    elif isinstance(obj, list):
        return [model_to_dict(item) for item in obj]
    else:
        return obj


def available_gpus() -> int:
    return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


def get_num_nodes(is_debug: bool = False) -> int:
    if is_debug:
        return 1
    if "SLURM_JOB_NUM_NODES" in os.environ:
        return int(os.environ["SLURM_JOB_NUM_NODES"])
    return 1


def build_tensor_config(recipe: Recipe, hparams: HyperParameters) -> TensorConfig:
    task = Task(hparams.task)
    n_back = hparams.n_back
    n_forward = hparams.n_forward

    input_variables = [Variable(var) for var in recipe.inputs]
    input_ts = [TimeSeriesType(ts) for ts in recipe.input_ts]
    if task == Task.CLASSIFICATION:
        classification_order = hparams.trend_model_order
        if TimeSeriesType.TREND_SLOPE not in input_ts:
            input_ts.append(TimeSeriesType.TREND_SLOPE)
        if classification_order == 2 and TimeSeriesType.TREND_CURVATURE not in input_ts:
            input_ts.append(TimeSeriesType.TREND_CURVATURE)

    target_variables = [Variable(var) for var in recipe.targets]
    target_ts = [TimeSeriesType(ts) for ts in recipe.target_ts]
    if task == Task.CLASSIFICATION:
        classification_order = hparams.trend_model_order
        target_ts = [TimeSeriesType.TREND_SLOPE]
        if classification_order == 2:
            target_ts.append(TimeSeriesType.TREND_CURVATURE)

    trend_model_window = hparams.trend_model_window
    trend_model_order = hparams.trend_model_order

    return TensorConfig(
        task=task,
        input_variables=input_variables,
        target_variables=target_variables,
        input_ts_types=input_ts,
        target_ts_types=target_ts,
        n_timesteps_back=n_back,
        n_timesteps_forward=n_forward,
        trend_model_window=trend_model_window,
        trend_model_order=trend_model_order,
        insert_dummy_variable=hparams.insert_dummy_variable,
        days_before=hparams.fake_ww_shift,
        artificial_noise=hparams.artificial_noise,
    )


def get_datasets(
    data_folder: str,
    dataset_configs: list[DataSetConfig],
    tensor_config: TensorConfig,
    is_debug: bool,
) -> list[CityDataSet]:
    data_path = Path(data_folder)

    datasets = []
    for dataset_config in dataset_configs:
        city = City(dataset_config.city)
        print("Building dataset for city ", city.value)
        filename = dataset_config.filename
        date_start = dataset_config.date_start
        date_end = dataset_config.date_end
        file_path = data_path / filename

        dataset = CityDataSet(
            path=file_path,
            city=city,
            start_date=date_start if not is_debug else "2022-01-01",
            end_date=date_end if not is_debug else "2023-02-15",
            tensor_config=tensor_config,
        )
        datasets.append(dataset)
    return datasets


def get_datamodules(
    datasets: list[CityDataSet],
    dataset_configs: list[DataSetConfig],
    hparams: HyperParameters,
) -> list[CityDataModule]:
    datamodules = []
    for dataset, dataset_config in zip(datasets, dataset_configs):
        assert dataset.city == City(dataset_config.city)
        date_split = dataset_config.date_split
        validation_split = dataset_config.validation_split
        if hparams.task == "CLASSIFICATION":
            batch_size = hparams.classification_batch_size
        else:
            batch_size = hparams.batch_size
        datamodule = CityDataModule(
            city_dataset=dataset,
            test_split_date=date_split,
            batch_size=batch_size,
            n_splits=hparams.n_splits,
            validation_split=validation_split,
            task=Task(hparams.task),
            log_transform=hparams.log_transform,
        )
        datamodules.append(datamodule)
    return datamodules


def pick_accelerator() -> str:
    if torch.cuda.is_available():
        return "gpu"
    elif sys.platform == "darwin":
        return "cpu"  # "mps"
    return "cpu"


def pick_model_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif sys.platform == "darwin":
        return "mps"
    return "cpu"


def pick_devices() -> list[int] | int:
    if sys.platform == "darwin":
        return 1
    elif torch.cuda.is_available():
        return list(range(available_gpus()))
    return []


def learning_loop(
    city_datamodules: list[CityDataModule],
    hparams: HyperParameters,
    tensor_config: TensorConfig,
    log_directory: str,
    metrics_directory: str,
    models_directory: str,
    recipe_name: str,
    is_debug: bool,
    use_comet: bool,
    unseen_city_name: str | None = None,
) -> None:
    if not os.path.isdir(log_directory):
        raise ValueError(f"Log directory does not exist: {log_directory}")
    if not os.path.isdir(metrics_directory):
        raise ValueError(f"Metrics directory does not exist: {metrics_directory}")
    if not os.path.isdir(models_directory):
        raise ValueError(f"Model directory does not exist: {models_directory}")
    if unseen_city_name:
        normal_datamodule = CombinedDataModule(
            datamodules=[dm for dm in city_datamodules if dm.city != unseen_city_name],
            batch_size=city_datamodules[0].batch_size,
        )
        unseen_datamodule = CombinedDataModule(
            datamodules=[dm for dm in city_datamodules if dm.city == unseen_city_name],
            batch_size=city_datamodules[0].batch_size,
        )
        full_datamodule = CombinedDataModule(
            datamodules=city_datamodules,
            batch_size=city_datamodules[0].batch_size,
        )
    else:
        normal_datamodule = CombinedDataModule(
            datamodules=city_datamodules,
            batch_size=city_datamodules[0].batch_size,
        )
        full_datamodule = None
    best_model_paths = []
    model_results = {}
    last_model = None
    hparams.patience = 2 if is_debug else hparams.patience
    for split in range(hparams.n_splits):
        if split != 2:
            # print("not training on split ", split)
            normal_datamodule.setup("fit")
            if unseen_city_name:
                unseen_datamodule.setup("fit")
                full_datamodule.setup("fit")
            continue
        if unseen_city_name:
            model = create_model(
                hparams.model_type,
                hparams,
                [dm for dm in city_datamodules if dm.city != unseen_city_name],
                tensor_config,
            )
        else:
            model = create_model(
                hparams.model_type,
                hparams,
                city_datamodules,
                tensor_config,
            )
        model = prepare_model(
            model, hparams.random_seed, recipe_name + "_split_" + str(split)
        )
        logger = setup_logger(
            log_directory, (is_debug or not use_comet), model.model_name
        )
        hparam_dict = model_to_dict(hparams)
        logger.log_hyperparams(hparam_dict)  # type: ignore

        early_stopping_training = EarlyStopping(
            monitor="validation_loss_total_prediction",
            mode="min",
            min_delta=0.005,
            patience=hparams.patience,
            verbose=True,
        )
        mod_checkpoint_training = ModelCheckpoint(
            monitor="validation_loss_total_prediction",
            mode="min",
            filename="{epoch}-{validation_loss_total:.4f}",
            save_top_k=1,
            verbose=True,
        )
        normal_trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            num_nodes=1,
            max_epochs=MAX_EPOCHS if not is_debug else 1,
            logger=logger,
            log_every_n_steps=1 if is_debug else 10,
            callbacks=[early_stopping_training, mod_checkpoint_training],  # type: ignore
            deterministic=True,
            enable_progress_bar=True,
        )
        model.log_fig(plot_initial_tsne(normal_datamodule), "initial_tsne")
        normal_trainer.fit(model, datamodule=normal_datamodule)
        trainer_test_result = normal_trainer.test(model, datamodule=normal_datamodule)  # type: ignore

        if unseen_city_name:
            early_stopping_inverse = EarlyStopping(
                monitor="validation_loss_total_prediction",
                mode="min",
                min_delta=0.005,
                patience=hparams.patience,
                verbose=True,
            )
            mod_checkpoint_inverse = ModelCheckpoint(
                monitor="validation_loss_total_prediction",
                mode="min",
                filename="{epoch}-{validation_loss_total:.4f}_unseen",
                save_top_k=1,
                verbose=True,
            )
            if not hparams.use_identity_for_city_heads:
                new_head = CityHeadModule(
                    city=unseen_datamodule.datamodules[0].city,
                    n_days_back=int(model.n_days_back),
                    n_in_channels=int(model.n_in_channels_city),
                    n_middle_channels=int(model.n_middle_channels_city),
                    n_out_channels=int(model.n_out_channels_city),
                    n_layers=int(model.n_layers_city),
                    dropout_rate=model.dropout_rate,
                    kernel_size=model.city_kernel_size,
                    leaking_rate=model.leaking_rate,
                    pool_type=model.pool_type,
                    conv_padding=int(model.city_conv_padding),
                    conv_dilation=int(model.city_conv_dilation),
                    conv_stride=int(model.city_conv_stride),
                    pool_kernel_size=int(model.city_pool_kernel_size),
                    pool_padding=int(model.city_pool_padding),
                    pool_stride=int(model.city_pool_stride),
                    pool_dilation=int(model.city_pool_dilation),
                    activation=model.activation,
                )
            else:
                new_head = IdentityModule(output_length=hparams.n_back)
            model.add_city_head(unseen_city_name, new_head)
            model.log_fig(
                plot_initial_tsne(full_datamodule), "initial_tsne (unseen data)"
            )
            model.training_mode = TrainingMode.NEW_HEAD
            new_head_trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                num_nodes=1,
                max_epochs=MAX_EPOCHS if not is_debug else 1,
                logger=logger,
                log_every_n_steps=1 if is_debug else 1,
                callbacks=[early_stopping_inverse, mod_checkpoint_inverse],  # type: ignore
                deterministic=True,
                enable_progress_bar=True,
            )
            if not hparams.use_identity_for_city_heads:
                model.configure_optimizers()
                new_head_trainer.fit(model, datamodule=unseen_datamodule)
            new_head_trainer.test(model, datamodule=unseen_datamodule)
            model.log_fig(
                plot_initial_tsne(full_datamodule), "final tsne (unseen data)"
            )
        if unseen_city_name and not hparams.use_identity_for_city_heads:
            checkpoint_callback = mod_checkpoint_inverse
        else:
            checkpoint_callback = mod_checkpoint_training
        best_model_paths.append(checkpoint_callback.best_model_path)
        model_results[split] = {
            "validation": deepcopy(model.complete_validation_results),
            "fold_path": checkpoint_callback.best_model_path,
            "kde_stats": deepcopy(model.kde_stats),
        }

        model_results[split]["trainer_test_results"] = trainer_test_result
        model_results[split]["test"] = deepcopy(model.complete_test_results)
        last_model = model

    test_results = [
        (split, x["trainer_test_results"]) for split, x in model_results.items()
    ]
    best_split_index = np.argmin(
        [x[1][0]["test_loss_total_prediction"] for x in test_results]
    )
    best_index = test_results[best_split_index][0]
    best_path = model_results[best_index]["fold_path"]
    kde_res = model_results[best_index]["kde_stats"]

    if last_model is None:
        raise ValueError("last_model is None")

    if unseen_city_name:
        new_model = create_model(
            hparams.model_type,
            hparams,
            [dm for dm in city_datamodules if dm.city != unseen_city_name],
            tensor_config,
        )
        new_model.add_city_head(unseen_city_name, new_head)
        chk = torch.load(best_path, weights_only=True)
        new_model.load_state_dict(chk["state_dict"])
    else:
        new_model = model
        new_model.load_from_checkpoint(
            best_path,
            cities=[dm.city for dm in city_datamodules],
            hparams=hparams,
            optimizer_config=hparams.optimizer,
            tensor_config=tensor_config,
            conv_config=hparams.city_conv,
            city_classifier_config=hparams.classifier,
            trial=None,
        )
    model = new_model
    model.kde_stats = kde_res
    name_model(model, recipe_name + "_best")
    results_filename = f"{metrics_directory}/{model.model_name}_{datetime.now().strftime('%Y-%m-%d')}_results.pt"
    to_save = SavedResults(
        model_name=model.model_name,
        cities=[model.cities],
        recipe=recipe_name,
        input_variables=model.input_variables,
        input_ts_types=model.input_ts_types,
        n_days_back=model.n_days_back,
        predicted_variables=model.predicted_variables,
        predicted_ts_types=model.predicted_ts_types,
        n_steps_forward=model.n_prediction_out,
        task=Task(hparams.task),
        complete_tests=model_results[best_index]["test"],
        last_validation=model_results[best_index]["validation"],
        n_classes=None,
        classifier_regularization=hparams.classifier_regularization,
        fake_ww_shift=hparams.fake_ww_shift,
        artificial_noise=hparams.artificial_noise,
        kde_stats=kde_res,
        ## hyperparameters
        # general
        model_type=hparams.model_type,
        train_classification=hparams.train_classification,
        train_prediction=hparams.train_prediction,
        use_identity_for_city_heads=hparams.use_identity_for_city_heads,
        loss_fn=hparams.loss_fn,
        target_type=hparams.target_type,
        weighted_loss=hparams.weighted_loss,
        batch_size=hparams.batch_size,
        pooling_type=hparams.city_conv.pooling_type,
        dropout_rate=hparams.city_conv.dropout_rate,
        activation=hparams.city_conv.activation,
        leaking_rate=hparams.city_conv.leaking_rate,
        # optimizer
        optimizer_type=hparams.optimizer.type,
        learning_rate=hparams.optimizer.learning_rate,
        momentum=hparams.optimizer.momentum,
        weight_decay=hparams.optimizer.weight_decay,
        learning_rate_decay=hparams.optimizer.learning_rate_decay,
        # model_structure
        entry_nn_middle=hparams.city_conv.entry_nn_middle,
        entry_nn_prediction=hparams.city_conv.entry_nn_prediction,
        n_middle_channels_city=hparams.city_conv.n_middle_channels_city,
        n_out_channels_city=hparams.city_conv.n_out_channels_city,
        n_layers_city=hparams.city_conv.n_layers_city,
        city_pooling_padding=hparams.city_conv.city_pooling_padding,
        city_pooling_kernel_size=hparams.city_conv.city_pooling_kernel_size,
        city_pooling_stride=hparams.city_conv.city_pooling_stride,
        city_pooling_dilation=hparams.city_conv.city_pooling_dilation,
        city_kernel_size=hparams.city_conv.city_kernel_size,
        city_conv_padding=hparams.city_conv.city_conv_padding,
        city_conv_dilation=hparams.city_conv.city_conv_dilation,
        city_conv_stride=hparams.city_conv.city_conv_stride,
        n_middle_channels_middle=hparams.city_conv.n_middle_channels_middle,
        n_out_channels_middle=hparams.city_conv.n_out_channels_middle,
        n_layers_middle=hparams.city_conv.n_layers_middle,
        middle_pooling_padding=hparams.city_conv.middle_pooling_padding,
        middle_pooling_kernel_size=hparams.city_conv.middle_pooling_kernel_size,
        middle_pooling_stride=hparams.city_conv.middle_pooling_stride,
        middle_pooling_dilation=hparams.city_conv.middle_pooling_dilation,
        middle_kernel_size=hparams.city_conv.middle_kernel_size,
        middle_conv_padding=hparams.city_conv.middle_conv_padding,
        middle_conv_dilation=hparams.city_conv.middle_conv_dilation,
        middle_conv_stride=hparams.city_conv.middle_conv_stride,
        n_middle_channels_prediction=hparams.city_conv.n_middle_channels_prediction,
        n_out_channels_prediction=hparams.city_conv.n_out_channels_prediction,
        n_layers_prediction=hparams.city_conv.n_layers_prediction,
        prediction_kernel_size=hparams.city_conv.prediction_kernel_size,
        prediction_pooling_padding=hparams.city_conv.prediction_pooling_padding,
        prediction_pooling_kernel_size=hparams.city_conv.prediction_pooling_kernel_size,
        prediction_pooling_stride=hparams.city_conv.prediction_pooling_stride,
        prediction_pooling_dilation=hparams.city_conv.prediction_pooling_dilation,
        prediction_conv_padding=hparams.city_conv.prediction_conv_padding,
        prediction_conv_dilation=hparams.city_conv.prediction_conv_dilation,
        prediction_conv_stride=hparams.city_conv.prediction_conv_stride,
        log_transform=hparams.log_transform,
        classifier_n_hidden=hparams.classifier.n_hidden,
        classifier_n_layers=hparams.classifier.n_layers,
    )
    torch.save(to_save, results_filename)

    model_filename = f"{models_directory}/{model.model_name}_{datetime.now().strftime('%Y-%m-%d')}_model.pt"
    torch.save(model.state_dict(), model_filename)


def setup_logger(log_directory: str, is_debug: bool, model_name: str):
    if not is_debug:
        logger = CometLogger(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=os.getenv("COMET_PROJECT"),
            workspace=os.getenv("COMET_WORKSPACE"),
            save_dir=log_directory,
        )
        logger.experiment.set_name(model_name)
    else:
        # use CSV logger
        logger = pl.loggers.CSVLogger(  # type: ignore
            save_dir=log_directory,
            name=model_name,
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    return logger


def learn_new_head(
    model_path: str,
    hparams: HyperParameters,
    city_datamodules: list[CityDataModule],
    extra_datamodule: CityDataModule,
    log_directory: str,
    tensor_config: TensorConfig,
    recipe_name: str,
    callbacks: list[str],
    is_debug: bool,
) -> None:
    model = create_model(
        hparams.model_type,
        hparams,
        city_datamodules,
        tensor_config,
    )
    model = prepare_model(
        model, hparams.random_seed, recipe_name + "newhead" + extra_datamodule.city
    )
    combined_datamodule = CombinedDataModule(
        datamodules=[extra_datamodule],
        batch_size=city_datamodules[0].batch_size,
    )
    checkpoint = torch.load(model_path, weights_only=True)
    # trained_model = model.load_from_checkpoint(
    #     model_path,
    #     cities=[dm.city for dm in city_datamodules],
    #     hparams=hparams,
    #     optimizer_config=hparams.optimizer,
    #     tensor_config=tensor_config,
    #     conv_config=hparams.city_conv,
    #     city_classifier_config=hparams.city_classifier,
    #     trial=None,
    # )
    model.load_state_dict(checkpoint)

    # create new head
    new_head = CityHeadModule(
        city=extra_datamodule.city,
        n_days_back=model.n_days_back,
        n_in_channels=model.n_in_channels_city,
        n_middle_channels=model.n_middle_channels_city,
        n_out_channels=model.n_out_channels_city,
        dropout_rate=model.dropout_rate,
        kernel_size=model.kernel_size,
        n_layers=model.n_layers_city,
        leaking_rate=model.leaking_rate,
    )

    # add an encoder head and make sure it's not locked
    model.add_city_head(extra_datamodule.city, new_head)  # not implemented
    # set up logger
    logger = setup_logger(log_directory, is_debug, model.name)
    early_stopping_training = EarlyStopping(
        monitor="validation_loss_total_prediction",
        mode="min",
        min_delta=0.005,
        patience=hparams.patience,
        verbose=True,
    )
    mod_checkpoint_training = ModelCheckpoint(
        monitor="validation_loss_total_prediction",
        mode="min",
        filename="{epoch}-{validation_loss_total:.4f}",
        save_top_k=1,
        verbose=True,
    )
    # set up a trainer
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        num_nodes=1,
        max_epochs=MAX_EPOCHS if not is_debug else 1,
        logger=logger,
        log_every_n_steps=1 if is_debug else 10,
        callbacks=[mod_checkpoint_training, early_stopping_training],  # type: ignore
        deterministic=True,
        enable_progress_bar=True,
    )
    # train
    trainer.fit(model, datamodule=combined_datamodule)
    # test
    model.training_mode = TrainingMode.NEW_HEAD
    trainer.test(model, datamodule=combined_datamodule, ckpt_path="best")  # type: ignore
    model_results = {
        "validation": deepcopy(model.complete_validation_results),
        "test": deepcopy(model.complete_test_results),
    }
    # save the model
    results_filename = f"Training/results/{model.model_name}_{datetime.now().strftime('%Y-%m-%d')}_results.pt"
    to_save = SavedResults(
        model_name=model.model_name,
        cities=[model.cities],
        recipe=recipe_name,
        input_variables=model.input_variables,
        input_ts_types=model.input_ts_types,
        n_days_back=model.n_days_back,
        predicted_variables=model.predicted_variables,
        predicted_ts_types=model.predicted_ts_types,
        n_steps_forward=model.n_prediction_out,
        task=Task(hparams.task),
        complete_tests=model_results["test"],
        last_validation=model_results["validation"],
        n_classes=None,
        classifier_regularization=hparams.classifier_regularization,
        fake_ww_shift=hparams.fake_ww_shift,
        artificial_noise=hparams.artificial_noise,
        kde_stats=model.kde_results,
        # hyperparameters
        # general
        model_type=hparams.model_type,
        train_classification=hparams.train_classification,
        train_prediction=hparams.train_prediction,
        use_identity_for_city_heads=hparams.use_identity_for_city_heads,
        loss_fn=hparams.loss_fn,
        target_type=hparams.target_type,
        weighted_loss=hparams.weighted_loss,
        batch_size=hparams.batch_size,
        pooling_type=hparams.city_conv.pooling_type,
        dropout_rate=hparams.city_conv.dropout_rate,
        activation=hparams.city_conv.activation,
        leaking_rate=hparams.city_conv.leaking_rate,
        # optimizer
        optimizer_type=hparams.optimizer.type,
        learning_rate=hparams.optimizer.learning_rate,
        momentum=hparams.optimizer.momentum,
        weight_decay=hparams.optimizer.weight_decay,
        learning_rate_decay=hparams.optimizer.learning_rate_decay,
        # model_structure
        entry_nn_middle=hparams.city_conv.entry_nn_middle,
        entry_nn_prediction=hparams.city_conv.entry_nn_prediction,
        n_middle_channels_city=hparams.city_conv.n_middle_channels_city,
        n_out_channels_city=hparams.city_conv.n_out_channels_city,
        n_layers_city=hparams.city_conv.n_layers_city,
        city_pooling_padding=hparams.city_conv.city_pooling_padding,
        city_pooling_kernel_size=hparams.city_conv.city_pooling_kernel_size,
        city_pooling_stride=hparams.city_conv.city_pooling_stride,
        city_pooling_dilation=hparams.city_conv.city_pooling_dilation,
        city_kernel_size=hparams.city_conv.city_kernel_size,
        city_conv_padding=hparams.city_conv.city_conv_padding,
        city_conv_dilation=hparams.city_conv.city_conv_dilation,
        city_conv_stride=hparams.city_conv.city_conv_stride,
        n_middle_channels_middle=hparams.city_conv.n_middle_channels_middle,
        n_out_channels_middle=hparams.city_conv.n_out_channels_middle,
        n_layers_middle=hparams.city_conv.n_layers_middle,
        middle_pooling_padding=hparams.city_conv.middle_pooling_padding,
        middle_pooling_kernel_size=hparams.city_conv.middle_pooling_kernel_size,
        middle_pooling_stride=hparams.city_conv.middle_pooling_stride,
        middle_pooling_dilation=hparams.city_conv.middle_pooling_dilation,
        middle_kernel_size=hparams.city_conv.middle_kernel_size,
        middle_conv_padding=hparams.city_conv.middle_conv_padding,
        middle_conv_dilation=hparams.city_conv.middle_conv_dilation,
        middle_conv_stride=hparams.city_conv.middle_conv_stride,
        n_middle_channels_prediction=hparams.city_conv.n_middle_channels_prediction,
        n_out_channels_prediction=hparams.city_conv.n_out_channels_prediction,
        n_layers_prediction=hparams.city_conv.n_layers_prediction,
        prediction_kernel_size=hparams.city_conv.prediction_kernel_size,
        prediction_pooling_padding=hparams.city_conv.prediction_pooling_padding,
        prediction_pooling_kernel_size=hparams.city_conv.prediction_pooling_kernel_size,
        prediction_pooling_stride=hparams.city_conv.prediction_pooling_stride,
        prediction_pooling_dilation=hparams.city_conv.prediction_pooling_dilation,
        prediction_conv_padding=hparams.city_conv.prediction_conv_padding,
        prediction_conv_dilation=hparams.city_conv.prediction_conv_dilation,
        prediction_conv_stride=hparams.city_conv.prediction_conv_stride,
        log_transform=hparams.log_transform,
        classifier_n_hidden_layers=hparams.city_classifier.n_hidden,
        classifier_n_hidden_neurons=hparams.city_classifier.n_layers,
    )
    torch.save(to_save, results_filename)

    model_filename = f"Training/models/{model.model_name}_{datetime.now().strftime('%Y-%m-%d')}_model.pt"
    torch.save(model.state_dict(), model_filename)
    with open("result_filenames.txt", "a") as f:
        f.write(f"{results_filename.split('/')[-1]}\n")
    # save the results


def read_config_file(config_file: str) -> Config:
    with open(config_file) as f:
        config = edict(yaml.safe_load(f))
    return Config(**config)


def build_city_conv_model(
    hparams: HyperParameters,
    tensor_config: TensorConfig,
    datamodules: list[CityDataModule],
    trial=None,
) -> CityConvModel:
    if hparams.city_conv is None:
        raise ValueError("CityConvConfig is None")
    if hparams.classifier is None:
        raise ValueError("CityClassifierConfig is None")
    return CityConvModel(
        cities=[dm.city for dm in datamodules],
        hparams=hparams,
        optimizer_config=hparams.optimizer,
        tensor_config=tensor_config,
        conv_config=hparams.city_conv,
        city_classifier_config=hparams.classifier,
        trial=trial,
    )


def name_model(model, recipe_name):
    model_name = f"{model.__class__.__name__}_{recipe_name}_{model.task.value}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    model.model_name = model_name  # type: ignore
    return model


def prepare_model(model, random_seed, recipe_name):
    device = pick_model_device()
    model = model.to(device)
    pl.seed_everything(random_seed)

    model = name_model(model, recipe_name)
    # model = torch.compile(model)
    model.to(device)
    return model


def create_model(
    model_type: str,
    hparams,
    datamodules: list[CityDataModule],
    tensor_config: TensorConfig,
):
    if model_type == "city_conv":
        model = build_city_conv_model(
            hparams=hparams,
            tensor_config=tensor_config,
            datamodules=datamodules,
            trial=None,
        )
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model


def main():
    load_dotenv()
    args = parse_args()
    if args.warnings:
        warnings.filterwarnings("error")
    is_debug = args.debug
    use_comet = args.use_comet
    recipename = args.recipename
    entry_nn = not args.no_entry_nn
    config = read_config_file(args.config)
    days_before = args.before
    artificial_noise = args.noise
    adversarial_factor = args.adv_factor
    encoders = args.enc
    hparams = config.hparams
    if encoders:
        hparams.use_identity_for_city_heads = False
        hparams.train_classification = True
    hparams.classifier_regularization = adversarial_factor
    hparams.recipe = recipename
    hparams.fake_ww_shift = days_before
    hparams.artificial_noise = artificial_noise
    hparams.loss_fn = args.loss_fn
    hparams.target_type = args.target
    hparams.batch_size = args.batch_size
    hparams.city_conv.entry_nn_middle = entry_nn
    hparams.city_conv.entry_nn_prediction = entry_nn
    hparams.log_transform = args.log
    hparams.city_conv.activation = args.activation_fn
    hparams.city_conv.leaking_rate = args.leaking_rate
    hparams.city_conv.dropout_rate = args.dropout_rate

    hp_table_path = args.hp_table
    hp_line = args.hp_line
    if hp_table_path and hp_line >= 0:
        hp_df = pd.read_csv(hp_table_path, index_col=0)
        hp_line_dict = hp_df.iloc[hp_line].to_dict()
        for key, val in hp_line_dict.items():
            key_bits = key.split(".")
            obj = hparams
            for key_bit in key_bits[:-1]:
                obj = getattr(obj, key_bit)
            setattr(obj, key_bits[-1], val)
    recipe = config.recipes[recipename]
    tensor_config = build_tensor_config(recipe, hparams)
    dataset_keys = args.datasets
    unseen_dataset = args.unseendataset
    if unseen_dataset:
        dataset_configs = [
            config.datasets[key] for key in dataset_keys + [unseen_dataset]
        ]
        unseen_name = config.datasets[unseen_dataset].city
    else:
        unseen_dataset = None
        dataset_configs = [config.datasets[key] for key in dataset_keys]
        unseen_name = ""
    datasets = get_datasets(
        config.data_folder, dataset_configs, tensor_config, is_debug
    )
    task = config.hparams.task
    cities = set([dataset.city for dataset in datasets])
    if len(cities) > 1:
        datasets = augment_datasets(datasets, task)
    datamodules = get_datamodules(datasets, dataset_configs, config.hparams)

    existing_model_path = args.model
    if existing_model_path != "":
        extra_dataset_key = args.newheaddataset
        if not os.path.exists(existing_model_path):
            raise FileNotFoundError(f"Could not find model {existing_model_path}")
        extra_dataset_config = config.datasets[extra_dataset_key]
        extra_dataset_ls = get_datasets(
            config.data_folder, [extra_dataset_config], tensor_config
        )
        extra_datamodules = get_datamodules(
            extra_dataset_ls, [extra_dataset_config], config.hparams
        )
        extra_datamodule = extra_datamodules[0]

        learn_new_head(
            model_path=existing_model_path,
            hparams=hparams,
            city_datamodules=datamodules,
            extra_datamodule=extra_datamodule,
            tensor_config=tensor_config,
            comet_config=config.comet,
            recipe_name=recipename,
            is_debug=is_debug,
        )
    else:
        learning_loop(
            datamodules,
            hparams,
            tensor_config,
            config.logs_folder,
            config.metrics_folder,
            config.models_folder,
            recipename,
            is_debug,
            use_comet,
            unseen_name,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="COVID-19 time series prediction model using wastewater data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="model_config.yaml",
        help="Path to the configuration file (default: model_config.yaml).",
    )
    parser.add_argument(
        "--recipename",
        type=str,
        default="basicph_n1_smooth",
        help="Name of the variable combination to use as defined in the recipes section of model_config.yaml.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["qc1_2021", "qc2_2021"],
        help="List of datasets to use for training, as defined in the datasets section of model_config.yaml.",
    )
    parser.add_argument(
        "--before",
        type=int,
        default=-1,
        help="Days to shift the wastewater SARS-COV-2 signal backward. When >= 0, reported cases replace the signal, shifted backward by this many days, and labeled as COVN1. Used to validate if the model can predict future signals from past observations.",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="MAE",
        help="Loss function to use during model training. Options are defined in losses.LOSSES_DICO.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="delta",
        help="Loss calculation method: 'full' computes loss on total difference between predicted and actual values, while 'delta' focuses on difference between t and t+h.",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        default=False,
        help="Add random noise to input variables to evaluate model robustness to measurement noise.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode with shortened training cycles and simplified logging.",
    )
    parser.add_argument(
        "--adv_factor",
        type=float,
        default=0.1,
        help="Regularization factor (0-1) controlling how strongly the model's encoder heads are encouraged to find similar encodings.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to an existing model file (.pt format) to use instead of training a new model.",
    )
    parser.add_argument(
        "--unseendataset",
        type=str,
        default="",
        help="Dataset to set aside for testing only, used to assess the model's generalization capability.",
    )
    parser.add_argument(
        "--newheaddataset",
        type=str,
        default="",
        help="Dataset to exclude from main training but use for training a new city-specific head before testing, to assess city-agnostic encoding capabilities.",
    )
    parser.add_argument(
        "--hp_table",
        type=str,
        default="",
        help="Path to CSV file containing hyperparameter configurations with parameter names as columns and values in rows.",
    )
    parser.add_argument(
        "--hp_line",
        type=int,
        default=-1,
        help="Row number in the hyperparameter table (--hp_table) to use for this experiment.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of samples per training batch. Overrides the value in model_config.yaml.",
    )
    parser.add_argument(
        "--activation_fn",
        type=str,
        default="leaky_relu",
        help="Activation function to use in the neural network. Overrides the value in model_config.yaml.",
    )
    parser.add_argument(
        "--leaking_rate",
        type=float,
        default=0.001,
        help="Leaking rate for leaky ReLU activation. Overrides the value in model_config.yaml.",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3,
        help="Dropout probability for regularization. Overrides the value in model_config.yaml.",
    )
    parser.add_argument(
        "--no_entry_nn",
        action="store_true",
        default=False,
        help="When set, removes the fully-connected layer at the beginning of model modules.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="Enable logging of training metrics and model information.",
    )
    parser.add_argument(
        "--use_comet",
        action="store_true",
        default=False,
        help="Use Comet.ml for experiment tracking instead of file logging. Requires COMET_API_KEY, COMET_WORKSPACE, and COMET_PROJECT environment variables.",
    )
    parser.add_argument(
        "--enc",
        action="store_true",
        default=False,
        help="Enable city-specific encoders in the model architecture.",
    )
    parser.add_argument(
        "--warnings",
        action="store_true",
        default=False,
        help="Treat warnings as errors, useful during development to catch and fix potential issues.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
