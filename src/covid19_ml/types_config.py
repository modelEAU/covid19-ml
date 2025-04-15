from datetime import datetime
from enum import Enum

from pydantic import BaseModel, field_validator


class DataSetConfig(BaseModel):
    city: str
    filename: str
    date_start: str
    date_end: str
    date_split: str
    validation_split: int

    @field_validator("date_start", "date_end", "date_split")
    def date_validator(cls, v):
        return datetime.strptime(v, "%Y-%m-%d") if isinstance(v, str) else v


class Optimizer(Enum):
    Adam = "Adam"
    AdamW = "AdamW"
    RAdam = "RAdam"
    NAdam = "NAdam"
    SparseAdam = "SparseAdam"
    SGD = "SGD"
    ASGD = "ASGD"
    Rprop = "Rprop"
    RMSprop = "RMSprop"
    Adadelta = "Adadelta"
    Adagrad = "Adagrad"
    Adamax = "Adamax"
    LBFGS = "LBFGS"


class OptimizerConfig(BaseModel):
    type: str
    learning_rate: float
    momentum: float
    weight_decay: float
    learning_rate_decay: float


class CityConvConfig(BaseModel):
    n_layers_city: int
    n_middle_channels_city: int
    n_out_channels_city: int
    city_pooling_padding: int
    city_pooling_kernel_size: int
    city_pooling_stride: int
    city_pooling_dilation: int
    city_conv_padding: int
    city_conv_dilation: int
    city_conv_stride: int
    city_kernel_size: int
    entry_nn_middle: bool
    entry_nn_prediction: bool
    n_layers_middle: int
    n_middle_channels_middle: int
    n_out_channels_middle: int
    middle_pooling_padding: int
    middle_pooling_kernel_size: int
    middle_pooling_stride: int
    middle_pooling_dilation: int
    middle_conv_padding: int
    middle_conv_dilation: int
    middle_conv_stride: int
    middle_kernel_size: int

    n_layers_prediction: int
    n_middle_channels_prediction: int
    n_out_channels_prediction: int
    prediction_kernel_size: int
    prediction_pooling_padding: int
    prediction_pooling_kernel_size: int
    prediction_pooling_stride: int
    prediction_pooling_dilation: int
    prediction_conv_padding: int
    prediction_conv_dilation: int
    prediction_conv_stride: int

    pooling_type: str
    dropout_rate: float
    leaking_rate: float
    activation: str


class SmallNNConfig(BaseModel):
    n_hidden: int
    n_layers: int


CityClassifierConfig = SmallNNConfig


class HyperParameters(BaseModel):
    recipe: str | None
    random_seed: int
    task: str
    trend_model_window: int
    trend_model_order: int
    denoise_window: int
    model_type: str
    optimizer: OptimizerConfig
    n_montecarlo_samples: int
    insert_dummy_variable: bool
    use_identity_for_city_heads: bool
    fake_ww_shift: int
    artificial_noise: bool
    kl_weight: float
    loss_fn: str
    city_conv: CityConvConfig | None
    classifier: CityClassifierConfig | None
    small_nn: SmallNNConfig | None

    batch_size: int

    classification_batch_size: int
    n_back: int
    n_forward: int
    n_splits: int
    patience: int
    target_type: str  # full, delta
    weighted_loss: bool
    classifier_regularization: float
    train_classification: bool
    train_prediction: bool
    log_transform: bool


class Recipe(BaseModel):
    inputs: list[str]
    input_ts: list[str]
    targets: list[str]
    target_ts: list[str]


class CometConfig(BaseModel):
    project_name: str
    workspace: str
    save_dir: str


class Config(BaseModel):
    models_folder: str
    data_folder: str
    logs_folder: str
    metrics_folder: str
    hparams: HyperParameters
    datasets: dict[str, DataSetConfig]
    recipes: dict[str, Recipe]
