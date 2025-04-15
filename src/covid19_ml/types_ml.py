from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pandas import Timestamp
from torch import Tensor


class TrainingMode(Enum):
    PREDICTION_AND_CITY_CLASSIFIER = "PREDICTION_AND_CITY_CLASSIFIER"
    NEW_HEAD = "NEW_HEAD"


class City(Enum):
    QUEBEC_EAST = "QUEBEC_EAST"
    QUEBEC_WEST = "QUEBEC_WEST"
    MONTREAL_NORTH = "MONTREAL_NORTH"
    MONTREAL_SOUTH = "MONTREAL_SOUTH"
    LAVAL_AUTEUIL = "LAVAL_AUTEUIL"  # lvl_1
    LAVAL_FABREVILLE = "LAVAL_FABREVILLE"  # lvl_2
    LAVAL_LAPINIERE = "LAVAL_LAPINIERE"  # lvl_5
    GATINEAU = "GATINEAU"
    STPAUL = "STPAUL"

    EASYVILLE_1 = "EASYVILLE_1"
    EASYVILLE_2 = "EASYVILLE_2"
    WEM_1 = "WEM_1"
    WEM_CITY1 = "WEM_CITY1"
    WEM_CITY2 = "WEM_CITY2"
    WEM_CITY3 = "WEM_CITY3"
    WEM_CITY4 = "WEM_CITY4"
    WEM_CITY5 = "WEM_CITY5"
    WEST_1 = "WEST_1"
    WEST_2 = "WEST_2"
    WEST_3 = "WEST_3"
    WEST_4 = "WEST_4"
    WEST_5 = "WEST_5"


class Variable(Enum):
    TIME = "TIME"
    CASES = "CASES"
    DEATHS = "DEATHS"
    HOSPITALIZED = "HOSPITALIZED"
    POS_TESTS_RATE = "POS_TESTS_RATE"
    PCT_VAX1 = "PCT_VAX1"
    PCT_VAX2 = "PCT_VAX2"
    BOD = "BOD"
    COD = "COD"
    TSS = "TSS"
    NH4 = "NH4"
    WATER_TEMP = "WATER_TEMP"
    FLOW = "FLOW"
    COVN1 = "COVN1"
    N1_FLOW = "N1_FLOW"
    COD_FLOW = "COD_FLOW"
    BOD_FLOW = "BOD_FLOW"
    NH4_FLOW = "NH4_FLOW"
    TSS_FLOW = "TSS_FLOW"
    RAINFALL = "RAINFALL"
    DUMMY = "DUMMY"


class TimeSeriesType(Enum):
    RAW = "RAW"
    SMOOTH = "SMOOTH"
    SLOPE = "SLOPE"
    CURVATURE = "CURVATURE"
    TREND_SLOPE = "TREND_SLOPE"
    TREND_CURVATURE = "TREND_CURVATURE"


class Task(Enum):
    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"


@dataclass
class Info:
    first_input_date: str
    last_input_date: str
    first_target_date: str
    last_target_date: str
    city: str
    augmented: bool = False


class RegressionMetric(Enum):
    MSE = "MSE"
    MSSE = "MSSE"
    RMSE = "RMSE"
    RMSSE = "RMSSE"  # Root Mean Squared Scaled Error
    MAE = "MAE"
    MASE = "MASE"
    NSE = "NSE"
    R2 = "R2"
    PEARSON = "PEARSON"
    SignAgreement = "SignAgreement"
    SSLE = "SSLE"


class ClassificationMetric(Enum):
    ACCURACY = "ACCURACY"
    F1SCORE = "F1SCORE"
    PRECISION = "PRECISION"
    RECALL = "RECALL"
    AUROC = "AUROC"


RegressionStats = dict[RegressionMetric, Tensor]
ClassificationStats = dict[ClassificationMetric, Tensor]


@dataclass
class HeadResults:
    name: str
    input: Tensor
    target: Tensor
    prediction: Tensor
    loss: Tensor
    metrics: RegressionStats | ClassificationStats

    def __repr__(self) -> str:
        return f"HeadResults({self.name})"


@dataclass
class HeadTensors:
    input: Tensor
    target: Tensor
    prediction: Tensor


class Stage(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


@dataclass
class StepResult:
    city: str
    task: str
    date_t: Timestamp
    x_item: Tensor
    y_item: Tensor
    city_predicted_label: Tensor
    city_output: Tensor
    info_item: dict
    head_results: dict[str, HeadTensors]
    stage: Stage
    mc_results: Optional[list[dict[str, Tensor]]]

    def __repr__(self) -> str:
        return f"StepResult({self.city}, {self.date_t})"


@dataclass
class SavedResults:
    model_name: str
    cities: list[str]
    recipe: str
    input_variables: list[Variable]
    input_ts_types: list[TimeSeriesType]
    n_days_back: int
    predicted_variables: list[Variable]
    predicted_ts_types: list[TimeSeriesType]
    n_steps_forward: int
    task: Task
    complete_tests: list[StepResult]
    last_validation: list[StepResult]
    n_classes: Optional[int]
    classifier_regularization: float
    fake_ww_shift: int
    artificial_noise: bool
    kde_stats: dict[int, dict[int, float]]

    ## hyperparameters
    log_transform: bool
    # general
    entry_nn_middle: bool
    entry_nn_prediction: bool
    model_type: str
    train_classification: bool
    train_prediction: bool
    use_identity_for_city_heads: bool
    loss_fn: str
    batch_size: int
    target_type: str
    weighted_loss: bool
    pooling_type: str
    dropout_rate: float
    activation: str
    leaking_rate: float
    # optimizer
    optimizer_type: str
    learning_rate: float
    momentum: float
    weight_decay: float
    learning_rate_decay: float
    # model_structure

    n_middle_channels_city: int
    n_out_channels_city: int
    n_layers_city: int
    city_pooling_padding: int
    city_pooling_kernel_size: int
    city_pooling_stride: int
    city_pooling_dilation: int
    city_kernel_size: int
    city_conv_padding: int
    city_conv_dilation: int
    city_conv_stride: int
    n_middle_channels_middle: int
    n_out_channels_middle: int
    n_layers_middle: int
    middle_pooling_padding: int
    middle_pooling_kernel_size: int
    middle_pooling_stride: int
    middle_pooling_dilation: int
    middle_kernel_size: int
    middle_conv_padding: int
    middle_conv_dilation: int
    middle_conv_stride: int
    n_middle_channels_prediction: int
    n_out_channels_prediction: int
    n_layers_prediction: int
    prediction_kernel_size: int
    prediction_pooling_padding: int
    prediction_pooling_kernel_size: int
    prediction_pooling_stride: int
    prediction_pooling_dilation: int
    prediction_conv_padding: int
    prediction_conv_dilation: int
    prediction_conv_stride: int
    classifier_n_layers: int
    classifier_n_hidden: int

    def __repr__(self) -> str:
        return f"SavedResults({self.model_name})"


@dataclass
class SampleData:
    info: Info
    task: Task
    head_results: list[HeadResults]


@dataclass
class TensorConfig:
    task: Task
    input_variables: list[Variable]
    target_variables: list[Variable]
    input_ts_types: list[TimeSeriesType]
    target_ts_types: list[TimeSeriesType]
    n_timesteps_back: int
    n_timesteps_forward: int
    trend_model_window: int
    trend_model_order: int
    insert_dummy_variable: bool
    days_before: int
    artificial_noise: bool
