import torch

from covid19_ml.types_config import Optimizer, OptimizerConfig


def create_optimizer(
    model, optim_config: OptimizerConfig, parameters
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    opt_type: Optimizer = Optimizer(optim_config.type)
    if opt_type == Optimizer.Adam:
        optimizer = torch.optim.Adam(
            parameters,
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
        )
    elif opt_type == Optimizer.AdamW:
        optimizer = torch.optim.AdamW(  # type: ignore
            parameters,
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
        )
    elif opt_type == Optimizer.SparseAdam:
        optimizer = torch.optim.SparseAdam(  # type: ignore
            parameters,
            lr=optim_config.learning_rate,
        )
    elif opt_type == Optimizer.NAdam:
        optimizer = torch.optim.NAdam(  # type: ignore
            parameters,
            lr=optim_config.learning_rate,
        )
    elif opt_type == Optimizer.RAdam:
        optimizer = torch.optim.RAdam(  # type: ignore
            parameters,
            lr=optim_config.learning_rate,
        )
    elif opt_type == Optimizer.SGD:
        optimizer = torch.optim.SGD(  # type: ignore
            parameters,
            lr=optim_config.learning_rate,
            momentum=optim_config.momentum,
            weight_decay=optim_config.weight_decay,
            nesterov=True,
        )
    elif opt_type == Optimizer.ASGD:
        optimizer = torch.optim.ASGD(  # type: ignore
            parameters,
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
        )
    elif opt_type == Optimizer.Rprop:
        optimizer = torch.optim.Rprop(  # type: ignore
            parameters,
            lr=optim_config.learning_rate,
        )
    elif opt_type == Optimizer.RMSprop:
        optimizer = torch.optim.RMSprop(  # type: ignore
            parameters,
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
        )
    elif opt_type == Optimizer.Adadelta:
        optimizer = torch.optim.Adadelta(  # type: ignore
            parameters,
            lr=optim_config.learning_rate,
            rho=optim_config.learning_rate_decay,
            weight_decay=optim_config.weight_decay,
        )
    elif opt_type == Optimizer.Adagrad:
        optimizer = torch.optim.Adagrad(  # type: ignore
            parameters,
            lr=optim_config.learning_rate,
            lr_decay=optim_config.learning_rate_decay,
            weight_decay=optim_config.weight_decay,
        )
    elif opt_type == Optimizer.Adamax:
        optimizer = torch.optim.Adamax(  # type: ignore
            parameters,
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
        )
    elif opt_type == Optimizer.LBFGS:
        optimizer = torch.optim.LBFGS(  # type: ignore
            parameters,
            lr=optim_config.learning_rate,
        )
    else:
        raise ValueError("Unknown optimizer type")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=10, min_lr=5e-5
    )
    return optimizer, scheduler  # type: ignore
