"""Neural operator utilities."""

from robotmath.neural_operators.deeponet_mass_spring import (
    DeepONet1D,
    OperatorTrainConfig,
    make_operator_dataset,
    operator_trajectory_mse,
    simulate_batch_mse,
    train_deeponet,
)
from robotmath.neural_operators.fno_mass_spring import FNO1D, fno_trajectory_mse, train_fno

__all__ = [
    "DeepONet1D",
    "FNO1D",
    "OperatorTrainConfig",
    "fno_trajectory_mse",
    "make_operator_dataset",
    "operator_trajectory_mse",
    "simulate_batch_mse",
    "train_deeponet",
    "train_fno",
]
