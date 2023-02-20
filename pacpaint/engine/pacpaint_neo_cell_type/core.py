"""
Copyright (c) Owkin Inc.
This source code is licensed under the GNU GENERAL PUBLIC LICENSE v3 license found in the
LICENSE file in the root directory of this source tree.

"""
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from pacpaint.models.deepmil import MLP
from pacpaint.engine.pacpaint_bc.trainer import trainer

def train(
    X: torch.Tensor,
    X_ids: np.ndarray,
    y: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    params: dict,
    device="cuda:0",
) -> Tuple[pd.DataFrame, pd.DataFrame, float, torch.nn.Module]:

    mask_val = np.array([i in val_ids for i in X_ids])
    mask_train = np.array([i in train_ids for i in X_ids])

    idx_train = np.where(mask_train)[0]
    idx_val = np.where(mask_val)[0]

    X_ids_train = X_ids[mask_train]
    X_ids_val = X_ids[mask_val]
    assert len(set(X_ids_train).intersection(X_ids_val)) == 0

    y_train = y.iloc[idx_train].values.squeeze()
    y_val = y.iloc[idx_val].values.squeeze()

    train_set = TensorDataset(torch.tensor(idx_train), torch.tensor(y_train))
    val_set = TensorDataset(torch.tensor(idx_val), torch.tensor(y_val))

    model = MLP(
        in_features=X[0].shape[-1], out_features=1, hidden=[128], activation=torch.nn.ReLU()
    )
    weight = np.sum(y[mask_train] == 0) / np.sum(y[mask_train] == 1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]).to(device))

    val_preds, val_auc, model = trainer(
        model=model,
        criterion=criterion,
        X=X,
        X_ids=X_ids,
        train_set=train_set,
        val_set=val_set,
        params=params,
        device=device,
        padded=False,
        metric_patient_wise=False
    )
    return val_preds, val_auc, model


