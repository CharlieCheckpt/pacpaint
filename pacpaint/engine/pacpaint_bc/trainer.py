"""
Copyright (c) Owkin Inc.
This source code is licensed under the GNU GENERAL PUBLIC LICENSE v3 license found in the
LICENSE file in the root directory of this source tree.

Trainer for Basal/Classic prediction (PACpAInt-B/C).
"""
from typing import Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def trainer(
    model: nn.Module,
    criterion: nn.Module,
    X: torch.Tensor,
    X_ids: np.ndarray,
    train_set: TensorDataset,
    val_set: TensorDataset,
    params: dict,
    device="cuda:0",
    padded=True,
    metric_patient_wise=True
) -> Tuple[pd.DataFrame, float, nn.Module]:
    dataloader = DataLoader(
        train_set,
        shuffle=True,
        pin_memory=False,
        batch_size=params["batch_size"],
        num_workers=0,
        drop_last=True,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    pbar = tqdm(total=len(dataloader))

    val_auc, val_loss, loss = np.nan, np.nan, np.nan
    for epoch in range(params["n_ep"]):
        model.train()
        pbar.reset()
        pbar.set_description(
            f"Epoch[{epoch}]: val_loss: {val_loss:.2f}, val_auc: {val_auc:.2f}"
        )

        for batch in dataloader:
            idx_b, labels_b = batch
            features_b = X[idx_b]

            if padded:
                # padding mask
                mask_b = features_b.sum(-1, keepdim=True) == 0.0

            optimizer.zero_grad()

            if padded:
                logits_b = model.forward(features_b.to(device), mask_b.to(device))
            else:
                logits_b = model.forward(features_b.to(device))

            loss = criterion(logits_b.squeeze(), labels_b.to(device))

            loss.backward()
            optimizer.step()

            pbar.set_description(
                f"Epoch[{epoch}]: val_loss : {val_loss:.2f}, val_auc: {val_auc:.2f}",
                refresh=True,
            )
            pbar.update(1)

        val_preds, _, val_auc = eval(
            model=model,
            criterion=criterion,
            X=X,
            X_ids=X_ids,
            dataset=val_set,
            device=device,
            padded=padded,
            metric_patient_wise=metric_patient_wise
        )

    pbar.close()
    return val_preds, val_auc, model


def eval(
    model: nn.Module,
    criterion: nn.Module,
    X: torch.Tensor,
    X_ids: np.ndarray,
    dataset: TensorDataset,
    device="cuda:0",
    padded=False,
    metric_patient_wise=True
) -> Tuple[pd.DataFrame, float, float]:
    dataloader = DataLoader(dataset, shuffle=False, pin_memory=False, batch_size=64, num_workers=0)

    model.eval()
    with torch.no_grad():
        y, y_hat, ids = [], [], []
        for batch in dataloader:
            idx_b, labels_b = batch
            features_b = X[idx_b]
            ids_b = X_ids[idx_b.numpy()]
            if padded:
                mask_b = features_b.sum(-1, keepdim=True) == 0.0
                logits_b = model.forward(features_b.to(device), mask_b.to(device))
            else:
                logits_b = model.forward(features_b.to(device))

            pred = torch.sigmoid(logits_b).squeeze()
            y.append(labels_b)
            y_hat.append(pred)
            ids.append(ids_b)

        # Loss
        y = torch.cat(y).to(device)
        y_hat = torch.cat(y_hat).to(device)
        loss = criterion(y_hat, y).cpu().numpy()

        ids = np.concatenate(ids)

        # Metric and predictions patient-wise
        y = y.cpu().numpy()
        y_hat = y_hat.cpu().numpy()
        preds = pd.DataFrame({"pred": y_hat, "label": y}, index=ids)
        if metric_patient_wise:
            preds = preds.groupby(preds.index).mean()
        auc = roc_auc_score(y_true=preds.label, y_score=preds.pred)

    return preds, loss, auc
