"""
Copyright (c) Owkin Inc.
This source code is licensed under the GNU GENERAL PUBLIC LICENSE v3 license found in the
LICENSE file in the root directory of this source tree.

Trainer for Molecular components prediction (PACpAInt-Comp).
"""
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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
    target_names: List[str],
    device="cuda:0",
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

    mean_val_corr, val_loss, loss = np.nan, np.nan, np.nan
    for epoch in range(params["n_ep"]):
        model.train()
        pbar.reset()
        pbar.set_description(
            f"Epoch[{epoch}]: val_loss: {val_loss:.2f}, mean_val_corr: {mean_val_corr:.2f}"
        )

        for batch in dataloader:
            idx_b, labels_b = batch
            features_b = X[idx_b]

            # padding mask
            mask_b = features_b.sum(-1, keepdim=True) == 0.0

            optimizer.zero_grad()

            # We assume the coords and level are not present
            preds_b = model.forward(features_b.to(device), mask_b.to(device))

            loss = criterion(preds_b.squeeze(), labels_b.to(device))

            loss.backward()
            optimizer.step()

            pbar.set_description(
                f"Epoch[{epoch}]: val_loss : {val_loss:.2f}, mean_val_corr: {mean_val_corr:.2f}",
                refresh=True,
            )
            pbar.update(1)

        val_preds, _, val_corrs = eval(
            model=model,
            criterion=criterion,
            X=X,
            X_ids=X_ids,
            dataset=val_set,
            target_names=target_names,
            device=device,
        )
        mean_val_corr = np.mean([c for c in val_corrs.values()])

    pbar.close()
    return val_preds, val_corrs, model


def eval(
    model: nn.Module,
    criterion: nn.Module,
    X: torch.Tensor,
    X_ids: np.ndarray,
    dataset: TensorDataset,
    target_names: List[str],
    device="cuda:0",
) -> Tuple[pd.DataFrame, float, float]:
    dataloader = DataLoader(dataset, shuffle=False, pin_memory=False, batch_size=64, num_workers=0)

    model.eval()
    with torch.no_grad():
        y, y_hat, ids = [], [], []
        for batch in dataloader:
            idx_b, labels_b = batch
            features_b = X[idx_b]
            ids_b = X_ids[idx_b.numpy()]
            mask_b = features_b.sum(-1, keepdim=True) == 0.0

            preds_b = model.forward(features_b.to(device), mask_b.to(device))

            y.append(labels_b)
            y_hat.append(preds_b)
            ids.append(ids_b)

        # Loss
        y = torch.cat(y).to(device)
        y_hat = torch.cat(y_hat).to(device)
        loss = criterion(y_hat, y).cpu().numpy()

        ids = np.concatenate(ids)

        # Metric and predictions patient-wise
        y = y.cpu().numpy()
        y_hat = y_hat.cpu().numpy()
        preds = pd.DataFrame(
            {
                **{f"pred_{target_names[i]}": y_hat[:, i] for i in range(y_hat.shape[1])},
                **{f"label_{target_names[i]}": y[:, i] for i in range(y.shape[1])},
            },
            index=ids,
        )
        # Get predictions patient-wise
        preds = preds.groupby(preds.index).mean()
        corrs = {}
        for t in target_names:
            corr, _ = pearsonr(preds[f"label_{t}"], preds[f"pred_{t}"])
            corrs[t] = corr

    return preds, loss, corrs
