"""
Copyright (c) Owkin Inc.
This source code is licensed under the GNU GENERAL PUBLIC LICENSE v3 license found in the
LICENSE file in the root directory of this source tree.

Train models to predict Basal/Classic in a cross-validation fashion.
"""
import numpy as np
import torch

from pacpaint.data.dataset import PACpAIntDataset
from pacpaint.engine.pacpaint_bc.core import train

PARAMS = {"batch_size": 16, "n_ep": 20, "lr": 1.0e-3}
N_TILES = 8000 

def main():
    dataset = PACpAIntDataset()
    X, _, _, X_ids = dataset.load_features(n_tiles=N_TILES)
    X = torch.from_numpy(X)
    y = dataset.load_purist()
    common_ids = set(y.index).intersection(X_ids)
    train_ids_cv, val_ids_cv = dataset.get_ids_cv_splits(labels=y.loc[common_ids])

    val_aucs = []
    for i, split in enumerate(train_ids_cv):

        _, val_auc, _ = train(
            X=X,
            X_ids=X_ids,
            y=y,
            train_ids=train_ids_cv[split],
            val_ids=val_ids_cv[split],
            params=PARAMS,
            device="cuda:0",
        )
        print(f"Split {i}: AUC={val_auc:.3f}")
        val_aucs.append(val_auc)

    print(f"Mean AUC={np.mean(val_aucs):.3f}")


if __name__ == "__main__":
    main()
