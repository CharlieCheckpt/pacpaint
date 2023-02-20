"""
Copyright (c) Owkin Inc.
This source code is licensed under the GNU GENERAL PUBLIC LICENSE v3 license found in the
LICENSE file in the root directory of this source tree.

Train models to:
    - distinguish tumor from normal tiles
    - distinguish tumor cells tiles from stroma tiles
"""
import numpy as np
import pandas as pd
import torch

from pacpaint.data.dataset import PACpAIntDataset
from pacpaint.engine.pacpaint_neo_cell_type.core import train

PARAMS = {"batch_size": 32, "n_ep": 2, "lr": 1.0e-3}
DEVICE = "cuda:0"


def main(model_type: str):
    dataset = PACpAIntDataset()
    X_, _, X_slidenames_, X_ids_ = dataset.load_features(tumor_filter=False)

    if model_type == "neo":
        annots = dataset.load_tumor_annots()
    else:
        annots = dataset.load_tumor_cell_annots()

    X, X_ids, y = [], [], []
    for slidename in annots:
        idx_slide = np.where(X_slidenames_ == slidename)[0][0]
        x = X_[idx_slide]
        x_id = X_ids_[idx_slide]

        # remove padding
        mask_not_padded = x.sum(-1) != 0.0
        x = x[mask_not_padded]

        X.append(x)
        X_ids.extend([x_id] * len(x))
        y.append(annots[slidename].iloc[: len(x)]["annot"].values)

    X = np.concatenate(X, 0)
    X = torch.from_numpy(X)
    X_ids = np.array(X_ids)
    y = pd.Series(np.concatenate(y, 0), index=X_ids)
    train_ids_cv, val_ids_cv = dataset.get_ids_cv_splits(
        labels=y[~y.index.duplicated()], stratify_on_labels=False
    )

    val_aucs = []
    for i, split in enumerate(train_ids_cv):

        _, val_auc, _ = train(
            X=X,
            X_ids=X_ids,
            y=y,
            train_ids=train_ids_cv[split],
            val_ids=val_ids_cv[split],
            params=PARAMS,
            device=DEVICE,
        )

        print(f"Split {i}: AUC={val_auc:.3f}")
        val_aucs.append(val_auc)

    print(f"Mean AUC={np.mean(val_aucs):.3f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["neo", "cell_type"], default="neo")
    args = parser.parse_args()
    main(model_type=args.model)
