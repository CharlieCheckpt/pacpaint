"""
Copyright (c) Owkin Inc.
This source code is licensed under the GNU GENERAL PUBLIC LICENSE v3 license found in the
LICENSE file in the root directory of this source tree.

Infer tumor/normal or tumor cell/stroma using a given model.
"""
from tqdm import tqdm
import pickle
import pandas as pd
import torch
from pacpaint.models.mlp import MLP
from pacpaint.data.dataset import PACpAIntDataset

MODEL_PATH = ""  # TO FILL
SAVE_DIR = ""  # TO FILL
DEVICE = "cuda:0"


def main():
    dataset = PACpAIntDataset()
    X, X_coords, X_slidenames, _ = dataset.load_features(n_tiles=100, tumor_filter=False)

    model = MLP(
        in_features=X[0].shape[-1], out_features=1, hidden=[128], activation=torch.nn.ReLU()
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    preds = {}
    for x, coord, slidename in tqdm(zip(X, X_coords, X_slidenames)):
        x = torch.from_numpy(x).to(DEVICE)
        with torch.inference_mode():
            preds_ = model.forward(x)
            preds_ = preds_.cpu().numpy().squeeze()
            preds_ = pd.DataFrame(
                {"z": coord[:, 0], "x": coord[:, 1], "y": coord[:, 2], "pred": preds_}
            )
            preds[slidename] = preds_

    pickle.dump(preds, open(f"{SAVE_DIR}/tumor_preds.pkl", "wb"))


if __name__ == "__main__":
    main()
