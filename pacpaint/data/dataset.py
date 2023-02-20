"""
Copyright (c) Owkin Inc.
This source code is licensed under the GNU GENERAL PUBLIC LICENSE v3 license found in the
LICENSE file in the root directory of this source tree.

Data loading class.
"""
from typing import Tuple, Dict
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

# To fill : path to features directory.
# Features directory should contains folder with names being the slide names, and features
# should be located inside each subfolder. That is something something of the form:
# TCGA-3E-AAAY-01Z-00-DX1.815B28D5-B285-49F5-8C41-83E424E7C4E0.svs/features.npy
PATH_FEATURES_DIR = Path("")
# To fill : path to tumor annotations
PATH_TUMOR_ANNOTS = Path("")
# To fill: path to tumor predictions (once obtained)
PATH_TUMOR_PREDS = Path("")


class PACpAIntDataset:
    def __init__(self) -> None:
        pass

    def get_patient_id_from_slidename(self, slidename: str) -> str:
        """Returns patient id given the patient slide name.
        For TCGA for instance patient_id = slidename[:12].
        """
        raise NotImplementedError

    def load_purist(self) -> pd.Series:
        """Loads PurIST labels and returns a pandas Series (n_patients, 1) of binary values
        such that 1 <-> basal, 0 <-> classic. Index must be patient id.
        """
        raise NotImplementedError


    def load_comps(self) -> pd.DataFrame:
        """Loads molecular components and returns a pandas DataFrame (n_patients, 4),
        where each column correspond to the value of each of the 4 components Basal, Classical, StromaActiv, StromaInactiv.
        Index must be patient id.
        """
        raise NotImplementedError

    def load_tumor_annots(self) -> Dict[str, pd.DataFrame]:
        """Loads tumor predictions and return a dictionary such that key is slide name, and
        value is a DataFrame where each row correspond to a tile and columns are
        `z` (deepzoom level), `x` (deepzoom x), `y` (deepzoom y), `annot` (tumor annotation: "tumor" or "normal").
        """
        annots = pd.read_pickle(PATH_TUMOR_ANNOTS)
        for s in annots:
            annot = annots[s].copy()
            annot["annot"] = (annot["annot"] == "tumor") * 1.0
            annots[s] = annot
        return annots

    def load_tumor_cell_annots(self) -> Dict[str, pd.DataFrame]:
        """Loads tumor predictions and return a dictionary such that key is slide name, and
        value is a DataFrame where each row correspond to a tile and columns are
        `z` (deepzoom level), `x` (deepzoom x), `y` (deepzoom y), `annot` (tumor annotation: "cell_tum" or "stroma").
        """
        raise NotImplementedError

    def load_tumor_preds(self) -> Dict[str, pd.DataFrame]:
        """Loads tumor predictions and return a dictionary such that key is slide name, and
        value is a DataFrame where each row correspond to a tile and columns are
        `z` (deepzoom level), `x` (deepzoom x), `y` (deepzoom y), `pred` (tumor prediction).
        """
        tumor_preds = pd.read_pickle(PATH_TUMOR_PREDS)
        return tumor_preds

    def load_features(
        self, n_tiles=10_000, tumor_filter=True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads features for each slide."""
        features_paths = list(PATH_FEATURES_DIR.glob("*/features.npy"))
        X = np.zeros((len(features_paths), n_tiles, 2051), dtype=np.float32)
        X_slidenames, X_ids = [], []
        tumor_preds = self.load_tumor_preds()

        for i, p in enumerate(tqdm(features_paths)):
            # Load features
            x = np.load(p, mmap_mode="r")[:n_tiles].copy()
            slidename = p.parents[0].name
            patient_id = self.get_patient_id_from_slidename(slidename)

            # Filter out non tumor tiles
            if slidename in tumor_preds and tumor_filter:
                tumor_preds_ = tumor_preds[slidename].iloc[: len(x)]
                mask_is_tumor = (tumor_preds_["pred"] > 0.5).values
                x = x[mask_is_tumor]
            elif tumor_filter:
                print(f"Could not find tumor predictions for {slidename}")

            # Fill the global features, slidenames and patient ids matrices
            X[i, : len(x), :] = x
            X_slidenames.append(slidename)
            X_ids.append(patient_id)

        return X[..., 3:], X[..., :3], np.array(X_slidenames), np.array(X_ids)

    def get_ids_cv_splits(
        self, labels: pd.Series, stratify_on_labels=True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Get patients for cross-validation"""
        cv_train_ids, cv_val_ids = {}, {}
        if stratify_on_labels:
            rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=777)
            for i, (train_idx, val_idx) in enumerate(rskf.split(labels, labels)):
                cv_train_ids[f"split_{i}"] = labels.iloc[train_idx].index.values.squeeze()
                cv_val_ids[f"split_{i}"] = labels.iloc[val_idx].index.values.squeeze()
        else:
            rkf = RepeatedKFold(n_splits=5, n_repeats=5)
            for i, (train_idx, val_idx) in enumerate(rkf.split(labels)):
                cv_train_ids[f"split_{i}"] = labels.iloc[train_idx].index.values.squeeze()
                cv_val_ids[f"split_{i}"] = labels.iloc[val_idx].index.values.squeeze()

        return cv_train_ids, cv_val_ids
