"""
Copyright (c) Owkin Inc.
This source code is licensed under the GNU GENERAL PUBLIC LICENSE v3 license found in the
LICENSE file in the root directory of this source tree.

Extract features from slides using a ResNet50 pre-trained on ImageNet dataset.
"""
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import torch
from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

# To fill
PATH_SLIDES = ""
PATH_TILES_COORDS = ""
SAVE_DIR = ""
SLIDES_FORMAT = ".svs"


class TilesDataset(Dataset):
    def __init__(self, slide: openslide.OpenSlide, tiles_coords: np.ndarray) -> None:
        self.slide = slide
        self.level = int(tiles_coords[0, 0])
        self.tiles_coords = tiles_coords
        self.transform = Compose(
            [ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        )
        self.dz = DeepZoomGenerator(slide, tile_size=224, overlap=0)

    def __getitem__(self, item: int):
        tile_coords = self.tiles_coords[item, 1:3].astype(int)
        im = self.dz.get_tile(level=self.level, address=tile_coords)
        im = self.transform(im)
        return im

    def __len__(self) -> int:
        return len(self.tiles_coords)


def extract_features(
    slide: openslide.OpenSlide, model: torch.nn.Module, tiles_coords: np.ndarray
) -> np.ndarray:
    dataset = TilesDataset(slide=slide, tiles_coords=tiles_coords)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    features = []
    for images in dataloader:
        with torch.inference_mode():
            features_b = model(images)
            features_b = features_b.cpu().numpy()
        features.append(features_b)
    features = np.concatenate(features)
    features = np.concatenate([tiles_coords, features], axis=1)
    return features


def main():
    model = resnet50(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Identity()
    model.eval()

    tiles_coords = pd.read_pickle(PATH_TILES_COORDS)
    for slide_path in tqdm(Path(PATH_SLIDES).glob(f"*/*{SLIDES_FORMAT}")):
        slide = openslide.open_slide(str(slide_path))
        slide_name = slide_path.name
        tiles_coords_ = tiles_coords[slide_name].values
        features = extract_features(slide=slide, model=model, tiles_coords=tiles_coords_)

        features_save_dir = Path(SAVE_DIR) / slide_name
        features_save_dir.mkdir(exist_ok=True, parents=True)
        np.save(features_save_dir / "features.npy", features)


if __name__ == "__main__":
    main()
