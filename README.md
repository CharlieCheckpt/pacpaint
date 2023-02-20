# PACpAInt: a histology-based deep learning model uncovers the extensive intratumor molecular heterogeneity of pancreatic adenocarcinoma
Code for the models described in the paper "PACpAInt: a histology-based deep learning model uncovers the extensive intratumor 
molecular heterogeneity of pancreatic adenocarcinoma".
The study presents an approach to predict molecular subtypes of Pancreatic Ductal adenocarcinoma from H&E histology slides using deep learning.

!["PACpAInt"](./assets/pacpaint.png)

Four machine learning models are described in the aforementioned paper:
- PACpAInt-Neo: predict tumor/non-tumor regions (tile-level).
- PACpAInt-Cell Type: predict tumor cells/stroma regions (tile-level).
- PACpAInt-BC: predict basal/classic molecular subtypes (slide-level).
- PACpAInt-Comp: predict basal/classic/stroma active/stroma inactive molecular components (slide-level).

# Install

To install openslide, do:
```bash
apt-get update -qq && apt-get install openslide-tools libgeos-dev -y 2>&1
```

Then to install pacpaint and its dependencies:
```bash
pip install -e .
```

# Dataset
The general class to load data is PACpAInt dataset present in `pacpaint/data/dataset.py`.
To make it work on your data, some methods should be implemented (see file).

# Feature Extraction
## ResNet50 pre-trained with supervised learning on ImageNet dataset
In the file `pacpaint/engine/feature_extraction/imagenet_extraction.py`, we provide an example on how to extract features
from each tiles, given their coordinates, using a ResNet50 pre-trained on ImageNet dataset.

## Wide ResNet50 x2 pre-training with MoCo-v2 on histology images
The feature extractor used in our study is a Wide ResNet50 x2, that was pre-trained with MoCo v2 on 4 million tiles from 
TCGA-COAD dataset. 

The code to train such model is available here: https://github.com/facebookresearch/moco.

Details about the parameters used for the training are given in our paper.

# Predictive models training

## Predict tumor/non-tumor (PACpAInt-Neo) and tumor cell/stroma (PACpAInt-Cell Type)

The script to train models to predict tumor and non-tumor regions or tumor cell / stroma 
regions in a cross-validated fashion is available here: `pacpaint/engine/pacpaint_neo_cell_type/train.py`.

It can be launched like this:
```bash
python pacpaint/engine/pacpaint_neo_cell_type/train.py --model neo
```

## Predict basal/classic molecular subtypes (PACpAInt-BC)

The script to train models to predict basal/classic molecular subtypes in 
a cross-validated fashion is available here: `pacpaint/engine/pacpaint_bc/train.py`.

It can be launched like this:
```bash
python pacpaint/engine/pacpaint_bc/train.py
```


## Predict basal/classic/stroma activ/stroma inactive molecular components (PACpAInt-BC)

The script to train models to predict the four molecular components in
a cross-validated fashion is available here: `pacpaint/engine/pacpaint_comp/train.py`.

It can be launched like this:
```bash
python pacpaint/engine/pacpaint_comp/train.py
```
