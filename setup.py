from setuptools import setup

setup(
    name='pacpaint',
    version='1.0.0',
    install_requires=[
        "torch",
        "torchvision",
        "openslide-python",
        "pandas",
        "tqdm",
        "scikit-learn"
    ]
)
