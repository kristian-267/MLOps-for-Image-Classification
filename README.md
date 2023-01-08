# DTU-MLOps-Group7

Project repository for DTU 02476 - MLOps courses in January 2023.

Authors: Chuansheng Liu, Xindi Wu, Chongchong Li, Mouadh Sadani

## Project Description
 
#### Overall goal of the project
The goal of the project is to use convolutional neural network-based architecture to classify images in computer vision.      

#### What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)
Because the task is to classify images, we are going to use [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models) framework to achieve our project goal.     

#### How do you intend to include the framework into your project
From the framework, we will import and modify the needed model. Besides, the framework provides many tools for data processing, tuning, training. We will also use whichever is useful to our project.     

#### What data are you going to run on (initially, may change)
We are going to use [ImageNet 1000 (mini)](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000). It includes 1000 classes. It is a more compressed version of the ImageNet dataset and contains 38,7k images. The ImageNet dataset is used widely for classification challenges and is useful to develop Computer Vision and Deep Learning algorithms.

#### What deep learning models do you expect to use
The model we expect to use is [ResNeSt](https://arxiv.org/pdf/2004.08955.pdf). It is a ResNet variant which stacking several Split-Attention blocks (conposed by featuremap group and split attention operations). It is easy to work with, computational efficient, and universally improves the learned feature representations to boost performance across image classification.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
