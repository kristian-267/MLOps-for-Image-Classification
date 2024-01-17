# MLOps for Image Classification

## Authors: 
Chuansheng Liu, Xindi Wu, Chongchong Li, Mouadh Sadani

## Project Description

### Overview
This repository is dedicated to implementing MLOps (Machine Learning Operations) for image classification tasks. The project combines the principles of DevOps with Machine Learning to streamline the development, deployment, and maintenance of image classification models.

### Features
- Image classification with state-of-the-art machine learning models.
- Cloud deployment using FastAPI for easy accessibility.
- Data drift detection to maintain model accuracy over time.
- Comprehensive testing scripts for data and model integrity.
- Containerization with Docker for consistent development and deployment environments.

### What framework do we use?
We use [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models) framework to achieve our project goal. From the framework, we importe and modify the needed model. Besides, the framework provides many tools for data processing, tuning, and training.   

### What data do we use to run on?
We use [ImageNet 1000 (mini)](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000). It includes 1000 classes. It is a more compressed version of the ImageNet dataset and contains 38.7k images. The ImageNet dataset is used widely for classification challenges and is useful to develop Computer Vision and Deep Learning algorithms.

### What deep learning model do we use?
The model we use is [ResNeSt](https://arxiv.org/pdf/2004.08955.pdf). It is a ResNet variant which stacking several Split-Attention blocks (conposed by featuremap group and split attention operations). It is easy to work with, computational efficient, and universally improves the learned feature representations to boost performance across image classification.

## Installation
- Copy repository:
```bash
git clone https://github.com/kristian-267/MLOps-for-Image-Classification.git
```
- Configure Environment:
```bash
cd MLOps-for-Image-Classification
pip install -r requirements
pip install -r requirements_tests
```
or
```bash
make requirements
```
- Download data:
```bash
dvc pull
```
or download data from: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000

## Usage
- Train model:
```bash
python src/models/train_model.py
```
or
```bash
make train
```
- Inference:
```bash
python src/models/predict_model.py
```
or
```bash
make predict
```
- Run unittest with coverage
```bash
coverage run --source=./src -m pytest tests/
```
or
```bash
make tests
```
- Create API (needs [Signoz](https://signoz.io))
```bash
make api
```

## Project Organization

    ├── LICENSE
    │
    ├── Makefile           <- Makefile with commands like `make train`.
    │
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── app                <- A fastapi to do inference.
    │
    ├── conf
    │   ├── data           <- Configurations for dataset.
    │   └── experiment     <- Configurations for training.
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── model_store        <- Applications for local and cloud deployment.
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
    │
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
    ├── tests              <- Unit tests code
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



<p><small>This project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Project Checklist

### Week 1

- [x] Create a git repository
- [x] Make sure that all team members have write access to the github repository
- [x] Create a dedicated environment for you project to keep track of your packages
- [x] Create the initial file structure using cookiecutter
- [x] Fill out the make_dataset.py file such that it downloads whatever data you need and
- [x] Add a model file and a training script and get that running
- [x] Remember to fill out the requirements.txt file with whatever dependencies that you are using
- [x] Remember to comply with good coding practices (pep8) while doing the project
- [x] Do a bit of code typing and remember to document essential parts of your code
- [x] Setup version control for your data or part of your data
- [x] Construct one or multiple docker files for your code
- [x] Build the docker files locally and make sure they work as intended
- [x] Write one or multiple configurations files for your experiments
- [x] Used Hydra to load the configurations and manage your hyperparameters
- [x] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally, consider running a hyperparameter optimization sweep.
- [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

- [x] Write unit tests related to the data part of your code
- [x] Write unit tests related to model construction and or model training
- [x] Calculate the coverage.
- [x] Get some continuous integration running on the github repository
- [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
- [x] Create a trigger workflow for automatically building your docker images
- [x] Get your model training in GCP using either the Engine or Vertex AI
- [x] Create a FastAPI application that can do inference using your model
- [x] If applicable, consider deploying the model locally using torchserve
- [x] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

- [x] Check how robust your model is towards data drifting
- [x] Setup monitoring for the system telemetry of your deployed model
- [x] Setup monitoring for the performance of your deployed model
- [x] If applicable, play around with distributed data loading
- [x] If applicable, play around with distributed model training
- [x] Play around with quantization, compilation and pruning for you trained models to increase inference speed

## License
The project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements
Credits to Chuansheng Liu, Xindi Wu, Chongchong Li, Mouadh Sadani. The external resources [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models), [ImageNet 1000 (mini)](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000), [ResNeSt](https://arxiv.org/pdf/2004.08955.pdf), and [Signoz](https://signoz.io) are used in the project.
