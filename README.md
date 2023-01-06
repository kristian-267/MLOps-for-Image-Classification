# DTU-MLOps-Group7
Project repository for DTU 02476 - MLOps courses in January 2023

## Project Description
 
#### Overall goal of the project
The goal of the project is to use convolutional neural network-based architecture to classify images in computer vision.      

#### What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)
Because the task is to classify images, we are going to use Pytorch Image Models framework to achieve our project goal.     

#### How do you intend to include the framework into your project
From the framework, we will import and modify the needed model. Besides, the framework provides many tools for data processing, tuning, training. We will also use whichever it useful to our project.     

#### What data are you going to run on (initially, may change)
We are going to use ImageNet 1000 (mini). It includes 1000 samples from ImageNet dataset.      

#### What deep learning models do you expect to use
The model we expect to use is ResNeSt. It is a ResNet variant which stacking several Split-Attention blocks (conposed by featuremap group and split attention operations). It is easy to work with, computational efficient, and universally improves the learned feature representations to boost performance across image classification.