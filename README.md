![](UTA-DataScience-Logo.png)

# Flower Image Classification and Transfer Learning

* **One Sentence Summary** This repository contains an image classification pipeline trained on the Kaggle "Flower Image Dataset" using convolutional neural networks and transfer learning in Google Colab.
https://www.kaggle.com/datasets/aksha05/flower-image-dataset/data

## Overview

  * **Definition of the tasks / challenge**  The task is to classify images of flowers into five categories — daisy, dandelion, rose, sunflower, and tulip — using a dataset of color images. The challenge lies in training a model that generalizes well to unseen flower images despite class imbalance and limited data.
  
  * **Your approach** This project uses a CNN-based architecture built with TensorFlow/Keras. The pipeline includes preprocessing, image augmentation, and two modeling approaches: one custom CNN and one using transfer learning (e.g. MobileNetV2). The dataset is split into training and validation subsets.
   
  * **Summary of the performance achieved** The best model achieved ~92% validation accuracy using transfer learning with fine-tuning. Early stopping and model checkpoints were used to avoid overfitting and retain the best version of the model.

## Summary of Workdone

### Data

* Data:
  * Type: JPEG color images
  * Input: Folder-based dataset with subfolders named after flower categories
  * Size: 5 classes, ~500+ images total
  * Split:
          * Training: 80%
          * Validation: 20%

#### Preprocessing / Clean up

* Moved all images into class-labeled subfolders (daisy/, rose/, etc.)
* Normalized pixel values to [0, 1] using Rescaling(1./255)
* Resized all images to 224x224

#### Data Visualization

* Plotted sample images from each class
* Verified class imbalance
* Observed varying background noise and color patterns

### Problem Formulation

* Define:
  * Input: RGB flower images (224x224x3)
  * Output: One of five classes (multiclass classification)

### Training

* Describe the training:
  * Platform: Google Colab
  * Hardware: GPU runtime
  * Epochs: 8
  * Callbacks: ModelCheckpoint, ReduceLROnPlateau
  * Difficulties: Initial directory setup and label parsing
  * Resolution: Used filename parsing and automated folder reorganization

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.
