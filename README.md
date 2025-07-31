![](UTA-DataScience-Logo.png)

# Flower Image Classification and Transfer Learning 🌸

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

* Accuracy vs. Epoch plotted
* Confusion matrix shown
* No overfitting observed due to regularization and augmentation

Model Type	Accuracy
Custom CNN	~80%
MobileNetV2	~92%



### Conclusions

* Transfer learning significantly outperforms small custom CNN
* Even small datasets can achieve strong performance with the right architecture and tuning



### Future Work

* Apply model to wild flower images (generalization test)
* Add test set and implement precision/recall/F1 scoring
* Use data augmentation more aggressively


## How to reproduce results

* On Colab:
   * Open the provided training_model.ipynb notebook
   * Run each cell in sequence
   * Use Google Drive to upload the dataset, or download via Kaggle API

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: Utility functions for file cleanup and plotting
  * data_preparation.ipynb: Organizes raw images into folders by class
  * visualization.ipynb: Visualizes dataset samples
  * training_model.ipynb: Trains CNN and MobileNetV2 models
  * performance.ipynb: Compares model accuracy and loss
  * requirements.txt: List of required packages




### Software Setup
* List all of the required packages.
  
* tensorflow
* matplotlib
* numpy
* pandas
* scikit-learn


### Data

Dataset from Kaggle: Flower Image Dataset
If using Colab:
* Download from Kaggle using API or upload manually to Google Drive
* Then mount Drive and unzip into your project folder

### Training

* Prepare data folders or run data_preparation.ipynb
* Open training_model.ipynb
* Set your base_path to your Drive location
* Run all cells

#### Performance Evaluation

* Evaluate using validation accuracy, loss curves, and confusion matrix
* Compare model outputs visually using performance.ipynb

## Citations

* Kaggle Dataset: https://www.kaggle.com/datasets/aksha05/flower-image-dataset
