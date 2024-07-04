# MPAIFR: Missing Persons Age-Invariant Face Recognition

## Overview

This project encompasses various components used in the thesis including the datasets, the AIFR (Age-Invariant Face Recognition) models, the API built in Python, and the web application built using Ruby on Rails. The repository is organized into the following main directories:

- `data/`: Contains the datasets used for training and evaluation.
- `aifr/`: Contains the implementations and training scripts for different models.
- `api/`: Contains the Python API for perfoming face verification.
- `app/`: Contains the Ruby on Rails application.

## Notice

The datasets and the model `.pth` files are stored in a Google Drive folder. You can access and download them from the following link:

[Google Drive - Datasets and Models](https://drive.google.com/drive/folders/1RW-mPgggTJYrA5Yv9mKcNXce6jPBp8JB?usp=sharing)

## Directory Structure

### Data
- **data**
  - `big/`: Large dataset.
  - `fgnet/`: Original FG-NET dataset.
  - `fgnet_split/`: Split FG-NET dataset into positive and negative pairs.
    - `negative/`
    - `positive/`
  - `small/`: Small dataset.
  - `test_big/`: Test set for large dataset.
  - `test_small/`: Test set for small dataset.
  - `train_big/`: Training set for large dataset.
  - `train_small/`: Training set for small dataset.

### Data Preprocessing
- **data_preprocessing**
  - `check_duplicates.py`: Script to check for duplicate images.
  - `compare_gender_age.py`: Script to compare gender and age labels of folders with predicted gender and age labels from the DeepFace model.
  - `split_1000_pos_neg.py`: Script to split dataset into 1000 positive and negative pairs.
  - `split.py`: 80/20 train/test dataset splitting script.

### Data Statistics
- **data_statistics**
  - `data_visualization.py`: Script for visualizing data statistics.
  - `images_per_age_group.py`: Script to analyze images per age group.
  - `images_per_age.py`: Script to analyze images per age.
  - `images_per_gender.py`: Script to analyze images per gender.
  - `images_per_folder.py`: Script to analyze images per folder.

### AIFR
- **aifr**
  - **backbone**
    - `custom.py`: Custom Backbone Neural Network implementation.
  - **models**
    - **multitask**
      - `model.py`: Multi-Task model definition.
      - `training.py`: Training script for Multi-Task model.
      - **results**
        - `80-20big/best-model-85-92.pth`
        - `80-20small/best-model-93-12.pth`
    - **multitask_dal**
      - `model.py`: Multi-Task + DAL model definition.
      - `training.py`: Training script for Multi-Task + DAL model.
      - **results**
        - `80-20big/best-model-86-24.pth`
        - `80-20small/best-model-93-89.pth`
        - `loo/best-model-94-61.txt`: Leave-One-Out evaluation results for MUltitask + DAL model
    - **singletask**
      - `model.py`: Singletask model definition.
      - `training.py`: Training script for singletask model.
      - **results**
        - `80-20big/best-model-85-65.pth`
        - `80-20small/best-model-93-02.pth`
  - **models_evaluation**
    - `config.py`: Configuration for models evaluation.
    - `eval.py`: Evaluation script for all models.
  - **models_training**
    - `config.py`: Configuration for models training.
    - `train.py`: Trainig script for all models.
  - **utils**
    - `image_loader.py`: Utility for loading images.
    - `margin_loss.py`: Implementations of margin loss.
    - `metrics.py`: Metric calculation utilities.
    - `model_handler.py`: Utilities for handling the model from the configuration.
    - `trainer_handler.py`: Utilities for handling the training from the configuration.

### API
- **api**
  - **aifr**
    - **models**
      - `model.py`: Model definitions for API.
      - **results**
        - `80-20small/best-model-93-89.pth`
    - **utils**
      - `margin_loss.py`: Implementation of margin loss for API.
  - **utils**
    - `similarity_handler.py`: Utility for handling similarity calculations.
  - `config.py`: Configuration for API.
  - `Dockerfile`: Docker configuration for API deployment.
  - `main.py`: Main script for running the API.
  - `requirements.txt`: Dependencies for the API.

### App
- **app**
  - `app/`: Application directory.
  - `bin/`: Binary files.
  - `config/`: Configuration files.
  - `db/`: Database migrations and schema.
  - `lib/`: Library files.
  - `public/`: Public assets.
  - `storage/`: File storage.
  - `test/`: Test cases.
  - `tmp/`: Temporary files.
  - `vendor/`: Vendor files.
  - `babel.config.js`: Babel configuration.
  - `config.ru`: Rack configuration.
  - `Gemfile`: Gem dependencies.
  - `Gemfile.lock`: Locked gem dependencies.
  - `LICENSE`: License file.
  - `package.json`: Node.js package configuration.
  - `postcss.config.js`: PostCSS configuration.
  - `Procfile`: Process file for deployment.
  - `Rakefile`: Rake configuration.
  - `yarn.lock`: Yarn lockfile.
