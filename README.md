# desastre_response_pipelines

## Overview
In this project there is a web application where an emergency worker can insert a new message and obtain classification results in several categories. The web application will also display visualizations of the data.

## Components
The project is divided into three major blocks:

### 1. ETL pipeline (process_data.py)
Loading of message and category data sets
Merge the two data sets
Data cleaning
Storage in an SQLite database

### 2. ML pipeline (train_classifier.py)
Loading data from the SQLite database
Dividing the dataset into training and testing sets
Creating a word processing and machine learning pipeline
Training and adjusting a model using GridSearchCV
Presentation of the results in the test set
Exporting the final model as a pickle file

### 3. Flask Web App (run.py)
Data visualizations using Plotly in the web application + Flask Web.

## Project Motivation
Third project developed on the Udacity platform in the Data Scientist Nanodegree program, and also the first challenge proposed on the Kaggle platform.
