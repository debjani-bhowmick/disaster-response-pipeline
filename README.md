# Disaster Response Pipeline Udacity Project

## Table of Contents
1.Description

2.Getting Started

>Dependencies
>Installing
>Executing Program
>File Description
>Additional Material

3.Authors

4.License

5.Acknowledgement

6.Screenshots

## Description
This project and the associated code are meant to serve as an assignment project for the partial fulfilment of the Udacity Data Scientist Nanodegree.The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis. The primary goal is to analyse the data and present the findings. We provide below details related to the motivation for this work, installation of the code, main findings etc below.

This project is divided in the following key sections:

Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
Build a machine learning pipeline to train the which can classify text message in various categories
Run a web app which can show model results in real time

## Getting Started

### Dependencies
Python 3.5+
Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
Natural Language Process Libraries: NLTK
SQLlite Database Libraqries: SQLalchemy
Model Loading and Saving Library: Pickle
Web App and Data Visualization: Flask, Plotly

### Installing
To clone the git repository:

git clone debjani-bhowmick/disaster-response-pipeline

### Executing Program:
You can run the following commands in the project's directory to set up the database, train model and save the model.

To run ETL pipeline to clean data and store the processed data in the database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/

### File Description
This project structure is divided into three directories:
* app/templates/*: templates/html files for web app.  run.py: This file can be used to launch the Flask web app used to classify disaster messages

* data/process_data.py: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

* models/train_classifier.py: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use


### Additional Material
In the data and models folder you can find two jupyter notebook that will help you understand how the model works step by step:

ETL Preparation Notebook: learn everything about the implemented ETL pipeline
ML Pipeline Preparation Notebook: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn
You can use ML Pipeline Preparation Notebook to re-train the model or tune it through a dedicated Grid Search section.


## Licensing, Authors, Acknowledgements
### Author: Debjani Bhowmick
