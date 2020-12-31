# Disaster Response Pipeline Udacity Project

## Table of Contents
1. [Description](Description)
2. [Getting Started](Getting_Started)
   * [Dependencies](Dependencies)
   * [Installing](Installing)
   * [Executing Program](Executing_Program)
   * [File Description](File_Description)
   * [Additional Material](Additional_Material)
3. [Authors](Authors)
4. [License](License)
5. [Acknowledgement](Acknowledgement)
6. [Screenshots](Screenshots)

## Description <a name="Description"></a>
This project and the associated code are meant to serve as an assignment project for the partial fulfilment of the Udacity Data Scientist Nanodegree.The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis. The primary goal is to analyse the data and present the findings. We provide below details related to the motivation for this work, installation of the code, main findings etc below.

This project is divided in the following key sections:

Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
Build a machine learning pipeline to train the which can classify text message in various categories
Run a web app which can show model results in real time

## Getting Started <a name="Getting_Started"></a>

### Dependencies <a name=" Dependencies"></a>
Python 3.5+ <br>
Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn <br>
Natural Language Process Libraries: NLTK <br>
SQLlite Database Libraqries: SQLalchemy <br>
Model Loading and Saving Library: Pickle <br>
Web App and Data Visualization: Flask, Plotly <br>

### Installing <a name="Installing"></a>
To clone the git repository:

```[git clone debjani-bhowmick/disaster-response-pipeline](https://github.com/debjani-bhowmick/disaster-response-pipeline)```

### Executing Program <a name="Executing_Program"></a>
You can run the following commands in the project's directory to set up the database, train model and save the model.

* Go to `data` folder to run ETL pipeline. This will clean the data and store the processed data in the database. Type
``` python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db```

* Go to `model` folder to run the ML pipeline. This will load the data from DB, trains classifier and saves the classifier as a pickle file. To exicute this function, from the terminal type 
```python train_classifier.py data/disaster_response_db.db models/classifier.pkl```

* Go to `app` folder from your terminal and type `python run.py` to run your web app.

### File Description <a name=" File_Description"></a>
This project structure is divided into three directories:

<b> app/templates/*:</b> templates/html files for web app.

<b> data/process_data/.py:</b> Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

<b> models/train_classifier.py:</b> A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use

<b> run.py:</b> This file can be used to launch the Flask web app used to classify disaster messages


### Additional Material <a name=" Additional_Material"></a>
In the `data` and `models` folder you can find two jupyter notebook that will help you understand how the model works step by step:

<b> ETL Preparation Notebook:</b> learn everything about the implemented ETL pipeline

<b> ML Pipeline Preparation Notebook:</b> look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn. Have some guidlines how the model can be improved for better accuracy.You can use ML Pipeline Preparation Notebook to re-train the model or tune it through a dedicated Grid Search section.

The Screenshots of the web app are provided in `Screenshots` folder.


### Lic<b>ensing, Authors, Acknowledgements <a name=" Licensing, Authors, Acknowledgements"></a>
<b> Author:</b> Debjani Bhowmick
<b> Acknowledgements: </b>Udacity for providing an amazing Data Science Nanodegree Program
