# Disaster Response Pipeline Udacity Project

### Table of Contents
1. [Description](Description)
2. [Getting Started](Getting Started)
  * [Dependencies](Dependencies)
  * [Installing](Installing)
  * [Executing Program](Executing Program)
  * [File Description](File Description)
  * [Additional Material](Additional Material)
3. [Authors](Authors)
4. [License](License)
5. [Acknowledgement](Acknowledgement)
6. [Screenshots](Screenshots)

### Description <a name="Description"></a>
This project and the associated code are meant to serve as an assignment project for the partial fulfilment of the Udacity Data Scientist Nanodegree.The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis. The primary goal is to analyse the data and present the findings. We provide below details related to the motivation for this work, installation of the code, main findings etc below.

This project is divided in the following key sections:

Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
Build a machine learning pipeline to train the which can classify text message in various categories
Run a web app which can show model results in real time

### Getting Started <a name="Getting Started"></a>

#### Dependencies <a name=" Dependencies"></a>
Python 3.5+ <br>
Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn <br>
Natural Language Process Libraries: NLTK <br>
SQLlite Database Libraqries: SQLalchemy <br>
Model Loading and Saving Library: Pickle <br>
Web App and Data Visualization: Flask, Plotly <br>

#### Installing <a name="Installing"></a>
To clone the git repository:

[git clone debjani-bhowmick/disaster-response-pipeline](https://github.com/debjani-bhowmick/disaster-response-pipeline)

#### Executing Program <a name="Executing Program"></a>
You can run the following commands in the project's directory to set up the database, train model and save the model.

* To run ETL pipeline that clean the data and store the processed data in the database. Type python data/process_data.py in your terminal, which will call data/disaster_messages.csv data/disaster_categories.csv and will save the processed data in  data/disaster_response_db.db
* To run the ML pipeline which loads data from DB, trains classifier and saves the classifier as a pickle file. Type python `models/train_classifier.py` in your terminal, which will call `data/disaster_response_db.db`  and will save trained model in `models/classifier.pkl`
* Run the command python run.py from the app's directory to run your web app.

#### File Description <a name=" File Description"></a>
This project structure is divided into three directories:

* app/templates/*: templates/html files for web app.  run.py: This file can be used to launch the Flask web app used to classify disaster messages

* data/process_data.py: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

* models/train_classifier.py: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use


#### Additional Material <a name=" Additional Material"></a>
In the data and models folder you can find two jupyter notebook that will help you understand how the model works step by step:

ETL Preparation Notebook: learn everything about the implemented ETL pipeline
ML Pipeline Preparation Notebook: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn. Have some guidlines how the model can be improved for better accuracy.
You can use ML Pipeline Preparation Notebook to re-train the model or tune it through a dedicated Grid Search section.


#### Licensing, Authors, Acknowledgements <a name=" Licensing, Authors, Acknowledgements"></a>
#### Author: Debjani Bhowmick
