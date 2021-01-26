# Disaster Response Pipeline

This repository is created as part of 'Disaster Response Pipeline' project of Data Scientist course.

# Project Motivation

In this project we will be working on developing a classifier which classifies the messages from people during natural disasters. We will be working on data obtained from Figure Eight. The project involves building a model for an API which takes the messages as input and classifies the messages into 36 pre defined classes so that the messages can be assigned to the respective departments. A screenshot of the classifier is attached in this repository.

# Files

1) disaster_messages.csv - This is a csv file provided by Figure Eight which has the messages.
2) disaster_categories.csv - This is a csv file which has the categories of the messages.
3) ETL Pipeline Preparation.ipynb - This is a Jupyter notebook which reads the files disaster_messages.csv and disaster_categories.csv, cleans the data stores the data in a SQL table.
4) ML Pipeline Preparation.ipynb - This is a Jupyter notebook which has the model preparation for message classification.
5) process_data.py - This is a Python file which runs the data preparation.
6) train_classifier.py - This is a Python file which creates the model and saves the model as a pickle file.
7) run.py - This is a Python file which runs an API to take messages from user and classify the messages.

The Structure of the file is as below:

app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- Disaster_Response.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
README.md

# Installations

This project will require Python V3. The libraries required are numpy, pandas, sklearn, pickle, plotly, sqlalchemy, re, NLTK, flask.

# Instructions

1) Navigate to the data folder and run following command: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
This takes the 2 files disaster_messages.csv and disaster_categories.csv as inputs and creates a database DisasterResponse.db
2) Navigate to the model folder and run the following command: python train_classifier.py ../data/DisasterResponse.db classifier.pkl
This takes the database table as input and outputs a pickle file.
3) Navigate to the app folder and run the following command: python run.py
This will run the API which takes the message from user and classify the message.

# Acknowledgement

I would like to thank Figure Eight for providing the data for the project.
