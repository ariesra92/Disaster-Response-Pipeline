# Disaster Response Pipelines


### Table of Contents

1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [Project Descriptions](#descriptions)
5. [Instructions](#instructions)
6. [Web App Visualizations](#webapp)


## Project Motivation<a name="motivation"></a>

The goal of the project is classifying the disaster messages. The data is taken from 
[Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. 
This project includes a web app where emergency worker can input a new message and get classification results in several
categories. The web app is also displays visualization of the data.

## Installation <a name="installation"></a>

While working on this project, I have used conda virtual enviroment and all the packages
I have used can be found in the `requirement.txt` file.

If you also want to use virtual environment, please use following script to set up
things easily.

`$ conda create --name <env> --file requirement.txt`


## Project Descriptions<a name = "descriptions"></a>
There are three components in this project like the following:

1. **ETL Pipeline:** `process_data.py` includes data cleaning pipeline, details:

- Loads the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. **ML Pipeline:** `train_classifier.py` includes ML pipeline, details:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. **Flask Web App:**

## Instructions <a name="instructions"></a>

To execute the app follow the instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python3 models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python3 run.py`

3. Click the 'PREVIEW' button to open the homepage

## Web App Visualizations <a name="webapp"></a>

***Distribution Graphs***
![Screenshot 1](https://github.com/ariesra92/Disaster-Response-Pipeline/blob/master/pics/ss1.jpg)

***App word search***
![Screenshot 2](https://github.com/ariesra92/Disaster-Response-Pipeline/blob/master/pics/ss2.jpg)
