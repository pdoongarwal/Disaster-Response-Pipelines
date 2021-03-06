# Disaster Response Pipelines

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Project Descriptions](#descriptions)
4. [Files Descriptions](#files)
5. [Instructions](#instructions)

## Installation <a name="installation"></a>

Packages used:

- pandas
- re
- sys
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- flask
- plotly
- sklearn
- joblib

All packages can be installed using pip.

## Project Motivation <a name="motivation"></a>

In this project, I analyzed disaster data from [Appen (formally Figure 8)](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.
Your project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## File Descriptions <a name = "descriptions"></a>

The project has three components:

1. **ETL Pipeline:**

- `process_data.py` - Data cleaning pipline
- Loads the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. **ML Pipeline:**

- `train_classifier.py` - ML pipline
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. **Flask Web App:**

- Web app enables the user to enter a disaster message, and then view the categories of the message.
- The web app also contains some visualizations that describe the data.

## File Descriptions <a name="files"></a>

The files structure is arranged as below:

    - README.md
    - LICENSE
    - workspace
    	- \app
    		- run.py: Flask file to run the app
    	    - \templates
                - master.html: main page of the web application
                - go.html: result web page
    	- \data
    		- disaster_categories.csv: categories dataset
    		- disaster_messages.csv: messages dataset
    		- DisasterResponse.db: disaster response database
    		- process_data.py: ETL process
    	- \models
    		- train_classifier.py: Train classification model
            - classifier.pkl: Saved model

## Instructions <a name="instructions"></a>

To execute the app follow the instructions:

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3000/
