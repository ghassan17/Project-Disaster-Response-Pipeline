# Data Engineering for Data Scientists 

## Table of Contents
 * [Installations](#installations)
 * [Overview](#overview)
 * [Project Structure](#project-structure)
 * [Scripts](#scripts)
 * [Instructions and Execution](#instructions-and-Execution)
 * [Acknowledgements](#Acknowledgements)

### Installations 

 sqlite3 · pickle · pandas · sys · sklearn · nltk · sqlalchemy · flask · json · plotly · re 


### Overview – Disaster Response Pipeline 
In this project you're going to be analysing thousands of real messages provided by figure8 that were sent through natural disasters either via social media or directly to disaster response organisations. 
 
You’ll building an ETL pipeline that processes message and category data from CSV files and load them into a SQL database, which your ML pipeline will then read from to create and save a multi-output supervised learning model. 
 
Then your web app will extract data from this database to provide data visualisation and use your model to classify new messages for 36 categories. 
 
Machine learning is critical to helping different organisations understand which messages are relevant to them and which messages to prioritise During these disasters is when we have the least capacity to filter out messages that matter and find basic messages such as using keywords searches to provide trivial results.

### Project Structure 

1. **ETL Pipeline**
A Python script, `process_data.py`, writes a data cleaning pipeline that:

 - Loads the ‘messages’ and ‘categories’ datasets
 - Merges the two datasets
 - Cleans the data
 - Stores it in a SQLite database
 
 
2. **ML Pipeline**
A Python script, `train_classifier.py`, writes a machine learning pipeline that:

 - Loads data from the SQLite database
 - Splits the dataset into training and test sets
 - Builds a text processing and machine learning pipeline
 - Trains and tunes a model using GridSearchCV
 - Outputs results on the test set
 - Exports the final model as a pickle file
 
A jupyter notebook `ML Pipeline Preparation` was used to do EDA to prepare the train_classifier.py python script. 

3. **Flask Web App**
The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. The outputs are shown below:
 

### Scripts

```bash
├── data:
   └── ‘process_data.py’ file contain the script to create ETL pipeline # data cleaning pipeline  
   └── disaster_categories.csv # data to process
   └── disaster_messages.csv # data to process  
  
├── ETL-Pipeline-Peparation:
   └── ETL Pipeline Preparation.ipynb
   └── Disaster_Response_DB.db

├── ML-Pipeline-Peparation:
      └──  classifer.pkl
      └── ML Pipeline Preparation.ipynb
├── models:
      └── train_classifier.py` file contain the script to create ML pipeline
├── app:
    │ └── templates
    │  ├── go.html
    │  ├── master.html
    │  
    └── run.py
```



      
 ### Instructions 

To execute the app follow the instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`


### Acknowledgments

* [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
* [Appen](https://appen.com/) for providing the relevant dataset to train the model 


