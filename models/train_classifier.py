"""
execute train_classifier.py scipt
"""
# Import libraries and load data from database.
import sys
from sqlalchemy import create_engine
import sqlite3

import re
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
import joblib

# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
     """
       This function loads data from the database
       and returns
       X : message features df
       Y : target df
       """
     engine = create_engine(f'sqlite:///{database_filepath}')
     df = pd.read_sql_table('pipelines', engine)
     X = df['message']  # Message Column
     Y = df.iloc[:,4:]  # Classification label
     return X, Y


def tokenize(text):
    """
       Tokenizes and lemmatizes text.

       Parameters:
       text: Text to be tokenized

       Returns:
       clean_tokens: Returns cleaned tokens
       """
    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalise case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
     This function is to build the model pipeline
     for classifing the disaster messages
     Returns:
       cv classification model
        """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators' : [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test):
    """
    This function evaluates the model.
    The function prints the f1 score, precision and recall for each output category of the dataset.
    Arguments:
    model, X_test, Y_test
    """
    y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col, classification_report(Y_test[col], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    This function saves the model as a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
      This function applies the Machine Learning Pipeline:
          1) Extract data from SQLite db
          2) Train ML model on training set
          3) Estimate model performance on test set
          4) Save trained model as Pickle file

      """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y  = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
