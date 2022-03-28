import sys
import pandas as pd
import numpy as np
import sqlite3
import nltk
import re
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pickle

nltk.download(['wordnet', 'punkt', 'stopwords', 'omw-1.4'])

def load_data(database_filepath):
    """
    Load data from db return data for ML
    Args:
      database_filepath(string): the path of database
    Return:
        X(dataframe): includes features
        Y(dataframe): includes targets
        category_names(list): category names
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('cleaned_disaster_data', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(Y.columns) # categories
    return X, Y, category_names


def tokenize(text):
    """
    Split text into words while eliminating stop words
    Args:
      text(string): the message
    Return:
      lemm_words(list of str): extracted a list of words
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words = word_tokenize(text)

    # Remove stop words
    words = [t for t in words if t not in stopwords.words("english")]

    # Lemmatization
    lemm_words = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemm_words


def build_model():
    """
    Trains the adaboost model with tokenization pipeline. Adaboost model and its parameter already evaluated and chosen
    in ML pipeline preparation notebook. Other models and cv didn't used in this section to speed up the process.
    Args:
      text (string): the message
    Return:
      model (object): returns trained model
    """
    pipeline_ada = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=30)))
    ])

    # parameters_ada = {
    #     'tfidf__use_idf': (True, False),
    #     'clf__estimator__n_estimators': [10, 30]
    # }

    # model = GridSearchCV(pipeline_ada, param_grid=parameters_ada, verbose=10, cv=2)
    model = pipeline_ada

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Calculates the accuracy, f1 score, precision and recall for each output category of the dataset
    Args:
        model: the classification model
        X_test(array): actual test features
        Y_test(array): actual test targets
        category_names(list): category names
    Return:
        result(dataframe): accuracy,f1, precision and recall scores for each label
    """
    y_pred = np.array(model.predict(X_test))
    Y_test = np.array(Y_test)
    metrics = []
    # Calculate performans metrics for each categories.
    for i in range(len(category_names)):
        accuracy = accuracy_score(Y_test[:, i], y_pred[:, i])
        precision = precision_score(Y_test[:, i], y_pred[:, i])
        recall = recall_score(Y_test[:, i], y_pred[:, i])
        f1 = f1_score(Y_test[:, i], y_pred[:, i])

        metrics.append([accuracy, precision, recall, f1])

    # Get result
    result = pd.DataFrame(data=np.array(metrics), index=category_names, columns=['Accuracy', 'Precision', 'Recall', 'F1'])
    print(result.mean)




def save_model(model, model_filepath):
    """
    Save a pickle file of the model
    Args:
        model: the classification model
        model_filepath (str): the path of pickle file
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
    # database_filepath, model_filepath = '../data/disaster_response.db', 'classifier.pkl'
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/disaster_response.db classifier.pkl')


if __name__ == '__main__':
    main()