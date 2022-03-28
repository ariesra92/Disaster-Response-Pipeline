import sys
import pandas as pd 
import numpy as np 
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load data from csv files and merge them
    Args:
        messages_filepath (string): the file path of messages csv file
        categories_filepath (string): the file path of categories csv file
    Return:
        messages (DataFrame): A dataframe of messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    messages = messages.merge(categories, how='inner', on='id')
    return messages


def clean_data(df):
    """
    Clean the Dataframe
    Args:
        df (DataFrame): A dataframe of messages and categories need to be cleaned
    Return:
        df (DataFrame): A cleaned dataframe
      """

    # create a dataframe of the 36 individual category columns
    print(df.head())
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.head(1)

    # extract a list of new column names for categories.
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,].tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # related column max value is 2 we sould convert them to 1.
    categories['related']=categories['related'].replace(2, 1)

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1, join = 'inner')

    # drop duplicates
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filename):
    """
    Save the Dataframe to the sqlite database
    Args:
        df (DataFrame): A dataframe of messages and categories
        database_filename (string): The file name of the database
       """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('cleaned_disaster_data', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        # messages_filepath, categories_filepath, database_filepath = 'disaster_messages.csv', 'disaster_categories.csv', 'disaster_response.db'

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:

        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'disaster_response.db')


if __name__ == '__main__':
    main()