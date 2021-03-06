# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from csv files and merge based in `id` column.

    Parameters:
    messages_filepath (str): File path of messages csv file
    categories_filepath (str): File path of categories csv file

    Return:
    DataFrame: A dataframe of messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')

    return df


def clean_data(df):
    """
    Clean the data.

    Parameters:
    df (DataFrame): A dataframe of messages and categories to be cleaned

    Returns:
    DataFrame: A cleaned dataframe of messages and categories
    """
    categories = df['categories'].str.split(';', expand=True)

    row = categories.loc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)

    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1, join="inner")
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save the Dataframe into database

    Parameters:
    df (DataFrame): A dataframe of messages and categories
    database_filename (str): File name of database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
