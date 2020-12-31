# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
############################################################


def load_data(messages_filepath, categories_filepath):
    """
    Load files from filepaths and combine them.
    Args:
    messages_filepath: str, Path to the CSV file containing messages
    categories_filepath: str, Path to the CSV file containing categories
    Return:
    df : pandas DataFrame, Combined data containing messages and categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how='outer', on='id')
    return df


def clean_data(df):
    """
    Clean the data and convert the category part
    of the data to numeric, and drop the original category column.
    Args:
        df: Combined data containing messages and categories
    Returns:
        df: Combined data containing messages and categories
         with categories cleaned up
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # Fix the categories columns name
    row = categories.iloc[[1]]
    category_colnames = [x[:-2] for x in row.values[0]]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # Removing entry that is non-binary
    df = df[df['related'] != 2]
    print('Duplicates remaining:', df.duplicated().sum())
    return df


def save_data(df, database_filename):
    """
    Save Data to SQLite Database Function
    Args:
        df : Combined data containing messages and
        categories with categories cleaned up
        database_filename : Path to SQLite destination database
    """
    engine = create_engine('sqlite:///' + database_filename)
    table_name = database_filename.replace(".db", "") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    return df


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[
                                                                    1:]

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
              'DisasterResponse.db')


if __name__ == '__main__':
    main()