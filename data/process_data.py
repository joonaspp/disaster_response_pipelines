import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load messages and categories dataset and merge datasets
    
    Parameters:
    messages_filepath: messages dataset
    categories_filepath: categories dataset
    
    Returns:
    df: the dataframe merge
    '''
    
    # load messages and categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how = 'left', on = ['id'])
    
    return df

def clean_data(df):
    '''
    cleaning the dataframe
    
    Parameters:
    df: dataframe
    
    Returns:
    df: cleaned dataframe
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row =  categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    # Remove rows with a related value of 2 from the dataset
    df = df[df['related'] != 2]
    
    return df


def save_data(df, database_filename):
    '''
    saving database
    
    Parameters:
    df: dataframe, database_filename: sql database
    
    '''
    
    #You can do this with pandas to_sql method combined with the SQLAlchemy library. Remember to import SQLAlchemy's create_engine in the first cell of this notebook to use it below
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df_projeto3', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()