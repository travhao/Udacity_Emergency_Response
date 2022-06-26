# import libraries
import sys
import pandas as pd
import re


def load_data(messages_filepath, categories_filepath):
#     load datasets and merge
    msg = pd.read_csv(messages_filepath)
    ctg = pd.read_csv(categories_filepath)
    df = pd.merge(msg, ctg, on = 'id')
    return df


def clean_data(df):
#     create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)
    
# select the first row of the categories dataframe
    row = categories.iloc[0]
    
# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
    category_colnames = []
    for item in row:
        test = re.findall(r"[a-zA-Z_]+",item)
        category_colnames.append(test[0])
    print(category_colnames)
    
# rename the columns of `categories`
    categories.columns = category_colnames  
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
        
    # drop the original categories column from `df`
    df.drop(['categories'], inplace = True, axis=1)

# concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)

#drop duplicates
    df = df.drop_duplicates()
    
    return df
# df[['id', 'message', 'original', 'genre', 'related','request']]
    
  
    
def save_data(df, database_filename):
# Save out dataset
    from sqlalchemy import create_engine
    database_filepath = 'sqlite:////home/workspace/'+ str(database_filename)
    engine = create_engine(database_filepath)
    df.to_sql('df', engine, if_exists = 'replace', index=False)   


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