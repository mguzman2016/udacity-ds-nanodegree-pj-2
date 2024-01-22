import sys
import pandas as pd
import sqlite3

def connect_to_database(filepath):
    """
    Establish a connection to the SQLite database specified by the filepath.
    
    Parameters:
    - filepath (str): The path to the SQLite database file.
    
    Returns:
    - connection: A SQLite connection object if the connection is established successfully.
    
    Raises:
    - sqlite3.Error: If the connection to the database fails, the program exits with error code 1.
    """
    try:
        connection = sqlite3.connect(filepath)
        print(f'Connected to database {filepath}')
        return connection
    except sqlite3.Error as e:
        print(f"Database connection failed: {e}")
        sys.exit(1)

def close_database_connection(connection):
    """
    Close the connection to the SQLite database.
    
    Parameters:
    - connection: The connection object to the SQLite database.
    
    Raises:
    - sqlite3.Error: If closing the database connection fails.
    """
    try:
        connection.close()
        print('Database connection closed.')
    except sqlite3.Error as e:
        print(f"Failed to close database connection: {e}")

def load_dataset(file_path, id_col="id"):
    """
    Load a dataset from a CSV file, dropping duplicate entries based on a specified column.
    
    Parameters:
    - file_path (str): The path to the CSV file.
    - id_col (str, optional): The column to check for duplicates. Defaults to 'id'.
    
    Returns:
    - DataFrame: A pandas DataFrame with the loaded data.
    """
    return pd.read_csv(file_path).drop_duplicates(subset=id_col, keep='first')

def save_to_table(df, table, connection):
    """
    Save a DataFrame to a SQLite table, replacing the table if it already exists.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame to save.
    - table (str): The name of the target SQLite table.
    - connection: The connection object to the SQLite database.
    
    Raises:
    - Exception: If an error occurs during the insertion into the database.
    """
    try:
        df.to_sql(table, connection, if_exists='replace', index=False)
    except Exception as e:
        print(f"Failed with error while inserting to database: {e}")   

def process_categories(categories):
    """
    Process the categories DataFrame by splitting the categories and
    exploding them into separate rows.
    
    Parameters:
    - categories (DataFrame): The DataFrame containing the 'categories' column and the 'id' of the message.
    
    Returns:
    - DataFrame: A processed DataFrame with 'category_name' and 'id' columns.
    """
    categories['categories'] = categories['categories'].str.split(';')
    exploded_categories = categories.explode('categories', ignore_index=False)
    
    split_categories = exploded_categories['categories'].str.split('-', expand=True)
    
    exploded_categories[['category_name', 'is_in_category']] = split_categories
    exploded_categories.drop('categories', axis=1, inplace=True)
    exploded_categories['is_in_category'] = exploded_categories['is_in_category'].astype(int)
    
    # To avoid warnings
    categories_to_save = exploded_categories[exploded_categories['is_in_category'] == 1].copy()

    categories_to_save.drop('is_in_category',axis=1,inplace=True)
    return categories_to_save

def process_messages(messages):
    """
    Filter out empty or whitespace-only messages from the DataFrame.
    
    Parameters:
    - messages (DataFrame): The DataFrame containing the 'message' column.
    
    Returns:
    - DataFrame: A DataFrame containing only non-empty messages.
    """
    return messages[messages['message'].str.strip() != '']

def main():
    """
    Main function to run the data processing script. It expects three command-line arguments:
    - The file path to the messages CSV.
    - The file path to the categories CSV.
    - The file path to the SQLite database.
    
    The script loads, cleans, and saves the messages and categories data to the database.

    Example usage:
    python process_data.py {messages file location} {categories file location} {output location to save the DB file}
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Provided paths\n MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        print(f'Opening connection to database {database_filepath}')
        
        db_connection = connect_to_database(database_filepath)
        print('Connected to database correctly')

        print('Loading messages')
        messages = load_dataset(messages_filepath)
        print(f'Total records: {len(messages)}')
        print(f'Saving messages to database: {database_filepath}.messages')
        messages_to_save = process_messages(messages)
        save_to_table(messages_to_save,'messages',db_connection)

        print('Loading categories')
        categories = load_dataset(categories_filepath)
        print(f'Total records: {len(categories)}')

        print('Cleaning categories data')
        categories_to_save = process_categories(categories)
        print(f'Total categories to save: {len(categories_to_save)}')

        print(f'Saving categories to database: {database_filepath}.categories')
        save_to_table(categories_to_save,'categories',db_connection)

        close_database_connection(db_connection)
        
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