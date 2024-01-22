import sys
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load data from a SQLite database and process it into a format suitable for model training.
    
    Parameters:
    - database_filepath (str): The file path to the SQLite database.
    
    Returns:
    - X (Series): The input variable (messages).
    - Y (DataFrame): The target variables (categories in one-hot encoded format).
    - category_names (Index): The names of the categories.
    """
    conn = sqlite3.connect(database_filepath)
    query = f"""
        SELECT 
            m.id,m.message ,
            c.category_name
        FROM messages m
        INNER JOIN categories c ON m.id = c.id
        ORDER BY m.id
        ;
    """
    df = pd.read_sql_query(query, conn)

    dummies = pd.get_dummies(df['category_name'])

    final_df = df.drop('category_name', axis=1).join(dummies).groupby('id').agg({
        'message': 'first',
        **{col: 'sum' for col in dummies.columns}
    }).reset_index()

    final_df.drop('id',axis=1,inplace=True)

    # We are also saving the label columns to the database as an intermediate step to be able
    # to print them on the web application

    categories_df = pd.DataFrame(dummies.columns, columns=['category_name'])
    categories_df['id'] = categories_df.index + 1

    categories_df.to_sql('categories_labels', conn, if_exists='replace', index=False)
    
    conn.close()

    return final_df['message'], final_df[dummies.columns], dummies.columns


def tokenize(text):
    """
    Tokenize, lemmatize, and remove stop words from the input text.
    
    Parameters:
    - text (str): The text to be processed.
    
    Returns:
    - lemmed (list of str): The processed text as a list of tokens.
    """
    normalized_text = text.lower()
    tokenized_text = word_tokenize(normalized_text)
    words = [w for w in tokenized_text if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed


def build_model():
    """
    Build a machine learning pipeline that processes text messages and predicts classifications for all categories.
    
    Returns:
    - cv (GridSearchCV): A GridSearchCV object with the pipeline and parameter grid.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', RandomForestClassifier())
    ])

    # parameters = {
    #     'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    #     'clf__n_estimators': [50, 100, 200],
    #     'clf__min_samples_split': [2, 3, 4]
    # }

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1),),
        'clf__n_estimators': [50],
        'clf__min_samples_split': [2]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model's performance on the test set.
    
    Parameters:
    - model: The trained model.
    - X_test (Series): The input variables for the test set.
    - Y_test (DataFrame): The true labels for the test set.
    - category_names (Index): The names of the categories.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    classification_values = classification_report(Y_test, y_pred, target_names=category_names)
    print(classification_values)

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    
    Parameters:
    - model: The trained model.
    - model_filepath (str): The file path to save the model to.
    """
    with open(model_filepath, 'wb') as file_to_write:
        pickle.dump(model, file_to_write)


def main():
    """
    Main function to run the machine learning pipeline. It loads data, splits it into a training set and a test set, 
    builds a model, trains the model, evaluates the model, and then saves the model to a pickle file.

    Example usage: 
    python train_classifier.py {DB location} {Output location to save the classifier}
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()