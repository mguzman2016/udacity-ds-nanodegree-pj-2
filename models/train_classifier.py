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
    normalized_text = text.lower()
    tokenized_text = word_tokenize(normalized_text)
    words = [w for w in tokenized_text if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', RandomForestClassifier())
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    classification_values = classification_report(Y_test, y_pred, target_names=category_names)
    print(classification_values)

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file_to_write:
        pickle.dump(model, file_to_write)


def main():
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