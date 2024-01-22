import json
import plotly
# I'm using the latest libraries so joblib is on it's own package now
import joblib
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Tokenize, lemmatize, and clean the input text.
    
    Parameters:
    - text (str): The text to be processed.
    
    Returns:
    - clean_tokens (list of str): The processed text as a list of clean tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
query = f"""
        select category_name cat_name_by_message, count(*) as cnt_of_messages
        from messages m 
        inner join categories c on m.id = c.id 
        group by category_name 
        order by count(*) desc
        limit 15
        ;
    """

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)
categories_labels = list(pd.read_sql_table('categories_labels', engine).sort_values(by="id")['category_name'])
categories_distribution = pd.read_sql_query(query, engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Main page of web app.
    - Extracts data for plotting
    - Creates plotly visualizations
    - Renders the main page with plotly graphs
    """
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    messages_by_cat_counts = list(categories_distribution['cnt_of_messages'])
    messages_by_cat_names = list(categories_distribution['cat_name_by_message'])
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=messages_by_cat_names,
                    y=messages_by_cat_counts
                )
            ],

            'layout': {
                'title': 'TOP 15 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Web page that handles user query and displays model results.
    - Receives user input text
    - Uses model to predict classification
    - Renders the go.html page with the classification results
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(categories_labels, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Main function to run the Flask app.
    - Runs the app on host 0.0.0.0, port 3000, with debug mode on.
    Example usage:
    python run.py
    """
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()