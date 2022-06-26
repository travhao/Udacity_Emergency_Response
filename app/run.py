import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index()

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    positive_counts = df.drop(['id','message','original','genre'], axis=1)
    pct_positive = positive_counts.sum()/positive_counts.count()
    pct_positive_names = list(pct_positive.index)

    positive_counts2 = df.drop(['id','message','original'], axis=1).groupby('genre')
    pct_positive2 = positive_counts2.sum()/positive_counts2.count()
    pct_positive_names2 = list(pct_positive2.index)
    
    bc_counts = df.drop(['id','message','original'], axis=1).groupby('genre').sum()
    bc_names = list(bc_counts.columns)
    bc_direct = list(bc_counts.loc['direct'])
    bc_news = list(bc_counts.loc['news'])
    bc_social = list(bc_counts.loc['social'])
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
       
         {
            'data': [
                Bar(
                    x=pct_positive_names,
                    y=pct_positive
                )
            ],

            'layout': {
                'title': 'Percent Messages Meeting Criteria',
                'yaxis': {
                    'title': "Percent Matching"
                },
                'xaxis': {
                    'title': "Variable"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=bc_names,
                    y=bc_direct,
                    name='direct'
                ),
                Bar(
                    x=bc_names,
                    y=bc_news,
                    name='news'
                ),
                Bar(
                    x=bc_names,
                    y=bc_social,
                    name='social'
                )
            ],

            'layout': {
                'title': 'Counts Matching by Genre',
                'yaxis': {
                    'title': "Count Matching"
                },
                'xaxis': {
                    'title': "Variable"
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
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()