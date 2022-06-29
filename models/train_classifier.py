import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
import re
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt','wordnet'])

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression


def load_data(database_filepath):
    database_filepath2 = 'sqlite:////home/workspace/'+ str(database_filepath)
    engine = create_engine(database_filepath2)
    df = pd.read_sql_table('df', engine)
    X = df.message.values
    Y = df.drop(['message','id','original','genre'], axis=1)
    category_names = Y.columns.values.tolist()
    return X, Y, category_names



def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model(X_train, Y_train):
    database_filepath, model_filepath = sys.argv[1:]
    pipelineRF = Pipeline([ 
        ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()), ('MOC',MultiOutputClassifier(RandomForestClassifier()))
    
    ])
    pipelineLR = Pipeline([ 
        ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()), ('MOC',MultiOutputClassifier(LogisticRegression()))
    
    ])

    
#     model = pipeline.fit(X_train, Y_train)
    parametersLR = {
    'MOC__estimator__fit_intercept': [True, False]
    }
    cv = GridSearchCV(pipelineLR, param_grid = parametersLR)
    model = cv.fit(X_train, Y_train)
#     model = pipelineLR
    print("Best Parameters", cv.best_params_)
    return model



def evaluate_model(model, X_test, Y_test, category_names, ):
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns = Y_test.columns)
    for column in Y_test.columns:
#     display_results(y_test[column].reset_index(drop=True), y_pred_df[column])
        print("Values for column",column,"\r\n",classification_report(Y_test[column].reset_index(drop=True), y_pred_df[column]))
#     print("Now program will run again but optimize some parameters of the estimator")
#     parametersRF = {
#     'MOC__estimator__min_samples_leaf': [1,2],
#     'MOC__estimator__n_estimators':[10, 20]
#     }
#     parametersLR = {
#     'MOC__estimator__fit_intercept': [True, False]
#     }
#     pipelineLR = Pipeline([ 
#         ('vect', CountVectorizer(tokenizer = tokenize)),
#     ('tfidf', TfidfTransformer()), 
#         ('MOC',MultiOutputClassifier(LogisticRegression()))
    
#     ])

#     cv = GridSearchCV(pipelineLR, param_grid = parametersLR)
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#     cv.fit(X_train, Y_train)
#     best_paramsLR = cv.best_params_
#     print("Best Parameters", cv.best_params_)
#     print("Rerunning with optimized parameters")
#     pipeline2 = Pipeline([
#         ('vect', CountVectorizer(tokenizer = tokenize)),
#         ('tfidf', TfidfTransformer()),
#         ('MOC',MultiOutputClassifier(RandomForestClassifier(n_estimators=20, min_samples_leaf = 1)))
#     ])
#     best_paramsRF = {'MOC__estimator__min_samples_leaf':best_paramsRF['MOC__estimator__min_samples_leaf'],
#               'MOC__estimator__n_estimators':best_paramsRF['MOC__estimator__n_estimators']}
#     best_paramsLR = {'MOC__estimator__fit_intercept':best_paramsLR['MOC__estimator__fit_intercept']}
#     best_paramsLR = {'MOC__estimator__fit_intercept':False}
#     pipelineLR.set_params(**best_paramsLR)
#     pipelineLR.fit(X_train, Y_train)
    
#     y_pred2 = pipelineLR.predict(X_test)
#     y_pred_df2 = pd.DataFrame(y_pred2, columns = Y_test.columns)

    # y_pred_df['related'].head()
    # y_test['related'].head()
#     y_pred2 = model.predict(X_test)
#     y_pred_df2 = pd.DataFrame(y_pred2, columns = Y_test.columns)
#     for column in category_names:
#     #     display_results(y_test[column].reset_index(drop=True), y_pred_df[column])
#         print("Values for column",column,"\r\n",classification_report(Y_test[column].reset_index(drop=True), y_pred_df2[column]))
        
    
def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("model_filepath", model_filepath)
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
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