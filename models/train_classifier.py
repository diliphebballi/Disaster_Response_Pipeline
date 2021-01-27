##Import libraries required
import sys
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine

def load_data(database_filepath):
    '''
    Read data from created SQL table
    Load messages into dataframe X
    Load the categories into dataframe Y
    Load the category headers into category_names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Disaster_Response', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names=Y.columns.values
    return X,Y,category_names

def tokenize(text):
    '''
    INPUT: takes the text as input
    Tokenize the text
    Remove stopwords
    OUTPUT: Returns clean tokens
    '''
    # tokenize text and initiate lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # Remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    ML pipeline for classifier
    Defined parameters to be passed to grid search
    OUTPUT: Returns the model
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
            
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    param = {
     'clf__estimator__n_estimators': [5, 10, 20]
    }
    
    model = GridSearchCV(pipeline, param_grid = param)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT: Model, 20% of the data and categories
    
    OUTPUT: Prints the precision, recall and f1-score for the 36 categories
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred, target_names=category_names))

def save_model(model, model_filepath):
    '''
    INPUT: This function takes the model and filepath as input
    This funtion saves the pickle file for the model in the filepath
    '''
    pickle.dump(model,open(model_filepath,'wb'))


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
