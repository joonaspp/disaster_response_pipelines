import sys
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    #Import Python libraries
    #Load dataset from database with read_sql_table
    #Define feature and target variables X and Y
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM df_projeto3", engine)
    X = df['message']
    Y = df.iloc[:, 4:40]
    category_names = list(Y.columns.values)
    
    return X, Y, category_names


def tokenize(text):
    # Write a tokenization function to process your text data
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return stemmed


def build_model():
    #Build a machine learning pipeline
    #This machine pipeline should take in the message column as input and output classification results on the other 36 categories in the dataset. You may find the MultiOutputClassifier helpful for predicting multiple target variables.
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    parameters = {
    'vect__max_df':[0.70,1.0],
    'clf__estimator__n_estimators': [30, 50]
    }
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    #Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's classification_report on each.
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    return 


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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