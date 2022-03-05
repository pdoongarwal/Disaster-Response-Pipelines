# import libraries
import re
import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """
    Load data from database.

    Parameters:
    database_filepath: Path of the database

    Returns:
    X (DataFrame) : Message features dataframe
    Y (DataFrame) : Target dataframe
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]

    return X, Y


def tokenize(text):
    """
    Clean, word tokenize and lemmatize the given text.

    Parameters:
    text(str): A message

    Returns:
    List of str: A list of the lemmatized words
    """
    clean_text = text.lower()
    clean_text = re.sub(r'[^a-zA-Z0-9]', ' ', clean_text)
    words = word_tokenize(clean_text)
    words = [word for word in words if word not in stopwords.words("english")]
    words = [WordNetLemmatizer().lemmatize(word) for word in words]
    return words


def build_model():
    """
    Build a model for classifing the disaster messages

    Returns:
    GridSearchCV: Classification model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        # 'tfidf__use_idf': (True, False),
        # 'clf__estimator__n_estimators': [5, 20]
        'clf__estimator__n_estimators': [5, 6]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the model and print the f1 score, precision and recall for each output category of the dataset.

    Parameters:
    model: Classification model
    X_test: test data
    Y_test: target data
    """
    y_pred = model.predict(X_test)

    for idx, col in enumerate(Y_test):
        print('Feature {}: {}'.format(idx + 1, col))
        print(classification_report(Y_test[col], y_pred[:, idx]))

    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    """
    Save model into a pickle file

    Parameters:
    model: Classification model
    model_filepath (str): Path of pickle file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
