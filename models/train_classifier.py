# import libraries
import sys
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
# import relevant functions/modules for nlp
import re
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize, RegexpTokenizer
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
##############################################################################


def load_data_from_db(database_filepath):
    """
    Load Data from the Database Function

    Arguments:
        database_filepath: Path to SQLite destination
        database (e.g. disaster_response_db.db)
    Output:
        X : a pandas dataframe containing features
        Y : a pandas dataframe containing labels
        category_names : List of categories name
    """

    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(
                                  ".db", "") + "_table"
    df = pd.read_sql_table(table_name, engine)
    # Remove child alone as it has all zeros only
    df = df.drop(['child_alone'], axis=1)
    # Given value 2 in the related field are neglible so
    # it could be error. Replacing 2 with 1 to consider it a valid response.
    # Alternatively, we could have assumed it to be 0 also. In the absence
    # of information I have gone with majority class.
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)
    X = df['message']
    y = df.iloc[:, 4:]
    # saving the category_names for future
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Behaviour: Split text into words and return the root form of the words
    Args:
      text(str): text data.
    Return:
      clean_tokens(list of str): List of tokens extracted
      from the provided text
    """
    # Normalize text:Convert to lowercase and Remove punctuation
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Extract all the urls from the provided text
    detected_urls = re.findall(url_regex, text)
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Tokenize text:Split text into words using NLTK
    words = word_tokenize(text)    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in words if t not in stop]
    # lemmatize as shown in the lesson
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w) for w in words]
    return clean_tokens


def build_model():
    """Return Grid Search model with pipeline and Classifier"""

    # Create a instance for RandomFrorestClassifier()
    estimator_rf = MultiOutputClassifier(RandomForestClassifier())

    pipeline_rf = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', estimator_rf)
        ])
    # Using grid search
    # Create Grid search parameters for Random Forest Classifier
    parameters_rf = {'clf__estimator__max_depth': [10, 50, None],
                     'clf__estimator__min_samples_leaf': [1, 2, 5, 10],
                     'clf__estimator__n_estimators': [10, 20]}

    cv_rf = GridSearchCV(pipeline_rf, param_grid=parameters_rf)

    return cv_rf


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate Model function
    This function applies a ML pipeline to a test set and prints
    the model performance
    Args:
    model -- A valid scikit ML Pipeline
    X_test -- Test features
    y_test -- test labels
    category_names = required, list of category strings
    Returns:
    None
    """
    # Get results and add them to a dataframe.
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])


def save_model(model, model_filepath):
    """Save model as pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
     """Load the data, run the model and save model"""
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