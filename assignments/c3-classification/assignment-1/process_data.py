import pandas as pd
from prefect import task, flow
from sklearn import impute
from sklearn.model_selection import train_test_split
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer

# ============================================================================== #
#                               Create tasks                                     #
# ============================================================================== #

@task
def get_data(path):
    """
    Reads in the data from the given path.
    """
    return pd.read_csv(path)

def get_train_test_split(data: pd.DataFrame):
    """
    Splits the data into training and testing sets.
    """
    train = data.drop(['RainTomorrow'], axis=1)
    test = data['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=0)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
    
def handle_numerical_data(data: dict):
    X_train = data['X_train']
    X_test = data['X_test']
    numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
    imputer = MeanMedianImputer(imputation_method='mean', variables=numerical)
    imputer.fit(X_train)
    imputer.transform(X_train)
    imputer.transform(X_test)
    assert(X_train[numerical].isnull().sum().sum() == 0)
    assert(X_test[numerical].isnull().sum().sum() == 0)
    data['X_train'] = X_train
    data['X_test'] = X_test
    return data
    
def handle_categorical_data(data: dict):
    X_train = data['X_train']
    X_test = data['X_test']
    categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
    imputer = CategoricalImputer(variables=categorical, imputation_method='frequent')
    imputer.fit(X_train)
    imputer.transform(X_train)
    imputer.transform(X_test)
    assert(X_train[categorical].isnull().sum().sum() == 0)
    assert(X_test[categorical].isnull().sum().sum() == 0)
    data['X_train'] = X_train
    data['X_test'] = X_test
    return data

# ============================================================================== #
#                               Create a flow                                    #
# ============================================================================== #

@flow
def process_data():
    pass

if __name__ == "__main__":
    process_data()