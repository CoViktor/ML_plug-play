from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_validate, GridSearchCV
from utils.script import rerunQ
from models.modeltypes import modeltype_catalogus

def prep_pipeline(X_train):
    # Identifying categoricals and numericals
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_train.select_dtypes(exclude=['object', 'category']).columns

    # Numerical preprocessing
    numerical_pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler()
    )

    # Categorical preprocessing
    categorical_pipeline = make_pipeline(
        OneHotEncoder(handle_unknown='ignore')
    )

    # ColumnTransformer 
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_pipeline, categorical_cols),
            ('num', numerical_pipeline, numerical_cols) 
        ],
        remainder='passthrough'
    )
    return preprocessor

def model_pipeline(model, preprocessor):
    pipeline = make_pipeline(preprocessor, model)
    print('Model trained')
    return pipeline

def predict_evaluate(pipe, X_train, y_train, X_test, y_test):
    print('Evaluating model...')
    y_pred = pipe.predict(X_test)
    print("train_score: ", round(pipe.score(X_train, y_train), 3))
    print("test_score: ", round(pipe.score(X_test, y_test), 2))
    print("ROC_AUC: ", round(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]), 3))
    report = classification_report(y_test, y_pred)
    print(report)
    cv_results = cross_validate(pipe, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True)
    print("Mean Test Accuracy:", round(cv_results['test_score'].mean(), 3))
    print("Mean Train Accuracy:", round(cv_results['train_score'].mean(), 3))
    print("Mean Fit Time:", round(cv_results['fit_time'].mean(), 3))
    print("Mean Score Time:", round(cv_results['score_time'].mean(), 3))

def gridsearch(pipe, param_grid, X_train, y_train):
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best parameters:", best_params)
    print("Best cross-validated score:", best_score)
    return best_params

def save_model(model, model_name):
    pass

def run_model_script(df, y, columns, modeltype):
    if y in columns:
        X = df[columns].drop(y, axis=1)
    else:
        X = df[columns]
    y = df[y]

    print('Splitting and training')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = prep_pipeline(X_train)
    modeltype, param_grid = modeltype_catalogus(modeltype)
    model = model_pipeline(modeltype, preprocessor)
    model.fit(X_train, y_train)
    predict_evaluate(model, X_train, y_train, X_test, y_test)
    GridQ = input('Would you like to tune hyperparameters (Gridsearch)? (y/n)')
    if GridQ == 'y':
        best_params = gridsearch(model, param_grid, X_train, y_train)
    if input('Would you like to save the model? (y/n)') == 'y':
        save_model(model, modeltype)
    rerunQ()


    # save metrics in csv with modeltype timestamped, see bear
    # save optimal parameters in csv with modeltype timestamped
    # -> or instead: ask to rerun model with best params?
    # if visualsQ == 'y': add visuals of eg ROC curve, confusion matrix, feature importance
    # make script that compares the models visually, over time, like bear