from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_validate, GridSearchCV
from utils.script import rerunQ

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
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validated score:", grid_search.best_score_)

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

    if modeltype == '1':
        modeltype = GradientBoostingClassifier()
        param_grid = {
    'gradientboostingclassifier__n_estimators': [100, 200, 300],
    'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.2],
    'gradientboostingclassifier__max_depth': [3, 5, 7]
    }
    elif modeltype == '2':
        modeltype = AdaBoostClassifier()
        param_grid = {
    'adaboostclassifier__n_estimators': [50, 100, 200],
    'adaboostclassifier__learning_rate': [0.01, 0.1, 1.0]
    }
    elif modeltype == '3':
        modeltype = RandomForestClassifier()
        param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [None, 10, 20],
    'randomforestclassifier__min_samples_split': [2, 5, 10]
    }
    elif modeltype == '4':
        modeltype = KNeighborsClassifier()
        param_grid = {
    'kneighborsclassifier__n_neighbors': [3, 5, 10],
    'kneighborsclassifier__weights': ['uniform', 'distance']
    }
    elif modeltype == '5':
        modeltype = DecisionTreeClassifier()
        param_grid = {
    'decisiontreeclassifier__max_depth': [None, 10, 20, 30],
    'decisiontreeclassifier__min_samples_split': [2, 10, 20],
    'decisiontreeclassifier__min_samples_leaf': [1, 5, 10]
    }
    elif modeltype == '6':
        modeltype = LogisticRegression()
        param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10],
    'logisticregression__penalty': ['l2', 'l1', 'l2', 'elasticnet', 'none']
    }
    elif modeltype == '7':
        modeltype = LinearRegression()
        param_grid = {
    'linearregression__fit_intercept': [True, False],
    'linearregression__normalize': [True, False]
    }
    else:
        print('Model not found')
        rerunQ()
    model = model_pipeline(modeltype, preprocessor)
    model.fit(X_train, y_train)
    predict_evaluate(model, X_train, y_train, X_test, y_test):
    GridQ = input('Would you like to tune hyperparameters (Gridsearch)? (y/n)')
    if GridQ == 'y':
        gridsearch(model, param_grid, X_train, y_train)
    if input('Would you like to save the model? (y/n)') == 'y':
        save_model(model, modeltype)
    rerunQ()


    # save metrics in csv with modeltype timestamped, see bear
    # save optimal parameters in csv with modeltype timestamped
    # if visualsQ == 'y': add visuals of eg ROC curve, confusion matrix, feature importance
    # make script that compares the models visually, over time, like bear