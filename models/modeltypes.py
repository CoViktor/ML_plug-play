from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from utils.script import rerunQ


def modeltype_catalogus(modeltype):
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
    elif modeltype == '8':
        modeltype = Ridge()
        param_grid = {
                'ridge__alpha': [0.1, 1.0, 10.0],
                'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
    elif modeltype == '9':
        modeltype = Lasso()
        param_grid = {
                'lasso__alpha': [0.001, 0.01, 0.1, 1.0],
                'lasso__max_iter': [500, 1000, 1500]
        }
    elif modeltype == '10':
        modeltype = ElasticNet()
        param_grid = {
                'elasticnet__alpha': [0.001, 0.01, 0.1, 1.0],
                'elasticnet__l1_ratio': [0.2, 0.5, 0.8],
                'elasticnet__max_iter': [1000, 2000]
        }
    elif modeltype == '11':
        modeltype = SVR()
        param_grid = {
                'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'svr__C': [0.1, 1, 10],
                'svr__epsilon': [0.01, 0.1, 0.2]
        }
    elif modeltype == '12':
        modeltype = DecisionTreeRegressor()
        param_grid = {
                'decisiontreeregressor__max_depth': [None, 5, 10, 20],
                'decisiontreeregressor__min_samples_split': [2, 5, 10],
                'decisiontreeregressor__min_samples_leaf': [1, 2, 4]
        }
    elif modeltype == '13':
        modeltype = RandomForestRegressor()
        param_grid = {
                'randomforestregressor__n_estimators': [100, 200, 300],
                'randomforestregressor__max_depth': [None, 10, 20],
                'randomforestregressor__min_samples_split': [2, 5, 10]
     }  
    else:
        print('Model not found')
        rerunQ()
    return modeltype, param_grid
