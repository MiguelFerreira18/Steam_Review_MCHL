from sklearn.model_selection import train_test_split
import pandas as pd
import SteamVariables as sv
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


class Regression:
    def linearRegression(self,X_train,Y_train,X_test,Y_test):

        pipe = Pipeline([('scaler', StandardScaler()), ('linearRegression', LinearRegression())])
        mod = GridSearchCV(estimator=pipe,
                            param_grid={'linearRegression__fit_intercept': [True, False]},
                            cv=5)
        mod.fit(X_train, Y_train)
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
        model = LinearRegression().fit(X_train, Y_train)
        print("Linear Regression value: " , model.score(X_test, Y_test))

    def ridgeRegression(self,X_train,Y_train,X_test,Y_test):
        pipe = Pipeline([('scaler', StandardScaler()), ('ridgeRegression', Ridge())])
        mod = GridSearchCV(estimator=pipe,
                            param_grid={'ridgeRegression__alpha': [0.01, 0.1, 1, 10, 100]},
                            cv=5)
        mod.fit(X_train, Y_train)
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
        model = Ridge(alpha=1.0).fit(X_train, Y_train)
        print("Ridge Regression value: " , model.score(X_test, Y_test))

    def ridgeRegressionCV(self,X_train,Y_train,X_test,Y_test):
        pipe = Pipeline([('scaler', StandardScaler()), ('ridgeRegressionCV', RidgeCV())])
        mod = GridSearchCV(estimator=pipe,
                            param_grid={'ridgeRegressionCV__alphas': np.arange(0.1,10.01)},
                            cv=5)
        mod.fit(X_train, Y_train)
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
        model = RidgeCV(alphas=np.arange(0.1,10,0.01))
        model.fit(X_train, Y_train)
        print("Ridge Regression Score: " , model.score(X_test, Y_test))
        print("Alpha:{:.2f}".format(model.alpha_))

    def lassoRegression(self,X_train,Y_train,X_test,Y_test):
        pipe = Pipeline([('scaler', StandardScaler()), ('lassoRegression', Lasso())])
        mod = GridSearchCV(estimator=pipe,
                            param_grid={'lassoRegression__alpha': [0.01, 0.1, 1, 10, 100]},
                            cv=5)
        mod.fit(X_train, Y_train)
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
        model = Lasso(alpha=0.10).fit(X_train, Y_train)
        print("Lasso Regression value: " , model.score(X_test, Y_test))

    def svmRegression(self, X_train, Y_train, X_test, Y_test):
        pipe = Pipeline([('scaler', StandardScaler()), ('svmRegression', svm.SVR())])
        mod = GridSearchCV(estimator=pipe,
                        param_grid={'svmRegression__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                    'svmRegression__C': [0.01, 0.1, 1, 10, 100],
                                    'svmRegression__gamma': [0.01, 0.1, 1, 10, 100]},
                        cv=5)
        mod.fit(X_train, Y_train)
        print("Best parameters are:", mod.best_params_)
        print("Best score is:", mod.best_score_)
        model = svm.SVR(kernel='linear', C=1, gamma=1)
        model.fit(X_train, Y_train)
        print("SVM Regression value:", model.score(X_test, Y_test))


    def knnRegression(self, X_train, Y_train, X_test, Y_test):
        pipe = Pipeline([('scaler', StandardScaler()), ('knnRegression', KNeighborsRegressor())])
        mod = GridSearchCV(estimator=pipe,
                        param_grid={'knnRegression__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                                    'knnRegression__weights': ['uniform', 'distance']},
                        cv=5)
        mod.fit(X_train, Y_train)
        print("Best parameters are:", mod.best_params_)
        print("Best score is:", mod.best_score_)
        model = KNeighborsRegressor(n_neighbors=5, weights='distance')
        model.fit(X_train, Y_train)
        print("KNN Regression value:", model.score(X_test, Y_test))


    def decisionTreeRegression(self, X_train, Y_train, X_test, Y_test):
        pipe = Pipeline([('scaler', StandardScaler()), ('decisionTreeRegression', RandomForestRegressor())])
        mod = GridSearchCV(estimator=pipe,
                        param_grid={'decisionTreeRegression__max_depth': [10]},
                        cv=5)
        mod.fit(X_train, Y_train)
        print("Best parameters are:", mod.best_params_)
        print("Best score is:", mod.best_score_)
        model = RandomForestRegressor(max_depth=10, random_state=0)
        model.fit(X_train, Y_train)
        print("R2 score:", r2_score(Y_test, model.predict(X_test)))
        print("MAE score:", mean_absolute_error(Y_test, model.predict(X_test)))

    def boomRegression