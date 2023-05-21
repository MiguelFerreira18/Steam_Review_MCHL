import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler




class Classification:
    def logistic(self,X_train,Y_train,X_test,Y_test):
        # Logistic Regression
        ##Make a pipeline and a gridsearch to find the best parameters
        pipe = Pipeline([('scaler', StandardScaler()), ('logistic', LogisticRegression())])
        mod = GridSearchCV(estimator=pipe,
                            param_grid={'logistic__C': [0.01, 0.1, 1, 10, 100]},
                            cv=5)
        mod.fit(X_train, Y_train)
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
        logisticClassifier = LogisticRegression(C=mod.best_params_['logistic__C'])
        logisticClassifier.fit(X_train, Y_train)
        Y_pred = logisticClassifier.predict(X_test)
        print("----------Classification report-----------")
        print(classification_report(Y_test, Y_pred))
        cm = confusion_matrix(Y_test, Y_pred)
        ConfusionMatrixDisplay(cm).plot()
        plt.show()


    def naiveBayes(self,X_train,Y_train,X_test,Y_test):
        ##Naive Bayes
        ##Make a pipeline and a gridsearch to find the best parameters
        pipe = Pipeline([('scaler', StandardScaler()), ('naiveBayes', GaussianNB())])
        mod = GridSearchCV(estimator=pipe,
                            param_grid={'naiveBayes__var_smoothing': np.logspace(0,-9, num=100)},
                             cv=5)
        mod.fit(X_train, Y_train)
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
        naiveBayes = GaussianNB(var_smoothing=mod.best_params_['naiveBayes__var_smoothing'])
        naiveBayes.fit(X_train, Y_train)
        Y_pred = naiveBayes.predict(X_test)
        print("----------Classification report-----------")
        print(classification_report(Y_test, Y_pred))
        cm = confusion_matrix(Y_test, Y_pred)
        ConfusionMatrixDisplay(cm).plot()
        plt.show()


    def knn(self,X_train,Y_train,X_test,Y_test):
        ##KNN
        ##Make a pipeline and a gridsearch to find the best parameters
        pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
        mod = GridSearchCV(estimator = pipe,
                            param_grid={'knn__n_neighbors': [1, 3, 5,6 ,7,8,9,10,11,12],
                                        'knn__weights': ['uniform', 'distance'],
                                        'knn__leaf_size':[1,4,7,10],
                                        'knn__p':[1,2]},
                            cv=5)
        mod.fit(X_train, Y_train)
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
        knn = KNeighborsClassifier(n_neighbors=mod.best_params_['knn__n_neighbors'])
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        print("----------Classification report-----------")
        print(classification_report(Y_test, Y_pred))
        cm = confusion_matrix(Y_test, Y_pred)
        ConfusionMatrixDisplay(cm).plot()
        plt.show()


        

    def svm(self,X_train,Y_train,X_test,Y_test):
        ##SVm
        ##Make a pipeline and a gridsearch to find the best parameter
        pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])
        mod = GridSearchCV(estimator=pipe,
                            param_grid={'svm__C': [0.01, 0.1, 1, 10, 100]},
                             cv=5)
        mod.fit(X_train, Y_train)
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
        svm = SVC(C=mod.best_params_['svm__C'])
        svm.fit(X_train, Y_train)
        Y_pred = svm.predict(X_test)
        print("----------Classification report-----------")
        print(classification_report(Y_test, Y_pred))
        cm = confusion_matrix(Y_test, Y_pred)
        ConfusionMatrixDisplay(cm).plot()
        plt.show()


    def decisionTree(self,X_train,Y_train,X_test,Y_test):
        ##Decision Tree
        ##Make a pipeline and a gridsearch to find the best parameters
        pipe = Pipeline([('scaler', StandardScaler()), ('decisionTree', DecisionTreeClassifier())])
        mod = GridSearchCV(estimator=pipe,
                        param_grid={'decisionTree__max_depth': [1, 3, 5, 7, 9]},
                        cv=5)
        mod.fit(X_train, Y_train)
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
        decisionTree = DecisionTreeClassifier(max_depth=mod.best_params_['decisionTree__max_depth'])
        decisionTree.fit(X_train, Y_train)
        Y_pred = decisionTree.predict(X_test)
        print("----------Classification report-----------")
        print(classification_report(Y_test, Y_pred))
        cm = confusion_matrix(Y_test, Y_pred)
        ConfusionMatrixDisplay(cm).plot()
        plt.show()

