import pandas as pd
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


import SteamVariables as sv

dt = pd.read_csv(sv.CSV_PATH, nrows=2000) # nrows=2000000
pd.set_option('display.max_columns', None)

# Eliminar colunas desnecessarias
dt.drop(["Unnamed: 0"], axis=1, inplace=True)
dt.drop([sv.STEAM_PURCHASE], axis=1, inplace=True)
dt.drop([sv.WRITTEN_DURING_EARLY_ACCESS], axis=1, inplace=True)
dt.drop([sv.APP_NAME], axis=1, inplace=True)
dt.drop([sv.REVIEW], axis=1, inplace=True)
dt.drop([sv.TIMESTAMP_UPDATED], axis=1, inplace=True)
dt.drop([sv.TIMESTAMP_CREATED], axis=1, inplace=True)
dt.drop([sv.LANGUAGE], axis=1, inplace=True)

# Resetar index
dt.reset_index(drop=True, inplace=True)

# Eliminar linhas com valores nulos
dt[sv.AUTHOR_NUM_GAMES_OWNED].fillna(dt[sv.AUTHOR_NUM_GAMES_OWNED].mean(), inplace=True)
dt[sv.AUTHOR_NUM_REVIEWS].fillna(dt[sv.AUTHOR_NUM_REVIEWS].mean(), inplace=True)
dt[sv.AUTHOR_PLAYTIME_FOREVER].fillna(dt[sv.AUTHOR_PLAYTIME_FOREVER].mean(), inplace=True)
dt[sv.AUTHOR_PLAYTIME_LAST_TWO_WEEKS].fillna(dt[sv.AUTHOR_PLAYTIME_LAST_TWO_WEEKS].mean(), inplace=True)
dt[sv.AUTHOR_LAST_PLAYED].fillna(dt[sv.AUTHOR_LAST_PLAYED].mean(), inplace=True)

# Transformar valores booleanos em inteiros
dt[sv.RECOMMENDED] = dt[sv.RECOMMENDED].map({True: 1, False: 0})

# ------------------------- Naive Bayes Classification -------------------------

# Separar dados em treino e teste
# dtTraining = dt.drop(sv.RECOMMENDED, axis=1)
# dtTest = dt[sv.RECOMMENDED]
# X_train, X_test, Y_train, Y_test = train_test_split(dtTraining, dtTest, test_size=0.3, random_state=0)
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, Y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d"
# % (X_test.shape[0], (Y_test != y_pred).sum()))
# print(classification_report(Y_test, y_pred))
# cm = confusion_matrix(Y_test, y_pred)
# ConfusionMatrixDisplay(cm).plot()
# plt.show()

# ------------------------- Naive Bayes Classification -------------------------

# ------------------------- Ridge and Lasso Regression -------------------------

dtTraining = dt.drop(sv.AUTHOR_NUM_GAMES_OWNED, axis=1)
dtTest = dt[sv.AUTHOR_NUM_GAMES_OWNED]
X_train, X_test, Y_train, Y_test = train_test_split(dtTraining, dtTest, test_size=0.3, random_state=0)

# rcv=RidgeCV(alphas=np.arange(0.1,100,0.01))
# rcv.fit(dtTraining, dtTest)
# print("Alpha:{:.2f}".format(rcv.alpha_))
# print(rcv.coef_)
# print(rcv.score(dtTraining,dtTest))

lf = LinearRegression()

lf.fit(X_train, Y_train) 

scores = cross_val_score(lf, X_train, Y_train, cv=5)

print(lf.score(X_test,Y_test))
print(scores.mean())

dfValue = np.array([[100, 100, 0, 5, 10, 0.5, 1, 0, 10, 10, 10, 10, 10, 10]])
lfn = lf.predict(dfValue)

clf=linear_model.Lasso(alpha=0.1)
clf.fit(dtTraining, dtTest)
print("R2={:.2f}".format(r2_score(dtTest,clf.predict(dtTraining))))
print("MAE={:.2f}".format(mean_absolute_error(dtTest,clf.predict(dtTraining))))

plt.plot(Y_test, "o")
plt.plot(clf.predict(X_test), "b")
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()

