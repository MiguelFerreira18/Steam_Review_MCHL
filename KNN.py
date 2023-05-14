from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import SteamVariables as sv

dt = pd.read_csv(sv.CSV_PATH, nrows=20000) # nrows=2000000
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


dtTraining = dt.drop(sv.RECOMMENDED, axis=1)
dtTest = dt[sv.RECOMMENDED]
X_train, X_test, Y_train, Y_test = train_test_split(dtTraining, dtTest, test_size=0.3, random_state=5)


#Create the Classifier
clf = KNeighborsClassifier(n_neighbors=5)
#Train the model using the training sets
clf.fit(X_train, Y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("----------Classification report-----------")
print(classification_report(Y_test, y_pred))

##AQUI PODIA VIR O OPEN AI, DAR CLASSIFICAÇÃO