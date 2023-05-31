from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import SteamVariables as sv
from sklearn.preprocessing import StandardScaler




dt = pd.read_csv(sv.CSV_PATH, nrows=20000) # nrows=2000000
pd.set_option('display.max_columns', None)


# Eliminar linhas com linguagem diferente das selecionadas

# Eliminar colunas desnecessarias
dt.drop([sv.STEAM_PURCHASE], axis=1, inplace=True)
dt.drop([sv.WRITTEN_DURING_EARLY_ACCESS], axis=1, inplace=True)
dt.drop([sv.LANGUAGE], axis=1, inplace=True)
dt.drop([sv.APP_NAME],axis=1,inplace=True)
dt.drop([sv.REVIEW],axis=1,inplace=True)
dt.drop([sv.TIMESTAMP_CREATED],axis=1,inplace=True)
dt.drop([sv.VOTES_FUNNY],axis=1,inplace=True)
dt.drop([sv.VOTES_HELPFUL],axis=1,inplace=True)
dt.drop([sv.AUTHOR_STEAMID],axis=1,inplace=True)
dt.drop([sv.AUTHOR_NUM_GAMES_OWNED],axis=1,inplace=True)
dt.drop([sv.AUTHOR_NUM_REVIEWS],axis=1,inplace=True)
dt.drop([sv.AUTHOR_PLAYTIME_FOREVER],axis=1,inplace=True)
dt.drop([sv.AUTHOR_PLAYTIME_LAST_TWO_WEEKS],axis=1,inplace=True)
dt.drop([sv.AUTHOR_LAST_PLAYED],axis=1,inplace=True)
dt.drop([sv.TIMESTAMP_UPDATED],axis=1,inplace=True)

##Make the recomended 1 if it is True or 0 it is False
dt[sv.RECOMMENDED] = dt[sv.RECOMMENDED].apply(lambda x: 1 if x==True else 0)
##Make received_for_free 1 if it is True or 0 it is False
dt[sv.RECEIVED_FOR_FREE] = dt[sv.RECEIVED_FOR_FREE].apply(lambda x: 1 if x==True else 0)

##use StandardScaler author.playtime_at_review
scaler = StandardScaler()


# separate features and class labels
X_features = dt
y_labels = dt[sv.RECEIVED_FOR_FREE]
model = AgglomerativeClustering(linkage="ward", n_clusters=3)
model.fit(X_features)
predicted_labels = model.labels_
linkage_matrix = linkage(X_features, 'ward')
plot = plt.figure(figsize=(14, 7))
dendrogram(
linkage_matrix,
truncate_mode='lastp',
p=20,
color_threshold=0,
)
plt.title('Hierarchical Clustering Dendrogram (linkage=ward)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.show()