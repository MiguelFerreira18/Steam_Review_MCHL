from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import SteamVariables as sv



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
dt[sv.AUTHOR_PLAYTIME_AT_REVIEW] = scaler.fit_transform(dt[sv.AUTHOR_PLAYTIME_AT_REVIEW].values.reshape(-1,1))







dt.dropna(inplace=True)
# Resetar index
dt.reset_index(drop=True, inplace=True)


print(dt)

wcss = [] # sum of the squared distance
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=0)
    kmeans.fit(dt)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


rf=KMeans(n_clusters=2)
clf=rf.fit(dt)
centroids=clf.cluster_centers_
score=silhouette_score(dt,clf.labels_)
print(centroids)
print(score)

X=dt
y_kmeans=clf.labels_
y_kmeans = clf.fit_predict(X)
# Visualising the clusters
cols = dt.columns
X=X.to_numpy()
print(type(X))
plt.scatter(X[y_kmeans == 0, 2],
    X[y_kmeans == 0, 3],
    s=100, c='purple',
    label='1')
plt.scatter(X[y_kmeans == 1, 2],
    X[y_kmeans == 1, 3],
    s=100, c='orange',
    label='2')

# Plotting the centroids of the clusters
plt.scatter(centroids[:, 0],
centroids[:, 1],
s=400, c='black',
marker="x",
label='Centroids')
#plt.ylim([293000, 291000])
plt.legend()
plt.show()