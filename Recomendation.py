import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import SteamVariables as sv

scaler = StandardScaler()
df = pd.read_csv(sv.CSV_PATH, nrows=9000000)
df = df.dropna()

#print(df.groupby('app_name').first().reset_index())

feature_columns = [sv.RECOMMENDED, sv.WEIGHTED_VOTE_SCORE, sv.COMMENT_COUNT,
                   sv.STEAM_PURCHASE, sv.RECEIVED_FOR_FREE, sv.WRITTEN_DURING_EARLY_ACCESS]
df[feature_columns] = df[feature_columns].astype(np.float32)

standardized_df = scaler.fit_transform(df[feature_columns])

pca = PCA(n_components=2)
standardized_df_pca = pca.fit_transform(standardized_df)
print(pca.explained_variance_ratio_)

nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
nn.fit(standardized_df)
distances, indices = nn.kneighbors(standardized_df)

# Create a dictionary that maps usernames to indices
indices_dict = {username: i for i, username in enumerate(df.index)}

# Define a function that generates recommendations for a given username
def generate_recommendation(username):
    try:
        # Find the index of the username in the standardized dataframe
        idx = indices_dict[username]

        # Find the indices of the nearest neighbors
        neighbor_indices = indices[idx]

        # Print the recommendations
        print('Recommendations for {}:'.format(username))
        for i, neighbor_index in enumerate(neighbor_indices):
            if i == 0:
                continue
            neighbor_username = df.index[neighbor_index]
            print('  {}: {}'.format(i, neighbor_username))
    except KeyError:
        print('No recommendations for {}'.format(username))

# Simulate for hollow knight and print the neighbors
generate_recommendation('Portal 2')
