import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import SteamVariables as sv

dt = pd.read_csv(sv.CSV_PATH, nrows=2000000) # nrows=2000000
pd.set_option('display.max_columns', None)
#Dominio: Reviews da Steam
#Tamanho: 2000000
# --------------------Descricao das colunas--------------------
#! app_id - ID do jogo, Discrete
#! app_name - Nome do jogo, Nominal
#! review_id - ID da review, Discrete
#! language - Linguagem da review, Nominal
#! review - Texto da review, Nominal
#! timestamp_created - Data de criacao da review, Discrete
#* timestamp_updated - Data de atualizacao da review, Discrete
#* recommended - Recomendado ou nao, Discrete
#* votes_helpful - Votos de utilidade, Discrete
#* votes_funny - Votos de humor, Discrete
#* weighted_vote_score - Ranking baseado no número de helpful votes, Continuous
#* comment_count - Numero de comentarios, Discrete
#* steam_purchase - Comprado na Steam, Discrete
#* received_for_free - Recebido de graça, Discrete
#? written_during_early_access - Escrito durante o Early Access, Discrete
#? author.steamid - ID do autor, Discrete
#? author.num_games_owned - Numero de jogos do autor, Discrete
#? author.num_reviews - Numero de reviews do autor, Discrete
#? author.playtime_forever - Tempo de jogo total do autor, Continuous
#? author.playtime_last_two_weeks - Tempo de jogo nas ultimas duas semanas do autor, Continuous
#? author.last_played - Ultima vez que o autor jogou, Discrete
# --------------------Descricao das colunas--------------------

languages = ["bulgarian", "croatian", "danish", "czech", "slovak", "slovenian", "slovak", "slovenian", 
"spanish", "estonian", "finnish", "french", "greek", "hungarian", "irish", "italian", 
"latvian", "lithuanian", "maltese", "dutch", "polish", "portuguese", "romanian", "swedish", "english", "brazilian"]

# Eliminar linhas com linguagem diferente das selecionadas
new_dt = dt[dt[sv.LANGUAGE].isin(languages)]
print(new_dt[sv.WEIGHTED_VOTE_SCORE].max())

# Eliminar colunas desnecessarias
new_dt.drop(["Unnamed: 0"], axis=1, inplace=True)
new_dt.drop([sv.STEAM_PURCHASE], axis=1, inplace=True)
new_dt.drop([sv.WRITTEN_DURING_EARLY_ACCESS], axis=1, inplace=True)

# Resetar index
new_dt.reset_index(drop=True, inplace=True)

# Eliminar linhas com valores nulos
new_dt[sv.AUTHOR_NUM_GAMES_OWNED].fillna(new_dt[sv.AUTHOR_NUM_GAMES_OWNED].mean(), inplace=True)
new_dt[sv.AUTHOR_NUM_REVIEWS].fillna(new_dt[sv.AUTHOR_NUM_REVIEWS].mean(), inplace=True)
new_dt[sv.AUTHOR_PLAYTIME_FOREVER].fillna(new_dt[sv.AUTHOR_PLAYTIME_FOREVER].mean(), inplace=True)
new_dt[sv.AUTHOR_PLAYTIME_LAST_TWO_WEEKS].fillna(new_dt[sv.AUTHOR_PLAYTIME_LAST_TWO_WEEKS].mean(), inplace=True)
new_dt[sv.AUTHOR_LAST_PLAYED].fillna(new_dt[sv.AUTHOR_LAST_PLAYED].mean(), inplace=True)

new_dt.dropna(inplace=True)

# Print da descricao da coluna author.num_games_owned do dataset
print(new_dt)

print("<----------------------<>---------------------->")
print("\n Timestamp Created: ", new_dt[sv.TIMESTAMP_CREATED].describe() , "\n")
print("<----------------------<>---------------------->")
print("\n Timestamp Updated: ", new_dt[sv.TIMESTAMP_UPDATED].describe(), "\n")
print("<----------------------<>---------------------->")
print("\n Votes Helpful: ", new_dt[sv.VOTES_HELPFUL].describe() , "\n")
print("<----------------------<>---------------------->")
print("\n Votes Funny: ", new_dt[sv.VOTES_FUNNY].describe() , "\n")
print("<----------------------<>---------------------->")
print("\n Weighted Vote Score: ", new_dt[sv.WEIGHTED_VOTE_SCORE].describe() , "\n")
print("<----------------------<>---------------------->")
print("\n Comment Count: ", new_dt[sv.COMMENT_COUNT].describe() , "\n")
print("<----------------------<>---------------------->")
print("\n Author Num Games Owned: ", new_dt[sv.AUTHOR_NUM_GAMES_OWNED].describe() , "\n")
print("<----------------------<>---------------------->")
print("\n Author Num Reviews: ", new_dt[sv.AUTHOR_NUM_REVIEWS].describe() , "\n")
print("<----------------------<>---------------------->")
print("\n Author Playtime Forever: ", new_dt[sv.AUTHOR_PLAYTIME_FOREVER].describe() , "\n")
print("<----------------------<>---------------------->")
print("\n Author Playtime Last Two Weeks: ", new_dt[sv.AUTHOR_PLAYTIME_LAST_TWO_WEEKS].describe() , "\n")
print("<----------------------<>---------------------->")
print("\n Author Last Played: ", new_dt[sv.AUTHOR_LAST_PLAYED].describe())
print("<----------------------<>---------------------->")

# Print da correlacao e covariancia entre as colunas author.num_games_owned e author.num_reviews
print("Covariancia entre tempo total de jogo com tempo total jogado quando a review foi publicada: \n" , new_dt[sv.AUTHOR_PLAYTIME_FOREVER].cov(new_dt[sv.AUTHOR_PLAYTIME_AT_REVIEW]))
print("Correlacao entre tempo total de jogo com tempo total jogado quando a review foi publicada: \n " , new_dt[sv.AUTHOR_PLAYTIME_FOREVER].corr(new_dt[sv.AUTHOR_PLAYTIME_AT_REVIEW]), "\n")

# Print da correlacao e covariancia entre as colunas author.votes_funny e author.votes_helpful
print("Covariancia entre votes helpful com votes funny: \n" , new_dt[sv.VOTES_HELPFUL].cov(new_dt[sv.VOTES_FUNNY]))
print("Correlacao entre votes helpful com votes funny: \n " , new_dt[sv.VOTES_HELPFUL].corr(new_dt[sv.VOTES_FUNNY]), "\n")

# Print da correlacao e covariancia entre as colunas author.num_games_owned e author.num_reviews
print("Covariancia entre numero de jogos com numero de reviews: \n" , new_dt[sv.AUTHOR_NUM_GAMES_OWNED].cov(new_dt[sv.AUTHOR_NUM_REVIEWS]))
print("Correlacao entre numero de jogos com numero de reviews: \n " , new_dt[sv.AUTHOR_NUM_GAMES_OWNED].corr(new_dt[sv.AUTHOR_NUM_REVIEWS]), "\n")

#Print da correlacao e covariancia entre numero de jogos e tempo de jogo total
print("Covariancia entre numero de jogos com tempo de jogo total: \n" , new_dt[sv.AUTHOR_NUM_GAMES_OWNED].cov(new_dt[sv.AUTHOR_PLAYTIME_FOREVER]))
print("Correlacao entre numero de jogos com tempo de jogo total: \n " , new_dt[sv.AUTHOR_NUM_GAMES_OWNED].corr(new_dt[sv.AUTHOR_PLAYTIME_FOREVER]))


# Print do heatmap entre as colunas
df_heatmap = new_dt.drop("author.steamid", axis=1) # Eliminar coluna author.steamid,pois esta nao faz sentido usar
df_heatmap = df_heatmap.drop("app_id", axis=1) # Eliminar coluna review_id,pois esta nao faz sentido usar
sns.heatmap(df_heatmap.corr(), annot=True)
plt.show()

#! Boxplot
new_dt["review_char_count"] = new_dt["review"].apply(len)

# Create a subset of the data that excludes reviews with character counts greater than 1000
steam_data_subset = new_dt[new_dt["review_char_count"] <= 100]

ax = sns.boxplot(x="app_name", y="review_char_count", data=steam_data_subset)

# Set the x-axis tick labels to be spaced farther apart
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)



#! Violinplot da coluna weighted_vote_score
sns.violinplot(x=sv.WEIGHTED_VOTE_SCORE, data=new_dt)
plt.show()

#! Scatterplot entre as colunas recomended e author.playtime_at_review
sns.scatterplot(x=sv.RECOMMENDED, y=sv.AUTHOR_PLAYTIME_AT_REVIEW, data=new_dt)
plt.xticks([0, 1], ['Not Recommended', 'Recommended'])
plt.show()


#! Histograma da coluna recommended,que mostra a quantidade de pessoas que recomendaram ou nao pelo menos um jogo
sns.histplot(data=new_dt, x=sv.RECOMMENDED, bins=2, discrete=True)
plt.title("Histogram of Review Ratings")
plt.xlabel("Recommended")
plt.xticks([0, 1], ['Not Recommended', 'Recommended'])
plt.ylabel("Count")
plt.show()

#! Histograma da coluna review_length, que mostra a quantidade de caracteres de cada review
new_dt['review_length'] = new_dt['review'].apply(lambda x: len(str(x)))
plt.figure(figsize=(10,6))
sns.histplot(data=new_dt, x='review_length', kde=True)
plt.title("Distribution of Review Length")
plt.xlabel("Review Length")
plt.ylabel("Count")
plt.show()

#! Scatterplot entre as colunas author.num_games_owned e author.playtime_forever
sns.scatterplot(data=new_dt, x=sv.AUTHOR_PLAYTIME_FOREVER, y=sv.AUTHOR_PLAYTIME_AT_REVIEW)
plt.show()

#? Apenas tirar print aos que fazem sentido?
sns.pairplot(new_dt[[sv.AUTHOR_NUM_GAMES_OWNED,sv.AUTHOR_PLAYTIME_FOREVER]])
plt.show()


#! Scatterplot entre as colunas author.votes_helpful e author.votes_funny
sns.scatterplot(data=new_dt, x=sv.VOTES_HELPFUL, y=sv.VOTES_FUNNY)
plt.show()


#! Scatterplot entre as colunas author.num_games_owned e author.num_reviews
sns.scatterplot(data=new_dt, x=sv.AUTHOR_NUM_GAMES_OWNED, y=sv.AUTHOR_NUM_REVIEWS)
plt.show()



"""
Transformacao {
    normalizaçao --Analisar
    padronizaçao --Analisar
    transformaçao linear --Analisar
}   
"""

##TEMP ALTERAR DEPOIS
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

print("TRANSFORMATION PART OF THE WORK\n")

#!Linear Transformation
new_dt['total_votes'] = (new_dt[sv.VOTES_HELPFUL] * 5 )/ (new_dt[sv.VOTES_HELPFUL].max())
print(new_dt['total_votes'].mean())


#!Normalization of total votes
myScalerMinMaxScaler = MinMaxScaler()
normalizedColumn = myScalerMinMaxScaler.fit_transform(new_dt[['total_votes']].values)
plt.plot(normalizedColumn)
plt.title("Total votes - Normalized")
plt.xlabel('Samples')
plt.ylabel('Normalized Total Votes')
plt.show()

#!Standardization of total votes
myStandardScaler = StandardScaler()
standardizedColumn = myStandardScaler.fit_transform(new_dt[['total_votes']])
plt.hist(standardizedColumn)
plt.title("Total votes - Standardized")
plt.xlabel('Tota Votes')
plt.ylabel('Frequency')
plt.show()




