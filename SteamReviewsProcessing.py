import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import SteamVariables as sv

dt = pd.read_csv(sv.CSV_PATH, nrows=50000) # nrows=2000000
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

# Eliminar colunas desnecessarias
new_dt.drop(["Unnamed: 0"], axis=1, inplace=True)
new_dt.drop([sv.WRITTEN_DURING_EARLY_ACCESS], axis=1, inplace=True)

# Resetar index
new_dt.reset_index(drop=True, inplace=True)

# Eliminar linhas com valores nulos
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

# Print da correlacao e covariancia entre as colunas author.num_games_owned e author.num_reviews
print("Covariancia entre votes helpful com votes funny: \n" , new_dt[sv.VOTES_HELPFUL].cov(new_dt[sv.VOTES_FUNNY]))
print("Correlacao entre votes helpful com votes funny: \n " , new_dt[sv.VOTES_HELPFUL].corr(new_dt[sv.VOTES_FUNNY]), "\n")

# Print da correlacao e covariancia entre as colunas author.num_games_owned e author.num_reviews
print("Covariancia entre numero de jogos com numero de reviews: \n" , new_dt[sv.AUTHOR_NUM_GAMES_OWNED].cov(new_dt[sv.AUTHOR_NUM_REVIEWS]))
print("Correlacao entre numero de jogos com numero de reviews: \n " , new_dt[sv.AUTHOR_NUM_GAMES_OWNED].corr(new_dt[sv.AUTHOR_NUM_REVIEWS]), "\n")

#Print da correlacao e covariancia entre as colunas recomended e author.playtime_at_review
print("Covariancia entre recomended com author.playtime_at_review: \n" , new_dt[sv.RECOMMENDED].cov(new_dt[sv.AUTHOR_PLAYTIME_AT_REVIEW]))
print("Correlacao entre recomended com author.playtime_at_review: \n " , new_dt[sv.RECOMMENDED].corr(new_dt[sv.AUTHOR_PLAYTIME_AT_REVIEW]))


# Print do heatmap entre as colunas
sns.heatmap(new_dt.corr(), annot=True)
plt.figure()

#! Scatter plot com Votes funny e Votes helpful language
#! total_Time Recommended or not
#! HISTOGRAMA COM O TEMPO DE JOGO ASSIM COMO O VIOLIN PLOT
"""
Transformacao {
    normalizaçao --Analisar
    padronizaçao --Analisar
    transformaçao linear --Analisar
}   
"""
#Nomralização
myScalerMinMaxScaler = MinMaxScaler()


#new_dt['author.playtime_forever'] = myScalerMinMaxScaler.fit_transform(new_dt[['author.playtime_forever']])
norm = myScalerMinMaxScaler.fit_transform(new_dt[[sv.AUTHOR_PLAYTIME_FOREVER]])
# Plot the histogram of the normalized feature
plt.hist(norm, bins=20)
plt.title("author.playtime_forever - gpt")
plt.xlabel('Author Playtime Forever (Normalized)')
plt.ylabel('Frequency')
plt.figure()

norm = myScalerMinMaxScaler.fit_transform(new_dt[[sv.AUTHOR_PLAYTIME_FOREVER]])
print(norm)
plt.plot(norm)
plt.figure()


#Standardização
numberColumns = [sv.WEIGHTED_VOTE_SCORE,sv.AUTHOR_PLAYTIME_FOREVER,sv.AUTHOR_PLAYTIME_LAST_TWO_WEEKS]
scale= StandardScaler()
scaled_data = scale.fit_transform(new_dt[numberColumns])
plt.hist(scaled_data,100)
plt.figure()


##Transformaão linear simples
linearTransformation = 4+2*new_dt[sv.AUTHOR_PLAYTIME_FOREVER]
print(linearTransformation)
plt.plot(linearTransformation)
plt.figure()


plt.plot(new_dt[sv.AUTHOR_PLAYTIME_FOREVER])
plt.show()

