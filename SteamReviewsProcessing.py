import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

dt = pd.read_csv('steamReviews.csv', nrows=200000) # nrows=2000000
# --------------------Descricao das colunas--------------------
# app_id - ID do jogo, Discrete
# app_name - Nome do jogo, Nominal
# review_id - ID da review, Discrete
# language - Linguagem da review, Nominal
# review - Texto da review, Nominal
# timestamp_created - Data de criacao da review, Discrete
# timestamp_updated - Data de atualizacao da review, Discrete
# recommended - Recomendado ou nao, Discrete
# votes_helpful - Votos de utilidade, Discrete
# votes_funny - Votos de humor, Discrete
# weighted_vote_score - Media ponderada dos votos, Continuous
# comment_count - Numero de comentarios, Discrete
# steam_purchase - Comprado na Steam, Discrete
# received_for_free - Recebido de graça, Discrete
# written_during_early_access - Escrito durante o Early Access, Discrete
# author.steamid - ID do autor, Discrete
# author.num_games_owned - Numero de jogos do autor, Discrete
# author.num_reviews - Numero de reviews do autor, Discrete
# author.playtime_forever - Tempo de jogo total do autor, Continuous
# author.playtime_last_two_weeks - Tempo de jogo nas ultimas duas semanas do autor, Continuous
# author.last_played - Ultima vez que o autor jogou, Discrete
# --------------------Descricao das colunas--------------------

languages = ["bulgarian", "croatian", "danish", "czech", "slovak", "slovenian", "slovak", "slovenian", 
"spanish", "estonian", "finnish", "french", "greek", "hungarian", "irish", "italian", 
"latvian", "lithuanian", "maltese", "dutch", "polish", "portuguese", "romanian", "swedish", "english", "brazilian"]

# Eliminar linhas com linguagem diferente das selecionadas
new_dt = dt[dt['language'].isin(languages)]

# Eliminar colunas desnecessarias
new_dt.drop(["Unnamed: 0"], axis=1, inplace=True)
new_dt.drop(["written_during_early_access"], axis=1, inplace=True)

# Resetar index
new_dt.reset_index(drop=True, inplace=True)

# Eliminar linhas com valores nulos
new_dt.dropna(inplace=True)

# Print da descricao da coluna author.num_games_owned do dataset
# print(new_dt.describe())
print(new_dt["author.num_games_owned"].describe())

# Print da correlacao entre as colunas author.num_games_owned e author.num_reviews
print("COVARIANCIA: " , new_dt["author.playtime_forever"].cov(new_dt["author.playtime_at_review"]))
print("CORRELACAO: " , new_dt["author.playtime_forever"].corr(new_dt["author.playtime_at_review"]))

# Print do heatmap entre as colunas
sns.heatmap(new_dt.corr(), annot=True)
plt.figure()


"""
Transformacao {
    normalizaçao --Analisar
    padronizaçao --Analisar
    transformaçao linear --Analisar
}   
"""
#Nomralização
myScalerMinMaxScaler = MinMaxScaler()
norm = myScalerMinMaxScaler.fit_transform(new_dt[['author.playtime_at_review']].values)
print(norm)
plt.plot(norm)
plt.figure()


#Standardização
numberColumns = ["weighted_vote_score","author.playtime_forever","author.playtime_last_two_weeks"]
scale= StandardScaler()
scaled_data = scale.fit_transform(new_dt[numberColumns])
plt.hist(scaled_data,100)
plt.figure()


##Transformaão linear simples
linearTransformation = 4+2*new_dt["author.playtime_forever"]
print(linearTransformation)
plt.plot(linearTransformation)
plt.figure()


plt.plot(new_dt["author.playtime_forever"])
plt.show()

