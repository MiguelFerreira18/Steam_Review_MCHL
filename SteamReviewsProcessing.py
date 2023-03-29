import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

dt = pd.read_csv('steamReviews.csv', nrows=100)
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

print(len(dt.dropna()))
# print(dt.columns)



