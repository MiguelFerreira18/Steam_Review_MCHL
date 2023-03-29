import pandas as pd

dt = pd.read_csv('steamReviews.csv', nrows=100)
# --------------------Descricao das colunas--------------------
# app_id - ID do jogo
# app_name - Nome do jogo
# review_id - ID da review
# language - Linguagem da review
# review - Texto da review
# timestamp_created - Data de criacao da review
# timestamp_updated - Data de atualizacao da review
# recommended - Recomendado ou nao
# votes_helpful - Votos de utilidade
# votes_funny - Votos de humor
# weighted_vote_score - Media ponderada dos votos
# comment_count - Numero de comentarios
# steam_purchase - Comprado na Steam
# received_for_free - Recebido de graça
# written_during_early_access - Escrito durante o Early Access
# author.steamid - ID do autor
# author.num_games_owned - Numero de jogos do autor
# author.num_reviews - Numero de reviews do autor
# author.playtime_forever - Tempo de jogo total do autor
# author.playtime_last_two_weeks - Tempo de jogo nas ultimas duas semanas do autor
# author.last_played - Ultima vez que o autor jogou
# --------------------Descricao das colunas--------------------

print(len(dt.dropna()))
# print(dt.columns)
