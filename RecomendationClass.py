import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import SteamVariables as sv
from sklearn.utils import shuffle
import difflib
import random
import sys
from nltk.sentiment import SentimentIntensityAnalyzer
from importlib import reload
from surprise import KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise import SVD
from surprise import Reader
from surprise.model_selection import train_test_split, cross_validate


class Recommendation:
    new_dt = None
    def __init__(self):
        df = pd.read_csv(sv.CSV_PATH, nrows=8000000)##Chama o dataset
        df = df.dropna()##Elimina os nas para mais segurança
        languages = [ ##Linguagens a ser removidas
            "bulgarian",
            "croatian",
            "danish",
            "czech",
            "slovak",
            "slovenian",
            "slovak",
            "slovenian",
            "spanish",
            "estonian",
            "finnish",
            "french",
            "greek",
            "hungarian",
            "irish",
            "italian",
            "latvian",
            "lithuanian",
            "maltese",
            "dutch",
            "polish",
            "portuguese",
            "romanian",
            "swedish",
            "english",
            "brazilian",
        ]

        # Eliminar linhas com linguagem diferente das selecionadas
        df = df[self.df[sv.LANGUAGE].isin(languages)]
        df = shuffle(df, random_state=60)#Mete as linhas do data set aleatórias (Perserva o id inicial da linha)
        sample_dt = self.df.sample(n=50000) ##Corta o dataSet que levou shuffle para 50 k de valores
        df = pd.DataFrame(sample_dt) 

        df["review_score"] = df["review_score"].apply(lambda x: self.__converter_valor(x))## Cria uma nova coluna com os novos valores convertidos entre 0 a 5
        feature_columnsNew = [sv.AUTHOR_STEAMID, sv.APP_NAME, "review_score"] 
        new_dt = df[feature_columnsNew]


        reader = Reader(rating_scale=(0, 1))##cria um novo Reader para o SVD

        data = Dataset.load_from_df(new_dt, reader=reader)
        trainSet, testSet = train_test_split(data, test_size=0.2, random_state=60)
        algo = SVD()
        print(data)
        algo.fit(trainSet)
        predictions = algo.test(testSet)
        accuracy.rmse(predictions)
        accuracy.mae(predictions)
        cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)


    def __sentiment(self, review):
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(review)
        return score

    # Função para converter os valores
    def __converter_valor(self, review):
        valor = self.__sentiment(review)
        if valor < -0.5:
            return 1
        elif valor >= -0.5 and valor < 0:
            return 2
        elif valor >= 0 and valor < 0.4:
            return 3
        elif valor >= 0.4 and valor < 0.8:
            return 4
        elif valor >= 0.8:
            return 5
        else:
            return 0


    def __get_game_id(book_title, metadata):
        """
        Gets the game ID for a game title based on the closest match in the metadata dataframe.
        """

        existing_titles = list(metadata[sv.APP_NAME].values)
        closest_titles = difflib.get_close_matches(book_title, existing_titles)
        book_id = metadata[metadata[sv.APP_NAME] == closest_titles[0]][
            sv.APP_ID
        ].values[0]
        return book_id

    def __get_game_info(book_id, metadata):
        """
        Returns some basic information about a book given the book id and the metadata dataframe.
        """

        book_info = metadata[metadata[sv.APP_ID] == book_id][[sv.APP_ID, sv.APP_NAME]]

        game_info = book_info["app_name"].values[0]

        return game_info

    def __predict_review(self,user_id, book_title, model, metadata):
        """
        Predicts the review (on a scale of 1-5) that a user would assign to a specific book.
        """

        book_id = self.__get_game_id(book_title, metadata)
        review_prediction = model.predict(uid=user_id, iid=book_id)
        return review_prediction.est

    def generate_recommendation(self,user_id, model, metadata, thresh=0.15):
        jogos = []
        """
        Generates a book recommendation for a user based on a rating threshold. Only
        books with a predicted rating at or above the threshold will be recommended
        """

        lista_app = list(self.new_dt["app_name"].unique())
        random.shuffle(lista_app)

        i = 0
        for book_title in lista_app:
            rating = self.__predict_review(user_id, book_title, model, metadata)
            print(rating)

            if rating >= thresh:
                book_id = self.__get_game_id(book_title, metadata)
                jogos.append(self.__get_game_info(book_id, metadata))
                i += 1
            if i == 3:
                return jogos
        return jogos
