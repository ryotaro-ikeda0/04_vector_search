import pandas as pd
from tabulate import tabulate

from base import BASESearch

class CrossEncoder(BASESearch):
    def __init__(self, search_keyword, search_vector):
        self.search_keyword = search_keyword
        self.search_vector = search_vector
        self.keyword_weight = 0.3
        self.vector_weight = 0.7

    def pre_process(self, query):
        return query

    def search(self, query, table_name, top_k=10) -> pd.DataFrame:
        df_keyword_search = self.search_keyword.search(query, top_k=top_k*5)
        df_vector_search = self.search_vector.search(query, table_name, top_k=top_k*5)
        re_ranking_scored = self.cross_encoder_re_score(df_keyword_search, df_vector_search)

        return re_ranking_scored[:top_k]

    def cross_encoder_re_score(self, df_keyword_search, df_vector_search) -> pd.DataFrame:

        df_marge = pd.merge(df_keyword_search, df_vector_search, on=['document_id', 'title'], how='outer').fillna(0)
        df_marge['score'] = df_marge['score_x'] * self.keyword_weight + df_marge['score_y'] * self.vector_weight
        df_re_ranking = df_marge.sort_values('score', ascending=False)

        return df_re_ranking