# from pydantic import BaseModel
from abc import ABC, abstractmethod
import pandas as pd

# Vector Store
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.vectorstores.pgvector import PGVector

# import env
from config import Settings

class BASESearch(ABC):
    """基本的な情報検索のクラス"""

    def __init__(self):
        pass

    @abstractmethod
    def pre_process(self):
        pass

    @abstractmethod
    def search(self):
        pass

    # 検索結果を辞書型に変換
    def convert_to_df(self, result: list) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "document_id": doc.metadata["document_id"],
                "title": doc.metadata["title"],
                "score": score
            }
            for doc, score in result
        ]).set_index('document_id')

class BASEVectorSearch(BASESearch, ABC):
    """基本的なベクトル検索のクラス"""

    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.settings = Settings()

    @abstractmethod
    def pre_process(self):
        pass

    # 検索
    def search(self, query, table_name, top_k) -> pd.DataFrame:
        query = self.pre_process(query)
        res_qdrant = self.qdrant_search(query, table_name, top_k)
        df_qdrant = self.convert_to_df(res_qdrant)
        return df_qdrant

    # Qdrant検索
    def qdrant_search(self, query, table_name, top_k):
        # Qdrant
        qdrant_client = QdrantClient(url=self.settings.QDRANT_URL)
        qdrant = Qdrant(
            client=qdrant_client,
            collection_name=table_name,
            embeddings=self.embeddings,
        )
        return qdrant.similarity_search_with_score(query=query, k=top_k)

    # pgVector検索
    # def pgvector_search(self, embeddings, query, table_name, top_k):
    #     # PGVector
    #     CONNECTION_STRING = PGVector.connection_string_from_db_params(
    #         driver="psycopg2",
    #         host="localhost",
    #         port=self.settings.PORT_PGVECTOR,
    #         database=self.settings.POSTGRES_DB,
    #         user=self.settings.POSTGRES_USER,
    #         password=self.settings.POSTGRES_PASSWORD,
    #     )
    #     pg_vector = PGVector(
    #         connection_string=CONNECTION_STRING,
    #         embedding_function=embeddings,
    #         collection_name=table_name,
    #     )
    #     return pg_vector.similarity_search_with_score(query=query, k=top_k)



class BASEVectorSave(ABC):
    """基本的なベクトル保存のクラス"""

    def __init__(self, base_docs, embeddings):
        self.base_docs = base_docs
        self.embeddings = embeddings
        self.settings = Settings

    @abstractmethod
    def preprocess(self):
        pass

    # Qdrant
    def save_qdrant(self, embeddings, docs, table_name):
        # Qdrant
        Qdrant.from_documents(
            documents=docs,
            embedding=embeddings,
            url=self.settings.QDRANT_URL,
            prefer_grpc=True,
            collection_name=table_name,
        )

    # pgVector
    def save_pgvector(self, embeddings, docs, table_name):
        CONNECTION_STRING = PGVector.connection_string_from_db_params(
            driver="psycopg2",
            host="localhost",
            port=self.settings.PORT_PGVECTOR,
            database=self.settings.POSTGRES_DB,
            user=self.settings.POSTGRES_USER,
            password=self.settings.POSTGRES_PASSWORD,
        )

        PGVector.from_documents(
            embedding=embeddings,
            documents=docs,
            collection_name=table_name,
            connection_string=CONNECTION_STRING,
        )