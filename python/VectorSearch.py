# basic
import pandas as pd

# import langchain
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain import PromptTemplate

# import from other files
from config import Settings
from base import BASEVectorSearch

settings = Settings()

# ベクトル検索クラス
class BasicSearch(BASEVectorSearch):
    # 初期化
    def __init__(self, llm, embeddings):
        super().__init__(llm, embeddings)

    # Queryの前処理
    def pre_process(self, query):
        return query

# HyDEベクトル検索クラス
class HyDESearch(BASEVectorSearch):

    # 初期化
    def __init__(self, llm, embeddings):
        embeddings = self._initialize_HyDE_embeddings(llm, embeddings)
        super().__init__(llm, embeddings)

    # Queryの前処理
    def pre_process(self, query):
        return query

    # HyDEEmbeddingsの初期化
    def _initialize_HyDE_embeddings(self, llm, embeddings):
        prompt_template = """What is this?
        question: {question}
        answer: """

        prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # initialize the hypothetical document embedder
        embeddings = HypotheticalDocumentEmbedder(llm_chain=llm_chain, base_embeddings=embeddings)

        return embeddings


# HyDEベクトル検索クラス（カスタム）
class HyDEWithCustomSearch(BASEVectorSearch):

    # 初期化
    def __init__(self, llm, embeddings):
        super().__init__(llm, embeddings)

    # Queryの前処理
    def pre_process(self, query):
        return self._get_HyDE_answer(self.llm, query)

    # 単にqueryをHypothesis的な回答に変換する
    def _get_HyDE_answer(self, llm, query):
        prompt_template = """Please write a passage to answer the question
        Question: {question}
        Passage:"""

        prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        return llm_chain.run(question=query)


# 類語検索クラス
class SynonymSearch(BASEVectorSearch):
    # 初期化
    def __init__(self, llm, embeddings):
        super().__init__(llm, embeddings)

    # Queryの前処理
    def pre_process(self, query):
        synonym_query = self._create_synonym(self.llm, query)
        print(synonym_query)
        return synonym_query

    # 類語を生成する
    def _create_synonym(self, llm, query):
        prompt_template = """Generate sentences similar to the following sentence:
        Target sentence: {query}
        Similar sentences:"""

        prompt = PromptTemplate(input_variables=["query"], template=prompt_template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        return query + llm_chain.run(query=query)
