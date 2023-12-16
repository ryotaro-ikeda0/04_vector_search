# import modules
import traceback
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from tabulate import tabulate

# import utilities
from BM25 import BestMatching
from VectorSearch import (
    BasicSearch,
    HyDESearch,
    HyDEWithCustomSearch,
    SynonymSearch,
)
from CrossEncoderReRanking import CrossEncoder
from Evaluation import Evaluator
from utilities import (
    initialize_model,
    load_docs,
    pd_multi_merge,
)
from dataset import (
    WandDataset,
    HomeDepotDataset,
)

# main
def experiment1():
    # 評価
    top_k = 10
    WANDS = WandDataset()
    llm, embeddings = initialize_model()
    evaluator = Evaluator()

    bestMatching = BestMatching(WANDS.get_docs())
    basicSearch = BasicSearch(llm, embeddings)
    hyDESearch = HyDESearch(llm, embeddings)
    synonymSearch = SynonymSearch(llm, embeddings)
    crossEncoder = CrossEncoder(bestMatching, basicSearch)

    df_eval_bm25 = search_eval(dataset=WANDS, search_model=bestMatching, evaluator=evaluator, top_k=top_k)
    df_eval_basic = search_eval(dataset=WANDS, search_model=basicSearch, evaluator=evaluator, top_k=top_k, table_name='original')
    df_eval_hyDE = search_eval(dataset=WANDS, search_model=hyDESearch, evaluator=evaluator, top_k=top_k, table_name='original')
    df_eval_synonym = search_eval(dataset=WANDS, search_model=synonymSearch, evaluator=evaluator, top_k=top_k, table_name='original')
    df_eval_crossEncoder = search_eval(dataset=WANDS, search_model=crossEncoder, evaluator=evaluator, top_k=top_k, table_name='original')

    df_eval = pd_multi_merge(
        [df_eval_bm25.T, df_eval_basic.T, df_eval_hyDE.T, df_eval_synonym.T, df_eval_crossEncoder.T],
        columns=['bm25', 'basic', 'hyDE', 'synonym', 'crossEncoder'],
        left_index=True, right_index=True, how='outer'
    )
    print(tabulate(df_eval, headers='keys', tablefmt='psql'))
    df_eval.to_csv('../data/out/eval_experiment1.csv', index=False)

def experiment2():

    # 評価
    top_k = 10
    WANDS = WandDataset()
    llm, embeddings = initialize_model()
    evaluator = Evaluator()

    basicSearch = BasicSearch(llm, embeddings)

    df_eval_original = search_eval(dataset=WANDS, search_model=basicSearch, evaluator=evaluator, top_k=top_k, table_name='original')
    df_eval_split = search_eval(dataset=WANDS, search_model=basicSearch, evaluator=evaluator, top_k=top_k, table_name='split')

    df_eval = pd_multi_merge(
        [df_eval_original.T, df_eval_split.T],
        columns=['original', 'split'],
        left_index=True, right_index=True, how='outer'
    )
    print(tabulate(df_eval, headers='keys', tablefmt='psql'))
    df_eval.to_csv('../data/out/eval_experiment2.csv', index=False)


def experiment3():

    # 評価
    top_k = 10
    WANDS = WandDataset(n=1000)
    llm, embeddings = initialize_model()
    evaluator = Evaluator()

    basicSearch = BasicSearch(llm, embeddings)

    df_eval_original = search_eval(dataset=WANDS, search_model=basicSearch, evaluator=evaluator, top_k=top_k, table_name='original_1000')
    df_eval_synopsis = search_eval(dataset=WANDS, search_model=basicSearch, evaluator=evaluator, top_k=top_k, table_name='synopsis')
    df_eval_synopsis_ja = search_eval(dataset=WANDS, search_model=basicSearch, evaluator=evaluator, top_k=top_k, table_name='synopsis_ja')
    df_eval_summary = search_eval(dataset=WANDS, search_model=basicSearch, evaluator=evaluator, top_k=top_k, table_name='summary')
    df_eval_summary_ja = search_eval(dataset=WANDS, search_model=basicSearch, evaluator=evaluator, top_k=top_k, table_name='summary_ja')
    df_eval_split = search_eval(dataset=WANDS, search_model=basicSearch, evaluator=evaluator, top_k=top_k, table_name='split_1000')

    df_eval = pd_multi_merge(
        [df_eval_original.T, df_eval_synopsis.T, df_eval_synopsis_ja.T, df_eval_summary.T, df_eval_summary_ja.T, df_eval_split.T],
        columns=['original_1000', 'synopsis', 'synopsis_ja', 'summary', 'summary_ja', 'split_1000'],
        left_index=True, right_index=True, how='outer'
    )
    print(tabulate(df_eval, headers='keys', tablefmt='psql'))
    df_eval.to_csv('../data/out/eval_experiment3.csv', index=False)


def search_eval(dataset, search_model, evaluator, top_k, **kwargs):

    start_time = time.time()

    # vector search
    docs = dataset.get_docs()
    queries = dataset.get_query()

    y_trues = np.empty((0, top_k))
    y_scores = np.empty((0, top_k))
    for index, (query_id, query) in enumerate(tqdm(queries)):

        # keyword search
        try:
            if "split" in kwargs and kwargs['table_name'] == 'split':
                df = search_model.search(query, top_k=top_k*10, **kwargs)

                # document_idが重複している場合があるので、document_idで重複を削除
                df = df.reset_index().drop_duplicates(subset=['document_id']).head(top_k).set_index('document_id')
            else:
                df = search_model.search(query, top_k=top_k, **kwargs)
        except Exception as e:
            print(f"!例外発生! query_id; {query_id}, query: {query}")
            traceback.print_exc()
            continue

        y_true = dataset.get_score(query_id=query_id, product_id_list=df.index.to_list())
        y_score = df['score'].tolist()

        y_trues = np.append(y_trues, [y_true], axis=0)
        y_scores = np.append(y_scores, [y_score], axis=0)

    df_eval = evaluator.eval(y_trues, y_scores)

    elapsed_time = time.time() - start_time
    print(f"elapsed_time: {elapsed_time}")

    return df_eval

if __name__ == "__main__":

    experiment1()
    # experiment2()
    # experiment3()