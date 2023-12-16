import numpy as np
import pandas as pd

from sklearn.metrics import ndcg_score
from sklearn.metrics import label_ranking_average_precision_score

class Evaluator():

    # 評価指標を計算する
    def eval(self, y_true: np.array, y_score:np.array):
        evaluation = {
            f"NDCG@{5}": [self.NDCG(y_true, y_score, 5)],
            f"NDCG@{10}": [self.NDCG(y_true, y_score, 10)],
            f"MRR@{5}": [self.MRR(y_true, 5)],
            f"MRR@{10}": [self.MRR(y_true, 10)],
            f"MAP{5}": [self.MAP(y_true, y_score, 5)],
            f"MAP{10}": [self.MAP(y_true, y_score, 10)],
        }

        return pd.DataFrame(data=evaluation)

    # NDCGスコア
    def NDCG(self, y_true: np.array, y_score: np.array, top_k: int) -> int:
        return ndcg_score(y_true, y_score, k=top_k)

        # MRRスコア
    def MRR(self, y_true: np.array, top_k: int) -> int:
        rs = np.where(y_true > 0, 1,0)[:top_k]
        rs = (np.asarray(r).nonzero()[0] for r in rs)
        return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

    # MAPスコア
    def MAP(self, y_true: np.array, y_score: np.array, top_k: int) -> int:
        y_true = np.where(y_true > 0, 1,0)[:top_k]
        y_score = y_score[:top_k]
        return label_ranking_average_precision_score(y_true, y_score)
