import os
import numpy as np
import pandas as pd
import warnings
from sklearn import metrics

class Annotation:
    def __init__(self, parent=None):
        self.__parent = parent
        self.__annotation = pd.DataFrame()

    def __call__(self, by=None, name=None):
        return self.add(by, name)
    
    def add(self, by=None, name=None):
        """Returns heuristic annotation dataframe of avatar
        :param by: {str|callble} csv_path or callable function. If function, function should return pd.Series.
            If None, returns current annotation
        :param annot:

        :returns: boolean array
        """

        if isinstance(by, str):
            csv_path = by
            assert os.path.splitext(csv_path)[1].lower() == '.csv', 'Wrong file type error. provide .csv file'
            if csv_path:
                data = pd.read_csv(csv_path, header=None)[0].values
                index = self.__parent.index
                N, K = len(data), len(index)
                if N != K:
                    data = np.zeros_like(index)[:N]
                    warnings.warn(f'Warning your input csv file length({N}) miss match with avatar index({K})')
                name = name if name else csv_path
                self.__annotation[name] = pd.Series(data=data, index=index).astype(bool)
            return self.__annotation

        elif callable(by):
            func = by
            if func:
                s = func(self.__parent)
                assert isinstance(s, pd.Series), 'Given annotation function should return pd.Series'
                name = name if name else str(func)
                self.__annotation[name] = s.astype(bool)
        elif by != None:
            raise Exception('Wrong argument type provided, should be function or csv file')
        return self.__annotation

    def intersection(self, columns=[]):
        """Returns row wise intersection of annotation df
        """
        if columns:
            self.__annotation[columns].all(axis=1)
        return self.__annotation.all(axis=1)
    
    def union(self, columns=[]):
        """Returns row wise union of annotation df
        """
        if columns:
            self.__annotation[columns].any(axis=1)
        return self.__annotation.any(axis=1)

    def iou(self, columns=[]):
        """Returns intersection over union score (0~1)
        """
        i = self.intersection(columns).astype(int).sum()
        u = self.union(columns).astype(int).sum()
        return i/u

    def get_indices(self, name):
        """Returns annotation indices.
        """
        return self.__annotation.index[self.__annotation[name]]

    def metrics(self, true, pred, includes=["confusion_matrix", "accuracy_score", "recall_score", "precision_score", "f1_score", "jaccard_score"]):
        """Returns metrics of given pred, true data

        :param true: np.array or list of int(0 or 1) or boolean, should be equal length with `pred`
        :param pred: np.array or list of int(0 or 1) or boolean

        :returns: dict of metric
        """
        ret = {}
        for include in includes:
            metric_func = getattr(metrics, include)
            
            true = true.astype(int) if true.dtype == bool else true
            pred = pred.astype(int) if pred.dtype == bool else pred
            ret[include] = metric_func(true, pred)
        return ret

