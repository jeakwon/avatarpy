import os
import pandas as pd

class Annotation:
    def __init__(self, parent=None):
        self.__parent = parent
        self.__annotation = pd.DataFrame()

    def __call__(self, by=None, name=None):
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
                index = self.__parent.index[:len(data)]
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
        if columns:
            self.__annotation[columns].all(axis=1)
        return self.__annotation.all(axis=1)
    
    def union(self, columns=[]):
        if columns:
            self.__annotation[columns].any(axis=1)
        return self.__annotation.any(axis=1)

    def iou(self, columns=[]):
        i = self.intersection(columns).astype(int).sum()
        u = self.union(columns).astype(int).sum()
        return i/u

# class HeuristicAnnotation:
#     def __init__(self, parent=None):
#         self.__parent = parent
#         self.__annotation = pd.DataFrame()

#     def __call__(self, func=None, annot=None):
#         """Returns heuristic annotation dataframe of avatar
#         :param func: callable function, should return pd.Series
#         :param annot:

#         :returns: boolean array
#         """
        
#         assert callable(func), 'func must be callable function'
#         assert isinstance(s, pd.Series), 'Given annotation function should return pd.Series'
#         if func:
#             s = func(self.__parent)
#             name = annot if annot else str(func)
#             self.__annotation[name] = pd.Series(data=data, index=index)
#         return self.__annotation

# class HumanAnnotation:
#     def __init__(self, parent=None):
#         self.__parent = parent
#         self.__annotation = pd.DataFrame()

#     def __call__(self, csv_path=None, annot=None):
#         """Returns human annotation dataframe of avatar
#         :param csv_path:
#         :param annot:

#         :returns: pd.DataFrame
#         """
