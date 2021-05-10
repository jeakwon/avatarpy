import pandas as pd

class Describe:
    def __init__(self, parent=None):
        self.__parent = parent

    def __repr__(self):
        return f'Extract object of {self.__parent}'

    def __call__(self, indices=None, include=['corr', 'stat'], assign_ID=True, assign_tags=True):
        df = pd.concat([getattr(self, x)(indices=indices, assign_ID=assign_ID, assign_tags=assign_tags) for x in include]).reset_index(drop=True)
        return df

    def corr(self, indices=None, features=['velocity', 'acceleration', 'angle', 'angle_velocity', 'angle_acceleration', 'stretch_index'], 
        assign_ID=True, assign_tags=True):
        """Returns pearson correlation of given features
        """
        corrs = []
        for feature in features:
            data = self.__parent[feature]
            if indices is not None:
                data = data.loc[indices]
            corrs.extend([
                self.__parent.corr(data)        .reset_index().rename(
                    columns={'index':'target', 0:'value'}).assign(feature=feature, category='correlation', type='pearson'),
                self.__parent.xcorr_max(data)   .reset_index().rename(
                    columns={'index':'target', 0:'value'}).assign(feature=feature, category='correlation', type='xcorr_max'),
                self.__parent.xcorr_lag(data)   .reset_index().rename(
                    columns={'index':'target', 0:'value'}).assign(feature=feature, category='correlation', type='xcorr_lag'),
            ])
        df = pd.concat(corrs).reset_index(drop=True)
        if assign_ID:
            df = df.assign(ID=self.__parent.ID)
        if assign_tags:
            df = df.assign(**self.__parent.tags)
        return df
    
    def stat(self, indices=None, features=['velocity', 'acceleration', 'angle', 'angle_velocity', 'angle_acceleration', 'vector_length'], 
        assign_ID=True, assign_tags=True):
        """Returns basic stat (mean, std, median, skew) of of given features
        """
        stats = []
        for feature in features:
            data = self.__parent[feature]
            if indices is not None:
                data = data.loc[indices]
            stats.extend([
                data.mean()   .reset_index().rename(
                    columns={'index':'target', 0:'value'}).assign(feature=feature, category='statistics', type='mean'),
                data.std()    .reset_index().rename(
                    columns={'index':'target', 0:'value'}).assign(feature=feature, category='statistics', type='std'),
                data.median() .reset_index().rename(
                    columns={'index':'target', 0:'value'}).assign(feature=feature, category='statistics', type='median'),
                data.skew()   .reset_index().rename(
                    columns={'index':'target', 0:'value'}).assign(feature=feature, category='statistics', type='skewness'),
                data.kurtosis()   .reset_index().rename(
                    columns={'index':'target', 0:'value'}).assign(feature=feature, category='statistics', type='kurtosis'),
            ])
        df = pd.concat(stats).reset_index(drop=True)
        if assign_ID:
            df = df.assign(ID=self.__parent.ID)
        if assign_tags:
            df = df.assign(**self.__parent.tags)
        return df
    