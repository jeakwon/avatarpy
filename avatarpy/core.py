import numpy as np
import pandas as pd
from scipy.signal import correlate

class Core:   
    def __getitem__(self, item):
        return getattr(self, item)
        
    def get_dot_product(self, vector1, vector2):
        r"""Calculates dot product of two T-series vectors (Nx3)
        """
        return np.einsum('ij,ij->i', vector1, vector2)
        
    def get_cross_product(self, vector1, vector2):
        r"""Calculates cross product of two T-series vectors (Nx3)
        """
        return np.cross(vector1, vector2)
    
    def get_distance(self, vector):
        r"""Calculates euclidian distance of T-series vector (Nx3)
        """
        return np.sqrt(self.get_dot_product(vector, vector))
    
    def get_angle(self, vector1, vector2):
        r"""Calcultes angle of T-series vector (Nx3)
        """
        numerator = self.get_dot_product(vector1, vector2)
        denominator = self.get_distance(vector1)*self.get_distance(vector2)
        return np.arccos(numerator/denominator)
    
    def get_triangular_area_by_vectors(self, vector1, vector2):
        r"""Calcultes triangular area of T-series vector (Nx3)
        """
        cross_product = np.cross(vector1, vector2)
        return np.linalg.norm(cross_product, axis=1)/2
    
    def get_triangular_area_by_coords(self, coord1, coord2, coord3):
        r"""Calcultes triangular area by 3 T-series coords (Nx3)
        """
        vector1, vector2 = coord1-coord3, coord2-coord3
        return self.get_triangular_area_by_vectors(vector1, vector2)
    
    def get_rotation_matrix_of_two_unit_vectors(self, unit_vector_a, unit_vector_b):
        r"""Rotates unit vector a onto unit vector b
        """
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        v = np.cross(unit_vector_a, unit_vector_b)
        N = v.shape[0]
        s = self.get_distance(v) # sine of angle
        c = self.get_dot_product(unit_vector_a, unit_vector_b) # cos of angle
        I = np.stack([np.eye(3)]*N)
        vx = np.zeros((N, 3, 3))
        vx[:, 2, 1]= v[:, 0]
        vx[:, 1, 2]= -v[:, 0]
        vx[:, 0, 2]= v[:, 1]
        vx[:, 2, 0]= -v[:, 1]
        vx[:, 1, 0]= v[:, 2]
        vx[:, 0, 1]= -v[:, 2]
        vx_2 = np.einsum('nij,njk->nik', vx, vx)
        const = (1-c)/s**2
        eq0, eq1, eq2 = I, vx, vx_2*const[:, np.newaxis, np.newaxis]
        R = eq0+eq1+eq2
        return R
    
    def get_rotation_matrix(self, vector_a, vector_b):
        r"""Returns rotation matrix require for rotation of vector a onto vector b
        
        - Assumed vectors are fixed at( 0, 0, 0)
        - Scale does not change
        """
        unit_vector_a = vector_a/self.get_distance(vector_a).reshape(-1, 1)
        unit_vector_b = vector_b/self.get_distance(vector_b).reshape(-1, 1)
        return self.get_rotation_matrix_of_two_unit_vectors(unit_vector_a, unit_vector_b)
    
    def get_xaxis_rotation_matrix(self, angles):
        r"""Returns rotation matrix for x axis rotation when angles are given
        """
        n = len(angles)
        s = np.cos(angles)
        c = np.sin(angles)
        R = np.zeros([n, 3, 3])
        R[:, 0, 0]=1
        R[:, 1, 1]=c
        R[:, 1, 2]=-s
        R[:, 2, 1]=s
        R[:, 2, 2]=c
        return R
    
    def get_yaxis_rotation_matrix(self, angles):
        r"""Returns rotation matrix for y axis rotation when angles are given
        """
        n = len(angles)
        s = np.cos(angles)
        c = np.sin(angles)
        R = np.zeros([n, 3, 3])
        R[:, 1, 1]=1
        R[:, 0, 0]=c
        R[:, 0, 2]=-s
        R[:, 2, 0]=s
        R[:, 2, 2]=c
        return R
    
    def get_zaxis_rotation_matrix(self, angles):
        r"""Returns rotation matrix for z axis rotation when angles are given
        """
        n = len(angles)
        s = np.cos(angles)
        c = np.sin(angles)
        R = np.zeros([n, 3, 3])
        R[:, 2, 2]=1
        R[:, 0, 0]=c
        R[:, 0, 1]=-s
        R[:, 1, 0]=s
        R[:, 1, 1]=c
        return R

    def flatten_pairwise_df(self, df, diagonal_only=True):
        r"""Flatten the pairwise symmetric df
        """
        if diagonal_only:
            mask = np.tril(np.ones(df.shape).astype(np.bool))
            s = df.mask(mask).stack()
        else:
            s = df.stack()
        s.index = s.index.map(lambda x: '_'.join(x[-2:]))
        return s

    def get_rolling_corr(self, df, window=20, center=True, **kwargs):
        r"""Returns flattened rolling correlations.
        """
        rolling_corr = df.rolling(window, center=center, **kwargs).corr()
        return rolling_corr.groupby(level=0).apply(lambda x: self.flatten_pairwise_df(x)).unstack()

    @staticmethod
    def xcorr(in1, in2=None):
        r"""Calculates cross correlation of two inputs

        :params in1: np.array or pd.Series
        :params in2: np.array or pd.Series
        :params normalize: bool
        :returns: dict['lags', 'corr', 'corr_max', 'lag']
        """
        def inval(x):
            if isinstance(x, pd.Series): 
                x = x.to_numpy()

            x = (x-x.mean())/x.std()
            return x

        def correlation_lags(in1_len, in2_len):
            return  np.arange(-in2_len + 1, in1_len)

        x1 = inval(in1)
        x2 = inval(in2)
        corr = correlate(x1, x2)

        lags = correlation_lags(len(x1), len(x2))
        lag = lags[corr.argmax()]

        corr /= len(corr)-1
        max_corr = np.clip(corr.max(), -1, 1)
        return dict(lags=lags, corr=corr, max=max_corr, lag=lag)

    @staticmethod
    def pairwise_apply_func(df, func):
        ndf = df._get_numeric_data()
        lbl = ndf.columns
        mat = ndf.to_numpy().T

        K = len(lbl)
        data = np.empty((K, K), dtype=float)
        mask = np.isfinite(mat)
        for i, ac in enumerate(mat):
            for j, bc in enumerate(mat):
                valid = mask[i] & mask[j]
                if valid.sum() < 1:
                    c = np.nan
                elif not valid.all():
                    c = func(ac[valid], bc[valid])
                else:
                    c = func(ac, bc)
                data[i, j] = c
        return pd.DataFrame(data, index=lbl, columns=lbl)

