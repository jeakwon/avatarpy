import numpy as np

class Core:   
    def __getitem__(self, item):
        return getattr(self, item)
        
    def get_dot_product(self, vector1, vector2):
        """Calculates dot product of two T-series vectors (Nx3)
        """
        return np.einsum('ij,ij->i', vector1, vector2)
    
    def get_distance(self, vector):
        """Calculates euclidian distance of T-series vector (Nx3)
        """
        return np.sqrt(self.get_dot_product(vector, vector))
    
    def get_angle(self, vector1, vector2):
        """Calcultes angle of T-series vector (Nx3)
        """
        numerator = self.get_dot_product(vector1, vector2)
        denominator = self.get_distance(vector1)*self.get_distance(vector2)
        return np.arccos(numerator/denominator)
    
    def get_rotation_matrix_of_two_unit_vectors(self, unit_vector_a, unit_vector_b):
        """Rotates unit vector a onto unit vector b
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
        """Returns rotation matrix require for rotation of vector a onto vector b
        
        - Assumed vectors are fixed at( 0, 0, 0)
        - Scale does not change
        """
        unit_vector_a = vector_a/self.get_distance(vector_a).reshape(-1, 1)
        unit_vector_b = vector_b/self.get_distance(vector_b).reshape(-1, 1)
        return self.get_rotation_matrix_of_two_unit_vectors(unit_vector_a, unit_vector_b)
    
    def get_xaxis_rotation_matrix(self, angles):
        """Returns rotation matrix for x axis rotation when angles are given
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
        """Returns rotation matrix for y axis rotation when angles are given
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
        """Returns rotation matrix for z axis rotation when angles are given
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