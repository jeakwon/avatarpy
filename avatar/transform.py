import numpy as np
from copy import deepcopy
from sklearn.linear_model import LinearRegression

class Transform:
    def __init__(self, parent=None):
        self.__parent = parent
        
    def __repr__(self):
        return f'Transformer object of {self.__parent}'

    def level(self):
        """수평맞추기
        """
        avatar = deepcopy(self.__parent)
        data = avatar.get_node_data(['lfoot', 'rfoot'])

        # Horizontal Regression on lfoot and rfoot node data
        model = LinearRegression() 
        model.fit(data[['x','y']], data[['z']])
        a, b = model.coef_[0]
        vector1 = np.stack([np.array([-a, -b, 1])]*len(avatar.data))
        vector2 = np.stack([np.array([ 0,  0, 1])]*len(avatar.data))
        R = avatar.get_rotation_matrix(vector1, vector2)
        avatar = avatar.transform.rotate(R)
        return avatar

    def add(self, vector):
        avatar = deepcopy(self.__parent)
        for k, v in avatar.nodes.items():
            cols = avatar._nodes[k]
            avatar.data[cols] = v+vector
        avatar.set_nodes()
        avatar.set_vectors()
        return avatar
    
    def sub(self, vector):
        avatar = deepcopy(self.__parent)
        for k, v in avatar.nodes.items():
            cols = avatar._nodes[k]
            avatar.data[cols] = v-vector
        avatar.set_nodes()
        avatar.set_vectors()
        return avatar

    def fix(self, node):
        avatar = deepcopy(self.__parent)
        for k, v in avatar.nodes.items():
            cols = avatar._nodes[k]
            avatar.data[cols] = v-avatar[node]
        avatar.set_nodes()
        avatar.set_vectors()
        return avatar
    
    def rotate(self, rotation_matrix):
        avatar = deepcopy(self.__parent)
        for k, v in avatar.nodes.items():
            cols = avatar._nodes[k]
            avatar.data[cols] = np.einsum('nij,nj->ni', rotation_matrix, v)
        avatar.set_nodes()
        avatar.set_vectors()
        return avatar
    
    def align_on_axis(self, offset_node='anus', direction_node='chest', axis='y'):
        avatar = deepcopy(self.__parent)
        avatar = avatar.transform.fix(node=offset_node)
        R = avatar.get_rotation_matrix(avatar[direction_node], avatar.get_unit_vector(axis=axis))
        avatar = avatar.transform.rotate(R)
        return avatar
    
    def align_on_plane(self, offset_node='anus', direction_node='chest', plane='xz'):
        axis = next(iter(set('xyz')-set(plane)))
        avatar = deepcopy(self.__parent)
        avatar = avatar.transform.fix(node=offset_node)
        R = avatar.get_rotation_matrix(avatar.xy_projection(direction_node), avatar.get_unit_vector(axis=axis))
        avatar = avatar.transform.rotate(R)
        return avatar