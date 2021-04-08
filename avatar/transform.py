import numpy as np
from copy import deepcopy

class Transform:
    def __init__(self, parent=None):
        self.__parent = parent
        
    def __repr__(self):
        return f'Transformer object of {self.__parent}'
    
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
    
    def align_on_axis(self, offset_node='anus', direction_node='chest', axis='x'):
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