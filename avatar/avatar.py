from avatar.core import Core
from avatar.transform import Transform
from avatar.animate import Animate

import numpy as np
import pandas as pd
from scipy.stats import zscore

class Avatar(Core):
    _nodes={
        'nose'  :[0,1,2],
        'neck'  :[3,4,5],   
        'anus'  :[6,7,8],
        'chest' :[9,10,11],
        'rfoot' :[12,13,14],
        'lfoot' :[15,16,17],
        'rhand' :[18,19,20],
        'lhand' :[21,22,23],
        'tip'   :[24,25,26],
    }
    
    _vectors={
        'head'      :{'head':'nose',    'tail':'neck'},
        'fbody'     :{'head':'neck',    'tail':'chest'},
        'hbody'     :{'head':'chest',   'tail':'anus'},
        'tail'      :{'head':'tip',     'tail':'anus'},
        'rarm'      :{'head':'rhand',   'tail':'chest'},
        'larm'      :{'head':'lhand',   'tail':'chest'},
        'rleg'      :{'head':'rfoot',   'tail':'anus'},
        'lleg'      :{'head':'lfoot',   'tail':'anus'},
        'nose2anus' :{'head':'anus',   'tail':'nose'},
        'lhand2nose':{'head':'nose',   'tail':'lhand'},
        'rhand2nose':{'head':'nose',   'tail':'rhand'},
    }
    
    _angles={
        'head':{'left':'head', 'right':'fbody'},
        'body':{'left':'fbody', 'right':'hbody'},
        'tail':{'left':'hbody', 'right':'tail'},
        'larm':{'left':'larm', 'right':'fbody'},
        'rarm':{'left':'rarm', 'right':'fbody'},
        'lleg':{'left':'lleg', 'right':'hbody'},
        'rleg':{'left':'rleg', 'right':'hbody'},
    }

    def __init__(self, csv_path, frame_rate=20, ID=None, horizontal_correction=True):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path, header=None)
        self.frame_rate=frame_rate
        if frame_rate: self.data.index/=frame_rate
        self.ID=ID if ID else self.csv_path
        self.tags=[]
        self.set_nodes()
        self.set_vectors()
        
        if horizontal_correction:
            self.data = self.transform.level().data
            self.set_nodes()
            self.set_vectors()


    def __repr__(self):
        tags = '\n'.join('# '+tag for tag in self.tags)
        return f'Avatar({self.ID})\n{tags}'
    
    def add_tag(self, tag):
        assert isinstance(tag, str), 'tag should be str'
        self.tags.append(tag)
        return self
            
    def set_nodes(self):
        for name, cols in self._nodes.items():
            data = self.get_node(cols)
            setattr(self, name, data)
    
    def set_vectors(self):
        for name, labels in self._vectors.items():
            data = self.get_vector(self[labels['head']], self[labels['tail']])
            setattr(self, name, data)
            
    @property
    def index(self):return self.data.index
    @property
    def nodes(self):return {key:self[key] for key in self._nodes.keys()}
    @property
    def vectors(self):return {key:self[key] for key in self._vectors.keys()}
    @property
    def x(self): return self.get_axis_data('x')
    @property
    def y(self): return self.get_axis_data('y')
    @property
    def z(self): return self.get_axis_data('z')
    @property
    def x_max(self):return self.x[self.nodes].values.max()
    @property
    def x_min(self):return self.x[self.nodes].values.min()
    @property
    def y_max(self):return self.y[self.nodes].values.max()
    @property
    def y_min(self):return self.y[self.nodes].values.min()
    @property
    def z_max(self):return self.z[self.nodes].values.max()
    @property
    def z_min(self):return self.z[self.nodes].values.min()
    @property
    def unit_vector_x(self):return self.get_unit_vector(axis='x')
    @property
    def unit_vector_y(self):return self.get_unit_vector(axis='y')
    @property
    def unit_vector_z(self):return self.get_unit_vector(axis='z')
    @property
    def node_data(self):
        return self.get_node_data(self.nodes.keys())

    def get_node_data(self, nodes):
        labeled_data = [self[node].assign(node=node) for node in nodes]
        return pd.concat(labeled_data).sort_index()

    def get_unit_vector(self, axis):
        if axis=='x': arr = np.array([1,0,0])
        elif axis=='y': arr = np.array([0,1,0])
        elif axis=='z': arr = np.array([0,1,0])
        return np.stack([arr]*len(self.index))
    
    def get_node(self, columns):
        return self.data[columns].set_axis(['x', 'y', 'z'], axis=1, inplace=False)
    
    def get_vector(self, head, tail):
        return pd.DataFrame(head.values-tail.values, columns=['x', 'y', 'z']).set_index(self.data.index)

    def get_axis_data(self, coord):
        data_dict = {}
        for name, data in self.nodes.items():
            data_dict[name]=data[coord].values
        for name, data in self.vectors.items():
            data_dict[name]=data[coord].values
        return pd.DataFrame(data_dict).set_index(self.data.index)
    
    @property
    def angle(self):
        data_dict = {}
        for name, vectors in self._angles.items():
            data_dict[name]=self.get_angle(self[vectors['left']], self[vectors['right']])
        return pd.DataFrame(data_dict).set_index(self.data.index)

    @property
    def vector_length(self):
        data_dict = {}
        for name, vector in self.vectors.items():
            data_dict[name]=self.get_distance(vector)
        return pd.DataFrame(data_dict).set_index(self.data.index)

    @property
    def vector_flexibility(self):
        return self.vector_length.apply(zscore)
    
    @property
    def distance(self):
        data_dict = {}
        for name, node in self.nodes.items():
            data_dict[name]=self.get_distance(node.diff())
        return pd.DataFrame(data_dict).set_index(self.data.index)
    
    @property
    def velocity(self):
        return self.distance*self.frame_rate
    
    @property
    def acceleration(self):
        return self.velocity.diff()*self.frame_rate
    
    @property
    def cummulative_distance(self):
        return self.distance.cumsum()
    
    @property
    def total_distance(self):
        return self.cummulative_distance.max()

    def get_vector_coords(self, vector_name):
        return dict(
            x = self.x[self._vectors[vector_name].values()],
            y = self.y[self._vectors[vector_name].values()],
            z = self.z[self._vectors[vector_name].values()],
        )
    
    def get_projection(self, vector, to):
        zero_axis = list(set('xyz')-set(to))
        kwargs = {axis:0 for axis in zero_axis}
        return vector.assign(**kwargs)
    
    def x_projection(self, node): 
        return self.get_projection(vector=self[node], to='x')
    
    def y_projection(self, node): 
        return self.get_projection(vector=self[node], to='y')
    
    def z_projection(self, node): 
        return self.get_projection(vector=self[node], to='z')
    
    def xy_projection(self, node): 
        return self.get_projection(vector=self[node], to='xy')
    
    def yz_projection(self, node): 
        return self.get_projection(vector=self[node], to='xz')
    
    def xz_projection(self, node): 
        return self.get_projection(vector=self[node], to='yz')
    
    @property
    def transform(self):
        return Transform(parent=self)
    
    @property
    def animate(self):
        return Animate(parent=self)