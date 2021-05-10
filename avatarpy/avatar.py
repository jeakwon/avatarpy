from avatarpy.core import Core
from avatarpy.transform import Transform
from avatarpy.animate import Animate
from avatarpy.describe import Describe


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

    def __init__(self, csv_path, frame_rate=20, ID=None, tags={}, horizontal_correction=True):
        self._csv_path = csv_path
        self._data = pd.read_csv(csv_path, header=None)
        self._frame_rate = frame_rate
        if frame_rate: self.data.index/=frame_rate
        self._ID=ID if ID else self.csv_path
        self._tags = tags
        self.set_nodes()
        self.set_vectors()
        
        if horizontal_correction:
            self.data = self.transform.level().data
            self.set_nodes()
            self.set_vectors()

    @property
    def csv_path(self):
        """Original coordinates file path"""
        return self._csv_path
    @csv_path.setter
    def csv_path(self, v):
        self._csv_path = v
    @property
    def data(self):
        """Raw data of repeated x, y, z coordinates of nodes"""
        return self._data
    @data.setter
    def data(self, v):
        self._data = v
    @property
    def frame_rate(self):
        """Number of frames recorded in 1 second. (a.k.a. data rate, sampling rate)"""
        return self._frame_rate
    @frame_rate.setter
    def frame_rate(self, v):
        self._frame_rate = v
    @property
    def ID(self):
        """User provided ID for avatar instance. (default: csv_path)"""
        return self._ID
    @ID.setter
    def ID(self, v):
        self._ID = v
    @property
    def tags(self):
        """User provided tags for avatar instance. add tags by dict (ex, {'genotype':'wt'})"""
        return self._tags
    @tags.setter
    def tags(self, v):
        self._tags = v

    def __repr__(self):
        return f'Avatar({self.ID})'

    @property
    def help(self):
        """Prints list of available functions and attributes of avatar"""
        functions, attributes  = [], []
        for item in dir(self.__class__):
            if item.startswith('_'):
                continue
            docstring = getattr(self.__class__, item).__doc__
            headline = docstring.split('\n')[0].strip() if docstring else ''
            if callable(getattr(self.__class__, item)):
                functions.append(f"\t{item}: \n\t\t{headline}")
            else:
                attributes.append(f"\t{item}: \n\t\t{headline}")
        funcs = '\n'.join(functions)
        attrs = '\n'.join(attributes)
        print(f"""Visit https://github.com/jeakwon/avatarpy\n
            Functions:\n{funcs}\nAttributes:\n{attrs}""")
            
    def set_nodes(self):
        """Set node attributes in avatar with predefined node info in `cls._nodes`"""
        for name, cols in self._nodes.items():
            data = self.get_node(cols)
            setattr(self, name, data)
    
    def set_vectors(self):
        """Set vector attributes in avatar with predefined vector info in `cls._nodes`"""
        for name, labels in self._vectors.items():
            data = self.get_vector(self[labels['head']], self[labels['tail']])
            setattr(self, name, data)
            
    @property
    def index(self):
        """Index of recording timecourse. If frame rate is provided, unit is second"""
        return self.data.index

    @property
    def nodes(self):
        """Dictionary of all nodes data"""
        return {key:self[key] for key in self._nodes.keys()}

    @property
    def vectors(self):
        """Dictionary of all vectors data"""
        return {key:self[key] for key in self._vectors.keys()}

    @property
    def x(self): 
        """x coordindates of nodes and vectors"""
        return self.get_axis_data('x')

    @property
    def y(self): 
        """y coordindates of nodes and vectors"""
        return self.get_axis_data('y')

    @property
    def z(self): 
        """z coordindates of nodes and vectors"""
        return self.get_axis_data('z')

    @property
    def x_max(self):
        """Maximum x coord value among all nodes"""
        return self.x[self.nodes].values.max()

    @property
    def x_min(self):
        """Minimum x coord value among all nodes"""
        return self.x[self.nodes].values.min()

    @property
    def y_max(self):
        """Maximum y coord value among all nodes"""
        return self.y[self.nodes].values.max()

    @property
    def y_min(self):
        """Minimum y coord value among all nodes"""
        return self.y[self.nodes].values.min()

    @property
    def z_max(self):
        """Maximum z coord value among all nodes"""
        return self.z[self.nodes].values.max()

    @property
    def z_min(self):
        """Minimum z coord value among all nodes"""
        return self.z[self.nodes].values.min()

    @property
    def unit_vector_x(self):
        """Numpy 2d array (N x 3) of x unit vector [1,0,0]"""
        return self.get_unit_vector(axis='x')

    @property
    def unit_vector_y(self):
        """Numpy 2d array (N x 3) of y unit vector [0,1,0]"""
        return self.get_unit_vector(axis='y')

    @property
    def unit_vector_z(self):
        """Numpy 2d array (N x 3) of z unit vector. [0,0,1]"""
        return self.get_unit_vector(axis='z')

    @property
    def node_data(self):
        """Coords data with labels of all nodes. [x, y, z, node]"""
        return self.get_node_data(self.nodes.keys())

    def get_node_data(self, nodes):
        """Returns coords data with labels of provided nodes names"""
        labeled_data = [self[node].assign(node=node) for node in nodes]
        return pd.concat(labeled_data).sort_index()

    def get_unit_vector(self, axis):
        """Returns numpy 2d array (N x 3) of given axis unit vector"""
        if axis=='x': arr = np.array([1,0,0])
        elif axis=='y': arr = np.array([0,1,0])
        elif axis=='z': arr = np.array([0,1,0])
        return np.stack([arr]*len(self.index))
    
    def get_node(self, columns):
        """Returns T-series node x, y, z coords by assigned index of columns"""
        return self.data[columns].set_axis(['x', 'y', 'z'], axis=1, inplace=False)
    
    def get_vector(self, head, tail):
        """Returns T-series vector x, y, z coords by assigned index of columns"""
        return pd.DataFrame(head.values-tail.values, columns=['x', 'y', 'z']).set_index(self.data.index)

    def get_axis_data(self, axis):
        """Returns axis data of all nodes and vectors"""
        data_dict = {}
        for name, data in self.nodes.items():
            data_dict[name]=data[axis].values
        for name, data in self.vectors.items():
            data_dict[name]=data[axis].values
        return pd.DataFrame(data_dict).set_index(self.data.index)
    
    @property
    def angle(self):
        """Returns T-series angles between predefined two vectors"""
        data_dict = {}
        for name, vectors in self._angles.items():
            data_dict[name]=self.get_angle(self[vectors['left']], self[vectors['right']])
        return pd.DataFrame(data_dict).set_index(self.data.index)
    
    @property
    def angle_velocity(self):
        """Returns T-series angles velocity between predefined two vectors"""
        return self.angle.diff()*self.frame_rate
    
    @property
    def angle_acceleration(self):
        """Returns T-series angles acceleration between predefined two vectors"""
        return self.angle_velocity.diff()*self.frame_rate

    @property
    def vector_length(self):
        """Returns length of all vectors"""
        data_dict = {}
        for name, vector in self.vectors.items():
            data_dict[name]=self.get_distance(vector)
        return pd.DataFrame(data_dict).set_index(self.data.index)

    @property
    def stretch_index(self):
        """Returns stretch_index which is equal to zscore of vector length"""
        return self.vector_length.apply(zscore)
    
    @property
    def distance(self):
        """Returns inter-frame distances of all coords"""
        data_dict = {}
        for name, node in self.nodes.items():
            data_dict[name]=self.get_distance(node.diff())
        return pd.DataFrame(data_dict).set_index(self.data.index)
    
    @property
    def velocity(self):
        """Returns moment velocity of all coords"""
        return self.distance*self.frame_rate
    
    @property
    def acceleration(self):
        """Returns moment acceleration of all coords"""
        return self.velocity.diff()*self.frame_rate
    
    @property
    def cummulative_distance(self):
        """Returns T-series cumulative distance of all coords"""
        return self.distance.cumsum()
    
    @property
    def total_distance(self):
        """Returns total explored distance of all coords """
        return self.cummulative_distance.max()

    def get_vector_coords(self, vector_name):
        """Retruns dataframe dicts of nodes given by vector name."""
        return dict(
            x = self.x[self._vectors[vector_name].values()],
            y = self.y[self._vectors[vector_name].values()],
            z = self.z[self._vectors[vector_name].values()],
        )
    
    def get_projection(self, vector, to):
        """Project 3D coordinates on to axis or plane"""
        zero_axis = list(set('xyz')-set(to))
        kwargs = {axis:0 for axis in zero_axis}
        return vector.assign(**kwargs)
    
    def x_projection(self, node): 
        """Project 3D coordinates on to x axis"""
        return self.get_projection(vector=self[node], to='x')
    
    def y_projection(self, node): 
        """Project 3D coordinates on to y axis"""
        return self.get_projection(vector=self[node], to='y')
    
    def z_projection(self, node): 
        """Project 3D coordinates on to z axis"""
        return self.get_projection(vector=self[node], to='z')
    
    def xy_projection(self, node): 
        """Project 3D coordinates on to xy plane"""
        return self.get_projection(vector=self[node], to='xy')
    
    def yz_projection(self, node): 
        """Project 3D coordinates on to yz plane"""
        return self.get_projection(vector=self[node], to='yz')
    
    def xz_projection(self, node): 
        """Project 3D coordinates on to xz plane"""
        return self.get_projection(vector=self[node], to='xz')
    
    @property
    def transform(self):
        """Transform module for coordinate change"""
        return Transform(parent=self)
    
    @property
    def animate(self):
        """Animation module for coordinate change"""
        return Animate(parent=self)

    @property
    def describe(self):
        """Feature Extract module for coordinate change"""
        return Describe(parent=self)

    def corr(self, data, window=None, center=True, **kwargs):
        """Returns rolling correlation with given property of data
        
        :param data: (str|pd.DataFrame)
        :param window: (int|None) if provided rolling corr is provided
        """
        if isinstance(data, str):
            data = self[data]
        if window==None:
            return self.flatten_pairwise_df(data.corr())
        return self.get_rolling_corr(data, window, center, **kwargs)

    def xcorr_lag(self, data, flatten=True):
        """Returns cross correlation time lag by parwise column calculation. See also xcorr_max
        """
        if isinstance(data, str):
            data = self[data]
        df = self.pairwise_apply_func(data, lambda in1, in2: self.xcorr(in1, in2)['lag']/self.frame_rate)
        if flatten:
            return self.flatten_pairwise_df(df)
        return df

    def xcorr_max(self, data, flatten=True):
        """Returns cross correlation max value by parwise column calculation. See also xcorr_lag
        """
        if isinstance(data, str):
            data = self[data]
        df = self.pairwise_apply_func(data, lambda in1, in2: self.xcorr(in1, in2)['max'])
        if flatten:
            return self.flatten_pairwise_df(df)
        return data