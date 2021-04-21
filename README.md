# avatarpy
Python analysis module for AVATAR recording system.

## Setup
```
pip install avatarpy
```

## Quick Start
```python
from avatarpy import Avatar
avatar = Avatar('<your_file>.csv')
```

### 1. Load with sample data
```python
from avatarpy import Avatar, dataset
csv_path = dataset['freely_moving'] # dataset is dict of csv_path provided by avatarpy package
avatar = Avatar(csv_path)
print(avatar.data.head())
```

Output>
```
            0         1         2         3         4         5         6         7         8   ...        18        19        20        21        22        23        24        25        26
0.00  7.563278  6.401241  1.350410  7.052756  5.516540  1.751832  4.075222 -0.440819  1.087099  ...  5.473783  5.572334 -0.249568  5.601366  6.874110 -0.223811  7.233006 -3.937290  3.460134
0.05  7.575983  6.197442  1.385232  7.026716  5.480618  1.777821  4.027716 -0.442995  1.033222  ...  5.482899  5.572342 -0.246786  5.601272  6.864918 -0.220855  7.059306 -4.178260  3.297818
0.10  7.561103  5.993541  1.419173  7.000675  5.444695  1.803809  3.980210 -0.445171  0.979356  ...  5.492015  5.572350 -0.244004  5.601178  6.855827 -0.217890  6.885703 -4.419129  3.135606
0.15  7.531137  5.933699  1.574473  6.960131  5.260929  1.883197  3.941283 -0.480223  0.957676  ...  5.482900  5.581542 -0.246813  5.381383  6.855808 -0.227845  6.823680 -4.460647  3.094629
0.20  7.551146  5.873858  1.731368  6.919584  5.077264  1.962684  3.902356 -0.515285  0.936006  ...  5.492015  5.572350 -0.244004  5.601365  6.855810 -0.223757  6.761554 -4.502164  3.053748

[5 rows x 27 columns]
```

### 2. Available functions and properties in avatar instance
```python
from avatarpy import Avatar
avatar = Avatar('<your_file>.csv')
avatar.help
```

Output>
```
Functions:
        get_angle:
                Calcultes angle of T-series vector (Nx3)
        get_axis_data:
                Returns axis data of all nodes and vectors
        get_distance:
                Calculates euclidian distance of T-series vector (Nx3)
        get_dot_product:
                Calculates dot product of two T-series vectors (Nx3)
        get_node:
                Returns T-series node x, y, z coords by assigned index of columns
        get_node_data:
                Returns coords data with labels of provided nodes names
        get_projection:
                Project 3D coordinates on to axis or plane
        get_rotation_matrix:
                Returns rotation matrix require for rotation of vector a onto vector b
        get_rotation_matrix_of_two_unit_vectors:
                Rotates unit vector a onto unit vector b
        get_unit_vector:
                Returns numpy 2d array (N x 3) of given axis unit vector
        get_vector:
                Returns T-series vector x, y, z coords by assigned index of columns
        get_vector_coords:
                Retruns dataframe dicts of nodes given by vector name.
        get_xaxis_rotation_matrix:
                Returns rotation matrix for x axis rotation when angles are given
        get_yaxis_rotation_matrix:
                Returns rotation matrix for y axis rotation when angles are given
        get_zaxis_rotation_matrix:
                Returns rotation matrix for z axis rotation when angles are given
        set_nodes:
                Set node attributes in avatar with predefined node info in `cls._nodes`
        set_vectors:
                Set vector attributes in avatar with predefined vector info in `cls._nodes`
        x_projection:
                Project 3D coordinates on to x axis
        xy_projection:
                Project 3D coordinates on to xy plane
        xz_projection:
                Project 3D coordinates on to xz plane
        y_projection:
                Project 3D coordinates on to y axis
        yz_projection:
                Project 3D coordinates on to yz plane
        z_projection:
                Project 3D coordinates on to z axis
Attributes:
        ID:
                User provided ID for avatar instance. (default: csv_path)
        acceleration:
                Returns moment acceleration of all coords
        angle:
                Returns T-series angles between predefined two vectors
        animate:
                Animation module for coordinate change
        csv_path:
                Original coordinates file path
        cummulative_distance:
                Returns cumulative distance of all coords
        data:
                Raw data of repeated x, y, z coordinates of nodes
        distance:
                Returns inter-frame distances of all coords
        frame_rate:
                Number of frames recorded in 1 second. (a.k.a. data rate, sampling rate)
        help:
                Prints list of available functions and attributes of avatar
        index:
                Index of recording timecourse. If frame rate is provided, unit is second
        node_data:
                Coords data with labels of all nodes. [x, y, z, node]
        nodes:
                Dictionary of all nodes data
        total_distance:
                Returns total explored distance of all coords
        transform:
                Transform module for coordinate change
        unit_vector_x:
                Numpy 2d array (N x 3) of x unit vector [1,0,0]
        unit_vector_y:
                Numpy 2d array (N x 3) of y unit vector [0,1,0]
        unit_vector_z:
                Numpy 2d array (N x 3) of z unit vector. [0,0,1]
        vector_length:
                Returns length of all vectors
        vector_length_zscore:
                Returns zscores of vector length from all vectors
        vectors:
                Dictionary of all vectors data
        velocity:
                Returns moment velocity of all coords
        x:
                x coordindates of nodes and vectors
        x_max:
                Maximum x coord value among all nodes
        x_min:
                Minimum x coord value among all nodes
        y:
                y coordindates of nodes and vectors
        y_max:
                Maximum y coord value among all nodes
        y_min:
                Minimum y coord value among all nodes
        z:
                z coordindates of nodes and vectors
        z_max:
                Maximum z coord value among all nodes
        z_min:
                Minimum z coord value among all nodes
```

### 3. Animate avatar in plotly
```python
from avatarpy import Avatar, dataset
csv_path = dataset['freely_moving'] # dataset is dict of csv_path provided by avatarpy package
avatar = Avatar(csv_path)
avatar.animate(avatar.index[0:100]).save('freely_moving_0_to_100.html')
# avatar.animate(avatar.index[:100]).show()
```
You can check sample avatar animation html file with link below created by plotly.
https://jeakwon.github.io/avatarpy/freely_moving_0_to_100.html