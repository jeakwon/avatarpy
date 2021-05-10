from avatarpy import Avatar, AvaLens, dataset
import os
import time
import numpy as np
import pandas as pd

lens = AvaLens(id_policy='basename', tag_policy='provide')
lens.add_folder(root=r"C:\Users\Jay\Desktop\avatarpy\avatarpy\data")
print(lens.avatars)
print(lens.search_event(func=lambda avatar: avatar.velocity['anus']>2, name='walk').describe())
# arr = np.array([True, False])
# print(type())
# print(lens.event_features(event_name='walk', ))

# print(avatar.describe.corr())
# print(avatar.describe.stat())
# print(avatar.describe())