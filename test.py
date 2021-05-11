from avatarpy import Avatar, AvaLens, dataset
import os
import time
import numpy as np
import pandas as pd

# lens = AvaLens(id_policy='basename', tag_policy='provide')
# lens.add_folder(root=r"C:\Users\Jay\Desktop\avatarpy\avatarpy\data")
# print(lens.avatars)
# print(lens.search_event(func=lambda avatar: avatar.velocity['anus']>2, name='walk').describe())

# print(avatar.describe.corr())
# print(avatar.describe.stat())
# print(avatar.describe())

csv_path = dataset['freely_moving']
avatar = Avatar(csv_path)
annot = avatar.annotation(r"C:\Users\Jay\Desktop\annotex.csv")
annot = avatar.annotation(by=lambda avatar: (avatar['velocity']<0.5).all(axis=1), name='freeze')
indices = avatar.annotation.get_indices('freeze')