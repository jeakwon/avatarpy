from avatarpy import Avatar, AvaLens, dataset
import os
import time
import numpy as np
import pandas as pd
from pprint import pprint

# lens = AvaLens(id_policy='basename', tag_policy='provide')
# lens.add_folder(root=r"C:\Users\Jay\Desktop\avatarpy\avatarpy\data")
# df = lens.describe(func_kws=dict(
#         walk=lambda avatar: (avatar['velocity']['anus']>2)
#     )
# )




# print(avatar.describe.corr())
# print(avatar.describe.stat())
# print(avatar.describe())

csv_path = dataset['freely_moving']
avatar = Avatar(csv_path)
print(avatar.gather())
# print(avatar.gather_corrs())
# annot = avatar.annotation(r"C:\Users\Jay\Desktop\annotex.csv")
# avatar.annotation.add(by=lambda avatar: (avatar['velocity']<0.5).all(axis=1), name='walk_search')
# avatar.annotation.add(by=lambda avatar: (avatar['acceleration']<0.5).all(axis=1), name='walk_human')
# indices = avatar.annotation.get_indices('freeze')
# df = avatar.annotation()
# print(df)
# metrics = avatar.annotation.metrics(true = df['walk_search'], pred = df['walk_human'])
# pprint(metrics)
