from avatarpy import Avatar, AvaLens, dataset
import os
import time
import numpy as np
import pandas as pd
from pprint import pprint

csv_path = dataset['freely_moving']
avatar = Avatar(csv_path)
print(avatar.area)