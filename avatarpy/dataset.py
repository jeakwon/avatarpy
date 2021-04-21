__all__ = ['datasets']

import os
import pkg_resources
import avatarpy

DATA_PATH = pkg_resources.resource_filename("avatarpy", 'data/')
FILES = os.listdir(DATA_PATH)
datasets = dict()
for FILE in FILES:
    name, ext = os.path.splitext(os.path.basename(FILE))
    if ext.lower()=='.csv':
        datasets[name] = os.path.join(DATA_PATH, FILE)