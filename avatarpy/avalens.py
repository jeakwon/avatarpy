import os
import numpy as np
import pandas as pd
from datetime import datetime
from avatarpy import Avatar

class AvaLens:
    def __init__(self, id_policy='filepath', tag_policy='provide'):
        r"""Lens for group analysis of avatars

        :param id_policy: {'filepath'(default))|'incremental'|'provide'|'basename'}
        :param tag_policy: {'provide'(default))|'dirname'}
        """
        assert id_policy in ['filepath','incremental','provide','basename'], 'wrong argument for id_policy'
        assert tag_policy in ['provide','dirname'], 'wrong argument for id_policy'
        self._avatars = []
        self.id_policy = id_policy
        self.tag_policy = tag_policy

    def __repr__(self):
        return f'AvaLens instance containing avatars: {self.avatars}'

    @property
    def avatars(self):
        """User Added avatars"""
        return self._avatars

    def add_file(self, csv_path, ID=None, tags={}, verbose=1):
        if self.id_policy == 'filepath':
            ID = csv_path
        elif self.id_policy == 'incremental':
            ID = len(self.avatars)
        elif self.id_policy == 'provide':
            assert ID is not None, 'User should provide ID or select id_policy among ["filpath", "incremental", "basname"]'
            ID = ID
        elif self.id_policy == 'basename':
            ID = os.path.splitext(os.path.basename(csv_path))[0]

        if self.tag_policy == 'dirname':
            tags = dict(tag=os.path.basename(os.path.dirname(csv_path)))
        elif self.tag_policy == 'provide':
            tags = tags
        avatar = Avatar(csv_path=csv_path, ID=ID, tags=tags)
        self.avatars.append(avatar)
        if verbose==1:
            print(f'[{datetime.now()}] Added new Avatar(csv_path={csv_path}, ID={ID}, tags={tags})', end='\r')
        if verbose==2:
            print(f'[{datetime.now()}] Added new Avatar(csv_path={csv_path}, ID={ID}, tags={tags})')
        return self

    def add_folder(self, root, ID=None, tags={}, verbose=1):
        for path, subdirs, files in os.walk(root):
            for name in files:
                if name.lower().endswith('.csv'):
                    csv_path = os.path.join(path, name)
                    self.add_file(csv_path, ID, tags, verbose=verbose)
        return self

    def describe(self, indices=None, include=['corr', 'stat'], assign_ID=True, assign_tags=True):
        describes = []
        for avatar in self.avatars: 
            desc = avatar.describe(indices=indices, include=include, assign_ID=assign_ID, assign_tags=assign_tags)
            describes.append(desc)
        return pd.concat(describes).reset_index(drop=True)

    @property
    def search_event(self):
        return SearchEvent(parent=self)

class SearchEvent:
    def __init__(self, parent=None):
        self.__parent = parent
        self.__events = []
        self.__event_name = ''

    def __call__(self, func, name, length=20, verbos=2):
        """Search event by given function.
        """
        assert callable(func), 'func should be callable'
        self.__events = []
        self.__event_name = name
        for avatar in self.__parent.avatars:
            boolean_series = func(avatar)
            assert isinstance(boolean_series, pd.Series), 'func should return pd.Series of boolean with index'
            assert boolean_series.dtype == bool, 'dtype of boolean_series should be bool'
            events = self.split_boolean_series(boolean_series)
            filtered_events = []
            for arr in events:
                if len(arr)<length:
                    continue
                else:
                    while len(arr)>=length:
                        filtered_events.append(arr[:length])
                        arr = arr[length:]
            if verbos==1:
                print(f'Total {len(filtered_events)} event was detected', end='\r')
            if verbos==2:
                print(f'Total {len(filtered_events)} event was detected')
            self.__events.append(filtered_events)
        return self

    @staticmethod
    def split_boolean_series(boolean_series):
        indices, boolean = boolean_series.index, boolean_series.values
        pos = np.nonzero(boolean[1:] != boolean[:-1])[0] + 1
        arrs = np.split(indices, pos)
        arrs = arrs[0::2] if boolean[0] else arrs[1::2]
        return arrs

    def describe(self, include=['corr', 'stat'], assign_ID=True, assign_tags=True, assign_event_name=True):
        describes = []
        for avatar, events in zip(self.__parent.avatars, self.__events) :
            for indices in events:
                desc = avatar.describe(indices=indices, include=include, assign_ID=assign_ID, assign_tags=assign_tags)
                describes.append(desc)
        df = pd.concat(describes).reset_index(drop=True)
        if assign_event_name:
            df = df.assign(event=self.__event_name)
        return df

    #TODO
    # def coord(self, incldue=['raw', 'align_on_axis', 'align_on_plane'], assign_event_name=True)
