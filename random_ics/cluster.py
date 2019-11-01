from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from tsfresh import extract_features
from random_ics import *
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


class Cluster:
    """
    aka the environment
    """

    def __init__(self, data, from_pickle=False,
                 pickle_file=EXTRACTED_FEATURE_DATA_PICKLE,
                 algorithm=KMeans, **algorithm_params):
        self.algorithm_params = algorithm_params
        self.algorithm = algorithm(**algorithm_params)
        self.from_pickle = from_pickle
        self.pickle_file = pickle_file
        self.data = data

        self.data = self.extract_features()
        self.original_feature_names = list(self.data.columns)
        # number the columns for ease
        self.data.columns = range(self.data.shape[1])
        self.possibilities = list(self.data.columns)

    def extract_features(self):
        if self.from_pickle and os.path.isfile(self.pickle_file):
            return pd.read_pickle(self.pickle_file)
        else:
            self.data.index.names = ['id', 'time']
            self.data = self.data.reset_index()
            self.data = extract_features(self.data, column_id='id')
            self.data = self.data.replace(0, np.nan).dropna(how='all', axis=1)
            self.data = self.data.loc[:, self.data.ne(1).all()]
            self.data.to_pickle(self.pickle_file)
            self.data.to_csv(os.path.join(PICKLES_DIRECTORY, 'data.csv'))
            return self.data

    def cluster(self, agent):
        data = self.data[agent.features]
        try:
            self.algorithm.fit(data)
            score = silhouette_score(data, self.algorithm.labels_)
        except ValueError:
            score = 0
        return score

    def get_feature_names(self, feature_list):
        return [self.original_feature_names[i] for i in feature_list]


class Agent:

    def __init__(self, features=[]):
        self.features = features  # current state
        # self.policy =

    def __len__(self):
        return len(self.features)

    def pick_a_feature(self, observation):
        """

        Args:
            observation: Observe the state of the environment. I.e. list of possible features to select

        Returns:

        """
        if not isinstance(observation, list):
            raise ValueError("observation should be a list. Got {}".format(type(observation)))

        which = np.random.choice(observation, replace=False)
        which_idx = observation.index(which)
        self.features.append(which)
        which = observation.pop(which_idx)
        return which, self.features

    def put_a_feature_back(self, possibilities):
        if not isinstance(possibilities, list):
            raise ValueError("possibilities should be a list. Got {}".format(type(possibilities)))
        which = np.random.choice(self.features, replace=False)
        possibilities.append(which)
        return which, possibilities

    def get_actions(self):
        return [self.pick_a_feature, self.put_a_feature_back]
