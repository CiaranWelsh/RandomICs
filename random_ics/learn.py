from random_ics import *
from cluster import *
from simulate import *
from model_strings import model_string
import numpy as np
import pandas as pd

import rl


class Qlearn:

    def __init__(self, env, agent, gamma=0.5):
        self.gamma = gamma
        self.agent = agent
        self.env = env

    def epsilon_greedy(self, a, eps=0.1):
        r = np.random.uniform()
        if r < 1 - eps:
            return a
        else:
            return np.random.choice

    def learn(self):
        # policy evaluation

        reward = 0
        traj = {}
        for i in range(10):  # steps per episode
            self.env.possibilities = self.agent.pick_a_feature(self.env.possibilities)  # action
            reward += self.gamma * self.env.cluster_kmeans(self.agent)
            traj[i] = reward

        # updating policy


if __name__ == '__main__':
    data = TimeCourse(
        model_string, n=10,
        from_pickle=True, pickle_file=SIMULATION_DATA_PICKLE).simulate_random_ics()

    c = Cluster(data, from_pickle=True, pickle_file=EXTRACTED_FEATURE_DATA_PICKLE)
    a = Agent(features=list(c.data.columns[0:5:1000]))

    Qlearn(c, a)
