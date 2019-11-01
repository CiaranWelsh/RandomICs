import pandas as pd
import numpy as np
from pyDOE import lhs
from scipy.stats.distributions import uniform
import matplotlib.pyplot as plt
import seaborn as sns
import tellurium as te

from random_ics import *


class TimeCourse:

    def __init__(self, ant_str, n=10, lower_bound=0.1, upper_bound=10,
                 end_time=100, num_simulation_points=101, from_pickle=False,
                 pickle_file=SIMULATION_DATA_PICKLE,
                 subtract_ic_normalisation=False):
        self.subtract_ic_normalisation = subtract_ic_normalisation
        self.from_pickle = from_pickle
        self.pickle_file = pickle_file
        self.end_time = end_time
        self.num_simulation_points = num_simulation_points
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.n = n

        self.rr = te.loada(ant_str)

    def simulate_random_ics(self):
        """
        Randomize initial concentration parameters using latin hypercube sampling
        and run a time course

        """
        if self.from_pickle and os.path.isfile(self.pickle_file):
            return pd.read_pickle(self.pickle_file)

        ics = [i.replace('[', '').replace(']', '') for i in self.rr.getFloatingSpeciesConcentrationIds()]

        original_ics = dict(zip(ics, self.rr.getFloatingSpeciesConcentrations()))
        sample = lhs(n=len(original_ics), samples=self.n, iterations=1, criterion=None)
        sample = uniform(self.lower_bound, self.upper_bound).ppf(sample)

        print('Simulating time series data')
        simulations = {}
        for i in range(sample.shape[0]):
            print('Percent Complete: {}%'.format(round(i / sample.shape[0] * 100, 2)))
            self.rr.reset()
            for j in range(sample.shape[1]):
                setattr(self.rr, ics[j], sample[i, j])
            data = self.rr.simulate(0, self.end_time, self.num_simulation_points)
            df = pd.DataFrame(data)
            df.columns = [i.replace('[', '').replace(']', '') for i in data.colnames]
            simulations[i] = df.set_index('time')

        df = pd.concat(simulations)

        df.to_pickle(self.pickle_file)

        if self.subtract_ic_normalisation:
            df = self.normalise(df)
        return df

    def normalise(self, df):
        dct = {}
        for label, df2 in df.groupby(level=0):
            dct[label] = df2.subtract(df2.iloc[0])
        df = pd.concat(dct)
        df.index = df.index.droplevel(0)
        df.to_pickle(self.pickle_file)
        return df


    @staticmethod
    def plot1(df, hspace=0.5, wspace=0.3, ncols=5, filename=None, **kwargs):
        print('plotting time series data')
        nplots = df.shape[1]
        if nplots == 1:
            ncols = 1
        nrows = int(nplots / ncols)
        remainder = nplots % ncols
        if remainder > 0:
            nrows += 1

        fig = plt.figure(figsize=(20, 10))
        for i, species in enumerate(df.columns):
            plot_data = df[[species]].reset_index()
            plot_data.columns = ['iterations', 'time', species]
            # print(plot_data)
            ax = plt.subplot(nrows, ncols, i + 1)
            sns.lineplot(
                x='time', y=species, hue='iterations',
                data=plot_data, ax=ax, **kwargs, legend=False,
                palette='Blues',
            )

            sns.despine(ax=ax, top=True, right=True)
            plt.title(species)
            plt.xlabel('')
            plt.ylabel('')
        plt.subplots_adjust(hspace=hspace, wspace=wspace)

        if filename is None:
            plt.show()
        else:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
        return df

    @staticmethod
    def plot2(df, labels, hspace=0.5, wspace=0.3, ncols=5, filename=None, **kwargs):
        nplots = df.shape[1]
        if nplots == 1:
            ncols = 1
        nrows = int(nplots / ncols)
        remainder = nplots % ncols
        if remainder > 0:
            nrows += 1

        ncolours = len(list(set(labels)))
        colours = sns.color_palette('hls', ncolours)

        labels_lookup = {i: labels[i] for i in range(len(labels))}

        fig = plt.figure(figsize=(20, 10))
        for i, species in enumerate(df.columns):
            plot_data = df[[species]].reset_index()
            plot_data.columns = ['iterations', 'time', species]
            plot_data['class'] = [labels_lookup[i] for i in plot_data['iterations']]
            ax = plt.subplot(nrows, ncols, i + 1)
            sns.lineplot(
                x='time', y=species, hue='class',
                data=plot_data, ax=ax, **kwargs, legend=False,
                palette=colours,

            )

            sns.despine(ax=ax, top=True, right=True)
            plt.title(species)
            plt.xlabel('')
            plt.ylabel('')
        plt.legend(loc=(0.1, 1), bbox_to_anchor=(1, 0.1))
        plt.subplots_adjust(hspace=hspace, wspace=wspace)

        if filename is None:
            plt.show()
        else:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
        return df
