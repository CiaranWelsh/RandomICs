from random_ics import *
from cluster import *
from simulate import *
from model_strings import model_string

from deap import base, creator

creator.create("FitnessMin", base.Fitness, weights=(-1,))

# print(FitnessMin)

data = TimeCourse(
    model_string, n=30,
    from_pickle=False, pickle_file=SIMULATION_DATA_PICKLE,
    subtract_ic_normalisation=True
).simulate_random_ics()

from sklearn.cluster import DBSCAN
c = Cluster(
    data, from_pickle=False, n_clusters=4,

)

a = Agent(features=[])#list(c.data.columns[0:5:1000]))

current_best = None
done = False
i = 0
# for i in range(1000):
while not done:
    pick_or_put = np.random.uniform()
    if pick_or_put > 0.1:
        print('picking ... ')
        action = a.get_actions()[0]
    else:
        print('returning ... ')
        action = a.get_actions()[1]
    from copy import deepcopy
    original_features = deepcopy(a.features)
    original_possibilities = deepcopy(c.possibilities)
    # updates a.features with chosen feature and removes that feature from c.possibilities
    which, _ = action(c.possibilities)
    score = round(c.cluster(a), 6)
    if current_best is None or score > current_best:
        print('assigning new best score: score: {}; current best: {}. {}'.format(score, current_best, score > current_best if current_best is not None else None))
        current_best = score
    else:
        # do not except change
        print('Not accepting change: score: {}; current best: {}. {}'.format(score, current_best, score > current_best))
        a.features = original_features
        c.possibilities = original_possibilities

    i += 1
    print('selection: {}'.format(which))
    print('Numbr of features held by agent: {}'.format(len(a.features)))
    print('agent features: {}'.format(a.features))
    print('number of observations left: {}'.format(len(c.possibilities)))
    print('score for run {}: {}\n'.format(i, score))

    if score > 0.95:# and all(c.features != -1):
        done = True

    """
    picking ... 
    assigning new best score: score: 0.859574; current best: 0.793626. True
    selection: 4899
    Numbr of features held by agent: 18
    agent features: [862, 1293, 1246, 5466, 22, 2877, 5879, 5608, 5411, 5878, 5139, 4976, 5367, 5386, 5876, 5468, 4975, 4899]
    number of observations left: 9200
    score for run 9341: 0.859574
    """















