import unittest

from .simulate import *
from .cluster import *
from .model_strings import model_string


class TimCourseTests(unittest.TestCase):

    def test_correct_number_of_simulation_points(self):
        tc = TimeCourse(model_string, n=10, lower_bound=0.1, upper_bound=100,
                        end_time=100, num_simulation_points=10)
        data = tc.simulate_random_ics()
        expected = 10 * 10
        self.assertEqual(expected, data.shape[0])

    def test_plot(self):
        fname1 = os.path.join(PICKLES_DIRECTORY, 'random_ts_no_norm.png')
        fname2 = os.path.join(PICKLES_DIRECTORY, 'random_ts_norm.png')
        tc = TimeCourse(model_string, n=10, lower_bound=0.1, upper_bound=10,
                        end_time=50, num_simulation_points=50,
                        )
        data = tc.simulate_random_ics()
        norm_data = tc.normalise(data)
        # TimeCourse.plot1(data, filename=fname1)
        # TimeCourse.plot1(norm_data, filename=fname2)


class ClusterTests(unittest.TestCase):

    def setUp(self) -> None:
        self.data = TimeCourse(
            model_string, n=10,
            from_pickle=True, pickle_file=SIMULATION_DATA_PICKLE).simulate_random_ics()

    def test_feature_extraction(self):
        c = Cluster(self.data, from_pickle=True, pickle_file=EXTRACTED_FEATURE_DATA_PICKLE)
        expected = (10, 9205)
        self.assertEqual(expected, c.data.shape)

    def test_cluster(self):
        c = Cluster(self.data, from_pickle=True)
        a = Agent(features=list(c.data.columns[0:5:1000]))
        expected = 0.5370586007564173
        actual = c.cluster_kmeans(a)
        self.assertEqual(expected, actual)

    def test_agent_action1(self):
        c = Cluster(self.data, from_pickle=True)
        a = Agent(features=list(c.data.columns[0:5:1000]))
        expected = len(a) + 1  # should gain 1 feature
        c.possibilities = a.pick_a_feature(c.possibilities)
        self.assertEqual(expected, len(a.features))

    def test_agent_action2(self):
        c = Cluster(self.data, from_pickle=True)
        a = Agent(features=list(c.data.columns[0:5:1000]))
        expected = len(c.possibilities) - 1  # should loose a feature
        a.pick_a_feature(c.possibilities)
        self.assertEqual(expected, len(c.possibilities))

    def test_result(self):
        c = Cluster(self.data, n_clusters=4, n_init=10, from_pickle=True)
        f = [862, 1293, 1246, 5466, 22, 2877, 5879, 5608, 5411, 5878, 5139, 4976, 5367, 5386, 5876, 5468, 4975, 4899]
        a = Agent(features=f)
        score = c.cluster(a)
        print(score, c.get_feature_names(f))
        labels = c.algorithm.labels_
        fname = os.path.join(PICKLES_DIRECTORY, 'example1.png')
        TimeCourse.plot2(self.data, labels, filename=fname)

    def test_result2(self):
        c = Cluster(self.data, n_clusters=4, n_init=10, from_pickle=True)
        f = [8434, 8735, 6626, 6610, 5891, 6766, 6746]
        a = Agent(features=f)
        score = c.cluster(a)
        print(score, c.get_feature_names(f))
        labels = c.algorithm.labels_
        fname = os.path.join(PICKLES_DIRECTORY, 'example2.png')
        TimeCourse.plot2(self.data, labels, filename=fname)

    def test_result3(self):
        c = Cluster(self.data, n_clusters=4, n_init=10, from_pickle=True)
        f = [954, 7205, 273, 7201, 7304, 792, 868, 7237, 369, 7209, 373, 7229, 7245, 7233, 824, 836, 872, 7217, 888, 7213, 860, 7225, 864, 7193, 385, 7221, 377]
        a = Agent(features=f)
        score = c.cluster(a)
        print(score, c.get_feature_names(f))
        labels = c.algorithm.labels_
        fname = os.path.join(PICKLES_DIRECTORY, 'out.png')
        TimeCourse.plot2(self.data, labels, filename=fname)

    def test_result5(self):
        tc = TimeCourse(
            model_string, n=10, subtract_ic_normalisation=True,
            from_pickle=True, pickle_file=SIMULATION_DATA_PICKLE)
        data = tc.simulate_random_ics()

        c = Cluster(data, n_clusters=4, n_init=10, from_pickle=True)
        f = [7115, 8568, 352, 5766, 5786, 821, 881, 869, 889, 380, 384, 8342, 873, 5794, 8564, 5774, 376, 877, 396,
             5778, 8584, 388, 5770, 5782, 8576]
        a = Agent(features=f)
        score = c.cluster(a)
        print(score, c.get_feature_names(f))
        labels = c.algorithm.labels_
        fname = os.path.join(PICKLES_DIRECTORY, 'out.png')
        TimeCourse.plot2(self.data, labels, filename=fname)

    def test_result4(self):
        c = Cluster(self.data, n_clusters=4, n_init=10, from_pickle=False)
        # a = Agent(features=f)
        # score = c.cluster(a)
        # print(score, c.get_feature_names(f))
        # labels = c.algorithm.labels_
        # fname = os.path.join(PICKLES_DIRECTORY, 'out.png')
        # TimeCourse.plot2(self.data, labels, filename=fname)


if __name__ == '__main__':
    unittest.main()
