import os

WORKING_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
PICKLES_DIRECTORY = os.path.join(WORKING_DIRECTORY, 'pickles_directory')


if not os.path.isdir(PICKLES_DIRECTORY):
    os.makedirs(PICKLES_DIRECTORY)

SIMULATION_DATA_PICKLE = os.path.join(PICKLES_DIRECTORY, 'simulation_data.pickle')
EXTRACTED_FEATURE_DATA_PICKLE = os.path.join(PICKLES_DIRECTORY, 'extraced_features.pickle')


















