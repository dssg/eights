import os
import sys

TESTS_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
DATA_PATH = os.path.join(TESTS_PATH, 'data')
EIGHTS_PATH = os.path.join(TESTS_PATH, '..')

def path_of_data(filename):
    return os.path.join(DATA_PATH, filename)

def add_to_python_path():
    sys.path.append(EIGHTS_PATH)
