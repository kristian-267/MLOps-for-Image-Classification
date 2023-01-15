import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data

N_IMAGENET_MINI_TRAIN = 34745
N_IMAGENET_MINI_VAL = 3923
IMAGENET_MINI_SHAPE = [3, 224, 224]
N_IMAGENET_MINI_CLASS = 1000
