import os
import numpy as np
import pandas as pd

tissuenet_train_path = "/home/claris/projects/def-watsoni2/claris/python_projects/deepcell-tf/tissuenet_v1.1_test.npz"
tissuenet_train = np.load(tissuenet_train_path)
X = tissuenet_train['X'][:,...,0][...,np.newaxis]
y = tissuenet_train['y'][:,...,0][...,np.newaxis]
np.save("/home/claris/projects/def-watsoni2/claris/python_projects/deepcell-tf/data/tissuenet/X.npy", X)
np.save("/home/claris/projects/def-watsoni2/claris/python_projects/deepcell-tf/data/tissuenet/y.npy", y)