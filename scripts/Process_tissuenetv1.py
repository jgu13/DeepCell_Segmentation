import os
import numpy as np
import pandas as pd

tissuenet_train_path = "/home/claris/projects/def-watsoni2/claris/python_projects/deepcell-tf/tissuenet_v1.1_test.npz"
tissuenet_train = np.load(tissuenet_train_path)
X = tissuenet_train['X'][:,...,0][...,np.newaxis]
y = tissuenet_train['y'][:,...,0][...,np.newaxis]
sample_inds = np.random.randint(low=0, high=X.shape[0], size=int(0.1*X.shape[0]))
sample_X = X[sample_inds,...,:]
sample_y = y[sample_inds,...,:]
np.save("/home/claris/projects/def-watsoni2/claris/python_projects/deepcell-tf/data/minitest/X.npy", sample_X)
np.save("/home/claris/projects/def-watsoni2/claris/python_projects/deepcell-tf/data/minitest/y.npy", sample_y)