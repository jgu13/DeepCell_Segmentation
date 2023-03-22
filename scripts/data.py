import numpy as np
import os
from deepcell import image_generators
from deepcell.utils import train_utils
from sklearn.model_selection import train_test_split

def get_mini_samples(data_dir, sample_size=0.1, train_size=0.8, test_size=0.1):
    # get a smaller sample of the whole dataset for testing purposes
    # sample_size is the proportion we take as sample out of the whole dataset 
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    print("Sample_size={}".format(int(0.1*X.shape[0])))
    sample_inds = np.random.randint(low=0, high=X.shape[0], size=int(sample_size*X.shape[0]))
    sample_X = X[sample_inds,...,:]
    sample_y = y[sample_inds,...,:]
    X_train, X_test, y_train, y_test = train_test_split(sample_X, sample_y, test_size=1-train_size, random_state=42, shuffle=True) # 80% training data
    test_size = 1 / ((1-train_size) / test_size) # what portion I should take from the remaining dataset 
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state=42) 
    return X_train, X_val, X_test, y_train, y_val, y_test 

def prep_data(data_dir, train_size=0.8, test_size=0.1, prep_mini_samples=False):
    if prep_mini_samples:
        return get_mini_samples(data_dir, train_size=train_size, test_size=test_size)
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    # split train, val, test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42, shuffle=True) # 80% training data
    test_size = 1 / ((1-train_size) / test_size) # what portion I should take from the remaining dataset 
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state=42) 
    return X_train, X_val, X_test, y_train, y_val, y_test

def data_generator(classes, transformations, X_train, y_train, X_val, y_val, min_objects, batch_size):
    # classes(dict): number of classes for each semantic head
    # transformations(dict): transformations to be used for train and validation dataset generator
    # X_train, y_train(ndarray): training images and masks
    # X_val, y_val(ndarray): validation images and masks
    # min_objects(int): number of minimum objects to exist in an image
    # batch_size(int): number of images per batch 

    transforms = list(classes.keys())
    transforms_kwargs = {'outer-distance': {'erosion_width': 0}}
    seed = 42

    # use augmentation for training but not validation
    datagen = image_generators.SemanticDataGenerator(**transformations)

    datagen_val = image_generators.SemanticDataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=0,
        vertical_flip=0)
        
    train_data = datagen.flow(
        {'X': X_train, 'y': y_train},
        seed=seed,
        transforms=transforms,
        transforms_kwargs=transforms_kwargs,
        min_objects=min_objects,
        batch_size=batch_size)

    val_data = datagen_val.flow(
        {'X': X_val, 'y': y_val},
        seed=seed,
        transforms=transforms,
        transforms_kwargs=transforms_kwargs,
        min_objects=min_objects,
        batch_size=batch_size)
    
    return train_data, val_data