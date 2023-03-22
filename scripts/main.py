import argparse
from data import prep_data, data_generator
from model import build_model, add_loss_to_semantic_heads, train, detect
import json
from tensorflow.keras.optimizers import SGD, Adam
from deepcell.utils.train_utils import rate_scheduler
import os
import numpy as np
import tensorflow as tf
import time

def main(args):
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    print("Finished building file structures.")
    X_train, X_val, X_test, y_train, y_val, y_test = prep_data(args.data_dir, args.train_size, args.test_size, prep_mini_samples=True)
    model = build_model(args.classes, input_size=X_train.shape[1:])
    print("Finished building model.")
    optimizer = None
    if args.optimizer == "Adam":
        optimizer = Adam(learning_rate=args.lr, decay=args.decay, clipnorm=args.clip_norm)
    else:
        optimizer = SGD(learning_rate=args.lr, weight_decay=args.decay, clipnorm=args.clip_norm)
    lr_sched = rate_scheduler(lr=args.lr, decay=args.decay)    
    # default transformations 
    transformations = {
        "rotation_range": 180,
        "shear_range": 0,
        "zoom_range": (0.75, 1.25),
        "horizontal_flip": True,
        "vertical_flip": True
    }
    train_data, val_data = data_generator(args.classes, transformations, X_train, y_train, X_val, y_val, args.min_objects, args.batch_size)
    loss_methods = add_loss_to_semantic_heads(model)
    print("Training data size: {}".format(X_train.shape[0]))
    print("Test data size: {}".format(X_test.shape[0]))

    model.compile(loss=loss_methods, optimizer=optimizer)

    print("==================Training starts===================")
    start_time = time.time()
    training_loss = train(model, args.model_dir, args.log_dir, args.model_name, lr_sched, train_data, val_data, args.n_epoch, args.batch_size) 
    print(training_loss.history)
    # with open(os.path.join(args.log_dir, "{}.json".format(args.model_name)), 'w') as f:
    #     json.dump(["training losses: ", training_loss.history], f)
    duration = time.time() - start_time
    print("Training takes {}s".format(duration))
    print("===================Training ends=====================")
    if bool(args.detect):
        if not args.detect_classes:
            detect_classes = args.classes
        else:
            detect_classes = args.detect_classes
        outputs = detect(detect_classes, input_size=X_test[1:], model_dir=args.model_dir, X_test=X_test)
        output_save_path = os.path.join(args.results_dir, args.model_name)
        if not os.path.isdir(output_save_path):
            os.makedirs(output_save_path)
        np.save(os.path.join(output_save_path, "output.npy"), outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load arguments to train a Panoptic model.')
    parser.add_argument('-f', '--file', type=str, help='Path to JSON file')
    args = parser.parse_args()

    with open(args.file) as f:
        json_data = json.load(f)

    main(argparse.Namespace(**json_data))