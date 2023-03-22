from deepcell.model_zoo.panopticnet import PanopticNet
from tensorflow.keras.losses import MSE
from deepcell import losses
from deepcell.utils.train_utils import get_callbacks
from deepcell.utils.train_utils import count_gpus
import os


def build_model(classes, input_size, backbone='resnet50', norm_method='std', location=True, include_top=False):
    # Location(bool): whether to include the Location.2DLayer
    # include_top(bool): whether to include the top layer 
    model = PanopticNet(
        backbone=backbone,
        input_shape=input_size,
        norm_method=norm_method,
        num_semantic_classes=classes,
        location=location,
        include_top=include_top)
    return model


# Create a dictionary of losses for each semantic head
def semantic_loss(n_classes):
    def _semantic_loss(y_true, y_pred):
        if n_classes > 1:
            return 0.01 * losses.weighted_categorical_crossentropy(
                y_true, y_pred, n_classes=n_classes)
        return MSE(y_true, y_pred)
    return _semantic_loss

def add_loss_to_semantic_heads(model):
    loss = {}
    # Give losses for all of the semantic heads
    for layer in model.layers:
        if layer.name.startswith('semantic_'):
            n_classes = layer.output_shape[-1]
            loss[layer.name] = semantic_loss(n_classes) # each layer has an independent loss
    return loss

def train(model, model_dir, log_dir, model_name, lr_sched, train_data, val_data, n_epoch=10, batch_size=1):

    model_path = os.path.join(model_dir, '{}.h5'.format(model_name))
    loss_path = os.path.join(model_dir, '{}.npz'.format(model_name))

    num_gpus = count_gpus()

    print('Training on', num_gpus, 'GPUs.')

    train_callbacks = get_callbacks(
        model_path,
        lr_sched=lr_sched,
        tensorboard_log_dir=log_dir,
        save_weights_only=num_gpus >= 2,
        monitor='val_loss',
        verbose=1)

    loss_history = model.fit(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=train_callbacks)
    
    return loss_history

def detect(classes, input_size, model_dir, X_test):
    model = build_model(
            input_size=input_size,
            num_semantic_heads=len(classes),
            num_semantic_classes=classes,
            include_top=True) # include the top prediction layer
    model.load_weights(model_dir, by_name=True)
    outputs = model.predict(X_test)
    return outputs