import tensorflow as tf

from os.path import join
from pathlib import Path

from tfm_core import config
from tfm_core.dnn.alexnet import alexnet
from tfm_core.dnn.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from tfm_core.dnn.residual_block import make_basic_block_layer, make_bottleneck_layer
from tfm_core.dnn.utilities import cifar10_dataset, own_dataset, checkpoint_callback, tensorboard_callback, save_model


NAME_TO_MODEL_BUILDER = {
    'alexnet': (alexnet, {'width': 227, 'height': 227}),
    'resnet_18': (resnet_18, {}),
    'resnet_34': (resnet_34, {}),
    'resnet50': (resnet_50, {}),
    'resnet_10': (resnet_101, {}),
    'resnet_152': (resnet_152, {}),
}


def train(model_name, epochs, batch_size, folder, test=False):
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Just in case model or scalars folders doesn't exist
    Path(join(config.MODELS_PATH, model_name)).mkdir(parents=True, exist_ok=True)
    Path(join(config.LOGS_PATH, 'scalars')).mkdir(parents=True, exist_ok=True)

    model_data = NAME_TO_MODEL_BUILDER.get(model_name)
    if model_data is None:
        print('Model <{}> not found'.format(model_name))
        return

    model_builder, kwargs = model_data

    loss = 'sparse_categorical_crossentropy'
    #loss = 'categorical_crossentropy'

    #optimizer = tf.keras.optimizers.Adadelta()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    batch_size = batch_size
    image_height = kwargs.get('height', 64)
    image_width = kwargs.get('width', 64)

    use_own_dataset = not test

    get_dataset = own_dataset if use_own_dataset else cifar10_dataset

    train_dataset, valid_dataset, steps_per_epoch, validation_steps, num_classes = get_dataset(
        folder=folder,
        batch_size=batch_size,
        image_height=image_height,
        image_width=image_width
    )

    model = model_builder(num_classes, **kwargs)
    model.build(input_shape=(None, image_height, image_width, 3))
    model.summary()

    callbacks = [
        checkpoint_callback(model, model_name=model_name),
        tensorboard_callback(model_name=model_name)
    ]

    model.compile(optimizer=optimizer,
        loss=loss,
        metrics=['acc']
    )

    model.fit(train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    save_model(model, model_name=model_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--folder', default='dataset',
        help='Dataset folder (default: dataset)')

    parser.add_argument('-m', '--model', default='resnet50',
        help='Model to train. Valid values <{}> (default: resnet50)'.format(
            '|'.join(NAME_TO_MODEL_BUILDER.keys())
        ))

    parser.add_argument('-t', '--test', default=False, action='store_true',
        help='Use test dataset (default False)')

    parser.add_argument('-e', '--epochs', type=int, default=5,
        help='Number of epochs to train (default: 5)')

    parser.add_argument('-b', '--batch', type=int, default=128,
        help='Batch size (default: 128)')

    args = parser.parse_args()

    train(args.model, args.epochs, args.batch, args.folder, test=args.test)