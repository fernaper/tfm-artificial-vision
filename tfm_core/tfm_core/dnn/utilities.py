import tensorflow as tf

from datetime import datetime
from os.path import join

from tfm_core import config


def cifar10_dataset(batch_size=64):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(10000)
    train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    train_dataset = train_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y))
    train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    train_dataset = train_dataset.repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).shuffle(10000)
    valid_dataset = valid_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    valid_dataset = valid_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y))
    valid_dataset = valid_dataset.repeat()

    return train_dataset, valid_dataset


def checkpoint_callback(model, model_name='resnet'):
    checkpoint_path = join(config.CHECKPOINTS_PATH, model_name, 'cp-{epoch:04d}.ckpt')
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=5)

    model.save_weights(checkpoint_path.format(epoch=0))

    return callback


def tensorboard_callback(model_name='resnet'):
    logdir = join(config.SCALARS_PATH, model_name + '_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    return tf.keras.callbacks.TensorBoard(log_dir=logdir)
