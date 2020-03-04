import pathlib
import tensorflow as tf
import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isdir, sep

from tfm_core import config


def cifar10_dataset(batch_size=64, **kwargs):
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

    steps_per_epoch = 50000//batch_size

    return train_dataset, valid_dataset, steps_per_epoch, 3


def own_dataset(batch_size=32, image_height=64, image_width=32):
    data_dir = pathlib.Path(join(config.DATA_PATH, 'dataset'))
    image_count = len(list(data_dir.glob('*/*.jpg')))

    class_names = np.array([item.name for item in data_dir.glob('*')])

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = image_generator.flow_from_directory(
        str(data_dir),
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = image_generator.flow_from_directory(
        str(data_dir),
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    steps_per_epoch = train_generator.samples // batch_size
    validations_steps = validation_generator.samples // batch_size

    '''
    train_data_gen = image_generator.flow_from_directory(
        directory=str(data_dir),
        batch_size=batch_size,
        shuffle=True,
        target_size=(image_height, image_width),
        classes = list(class_names)
    )'''

    return train_generator, validation_generator, steps_per_epoch, validations_steps


def show_batch(image_batch, label_batch, class_names):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(class_names[label_batch[n]==1][0].title())
        plt.axis('off')


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


def save_model(model, model_name='resnet'):
    model_path = join(config.MODELS_PATH, model_name)

    folders = [name for name in listdir(model_path) if isdir(join(model_path,name))]
    version = 1

    if folders:
        # Get the last version and create the next one
        version = min(max([int(folder) for folder in folders]) + 1, 1)

    model_path = join(model_path, str(version))

    tf.keras.models.save_model(
        model,
        model_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    print('\nSaved model: {}'.format(model_path))

'''
tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=fashion_model \
  --model_base_path="/mnt/60BA93F4BA93C546/CommonDocuments/GitHub/tfm-artificial-vision/models/resnet" >server.log 2>&1
'''