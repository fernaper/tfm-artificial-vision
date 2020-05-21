import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import requests
import tensorflow as tf

from datetime import datetime
from os import listdir
from os.path import join, isdir, sep
from tqdm import tqdm

from tfm_core import config


def get_labels(folder='dataset'):
    data_dir = pathlib.Path(join(config.DATA_PATH, folder))
    return np.array([item.name for item in data_dir.glob('*')])


def scale_images(input_folder='dataset', output_folder='dataset', width=64, height=64):
    input_dir = pathlib.Path(join(config.DATA_PATH, input_folder))
    image_count = len(list(input_dir.glob('*/*.jpg')))

    for image_path in tqdm(input_dir.glob('*/*.*'), total=image_count):
        image = cv2.imread(image_path.as_posix())
        image = cv2.resize(image, (width, height))
        class_name = image_path.parent.name

        output_dir = pathlib.Path(join(config.DATA_PATH, output_folder, class_name))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = join(output_dir.as_posix(),image_path.name)

        cv2.imwrite(output_file_path, image)


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

    return train_dataset, valid_dataset, steps_per_epoch, 3, 10


def own_dataset(folder='dataset',batch_size=32, image_height=64, image_width=32):
    data_dir = pathlib.Path(join(config.DATA_PATH, folder))
    image_count = len(list(data_dir.glob('*/*.jpg')))

    class_names = get_labels(folder)

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

    return train_generator, validation_generator, steps_per_epoch, validations_steps, len(class_names)


def show_batch(image_batch, label_batch, class_names):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(class_names[label_batch[n]==1][0].title())
        plt.axis('off')


def checkpoint_callback(model, model_name='resnet', period=5):
    checkpoint_path = join(config.CHECKPOINTS_PATH, model_name, 'cp-{epoch:04d}.ckpt')
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=period)

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
        version = max(max([int(folder) for folder in folders]) + 1, 1)

    model_path = join(model_path, str(version))

    tf.keras.models.save_model(
        model,
        model_path,
        #overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    print('\nSaved model: {}'.format(model_path))


def send_frame_serving_tf(frame, server='http://localhost:8501', model='resnet', width=64, height=64):
    #frame = cv2.resize(frame, (width, height))
    frame = frame / 255

    json_response = requests.post(
        '{server}/v1/models/{model}:predict'.format(server=server, model=model),
        json = {
            "instances": [frame.tolist()]
        }
    )

    response = json_response.json()
    print(response)

    if 'error' in response:
        return None

    predictions = np.array(response['predictions'][0])

    return predictions
