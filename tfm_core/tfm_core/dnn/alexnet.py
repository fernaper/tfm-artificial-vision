import tensorflow as tf

from tfm_core.dnn.residual_block import make_basic_block_layer, make_bottleneck_layer
from tfm_core.dnn.utilities import cifar10_dataset, own_dataset, checkpoint_callback, tensorboard_callback, save_model

class AlexNet(tf.keras.models.Sequential):

    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.add(tf.keras.layers.Conv2D(64, kernel_size=(11,11), strides= 4,
                        padding= 'valid', activation= 'relu',
                        input_shape=input_shape,
                        kernel_initializer= 'he_normal'))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                        padding= 'valid', data_format= None))

        self.add(tf.keras.layers.Conv2D(192, kernel_size=(5,5), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                        padding= 'valid', data_format= None)) 

        self.add(tf.keras.layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        self.add(tf.keras.layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        self.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        self.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                        padding= 'valid', data_format= None))

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(4096, activation= 'relu'))
        self.add(tf.keras.layers.Dense(4096, activation= 'relu'))
        self.add(tf.keras.layers.Dense(1000, activation= 'relu'))
        self.add(tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax))


def alexnet(num_classes, width=64, height=64):
    return AlexNet((width, height, 3), num_classes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--folder', default='dataset',
        help='Dataset folder (default: dataset)')

    parser.add_argument('-t', '--test', default=False, action='store_true',
        help='Use test dataset (default False)')

    parser.add_argument('-e', '--epochs', type=int, default=5,
        help='Number of epochs to train (default: 5)')

    parser.add_argument('-b', '--batch', type=int, default=128,
        help='Batch size (default: 128)')

    args = parser.parse_args()

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    loss = 'sparse_categorical_crossentropy'
    #loss = 'categorical_crossentropy'

    #optimizer = tf.keras.optimizers.Adadelta()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    batch_size = args.batch
    image_height = 227
    image_width = 227

    use_own_dataset = not args.test

    get_dataset = own_dataset if use_own_dataset else cifar10_dataset

    train_dataset, valid_dataset, steps_per_epoch, validation_steps, num_classes = get_dataset(
        folder=args.folder,
        batch_size=batch_size,
        image_height=image_height,
        image_width=image_width
    )

    model = alexnet(num_classes=num_classes, height=image_height, width=image_width)

    model.build(input_shape=(None, image_height, image_width, 3))
    model.summary()

    callbacks = [
        checkpoint_callback(model, model_name='alexnet'),
        tensorboard_callback(model_name='alexnet')
    ]

    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['acc']
    )

    model.fit(train_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    save_model(model, model_name='alexnet')