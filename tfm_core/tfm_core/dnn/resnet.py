'''
    El crédito de las clases ResNetTypeI y ResNetTypeII pertenece a calmisential
    Se encuentra público bajo licencia MIT en: https://github.com/calmisential/TensorFlow2.0_ResNet
'''

import tensorflow as tf

from tfm_core.dnn.residual_block import make_basic_block_layer, make_bottleneck_layer
from tfm_core.dnn.utilities import cifar10_dataset, own_dataset, checkpoint_callback, tensorboard_callback, save_model


class ResNetTypeI(tf.keras.Model):

    def __init__(self, layer_params, num_classes):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])

        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)

        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)

        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

        self.fc = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)


    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


class ResNetTypeII(tf.keras.Model):

    def __init__(self, layer_params, num_classes):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])

        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)

        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)

        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)


    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


def resnet_18(num_classes):
    return ResNetTypeI(layer_params=[2, 2, 2, 2], num_classes=num_classes)


def resnet_34(num_classes):
    return ResNetTypeI(layer_params=[3, 4, 6, 3], num_classes=num_classes)


def resnet_50(num_classes):
    return ResNetTypeII(layer_params=[3, 4, 6, 3], num_classes=num_classes)


def resnet_101(num_classes):
    return ResNetTypeII(layer_params=[3, 4, 23, 3], num_classes=num_classes)


def resnet_152(num_classes):
    return ResNetTypeII(layer_params=[3, 8, 36, 3], num_classes=num_classes)


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

    parser.add_argument('-m', '--model', type=int, default=50,
        help='Model type 18|34|50|101|152 (default: 50)')

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
    image_height = 64
    image_width = 64

    use_own_dataset = not args.test

    get_dataset = own_dataset if use_own_dataset else cifar10_dataset

    train_dataset, valid_dataset, steps_per_epoch, validation_steps, num_classes = get_dataset(
        folder=args.folder,
        batch_size=batch_size,
        image_height=image_height,
        image_width=image_width
    )

    models = {
        18: resnet_18,
        34: resnet_34,
        50: resnet_50,
        101: resnet_101,
        152: resnet_152
    }

    model = models[args.model](num_classes=num_classes)
    model.build(input_shape=(None, image_height, image_width, 3))
    model.summary()

    callbacks = [
        checkpoint_callback(model, model_name='resnet'),
        tensorboard_callback(model_name='resnet')
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

    save_model(model, model_name='resnet-{}'.format(args.model) if args.model != 50 else 'resnet')
