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
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    loss = 'sparse_categorical_crossentropy'
    #loss = 'categorical_crossentropy'

    #optimizer = tf.keras.optimizers.Adadelta()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)

    batch_size = 32 # 128
    image_height = 64
    image_width = 32
    use_own_dataset = True

    get_dataset = own_dataset if use_own_dataset else cifar10_dataset

    train_dataset, valid_dataset, steps_per_epoch, validation_steps = get_dataset(
        batch_size=batch_size,
        image_height=image_height,
        image_width=image_width
    )

    model = resnet_50(num_classes=10)
    model.build(input_shape=(None, image_height, image_width, 3))
    model.summary()

    callbacks = [
        checkpoint_callback(model, model_name='resnet'),
        tensorboard_callback(model_name='resnet')
    ]

    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['acc'])

    model.fit(train_dataset,
        epochs=30,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    save_model(model, model_name='resnet')
