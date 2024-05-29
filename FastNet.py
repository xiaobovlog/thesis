from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Add, Input, BatchNormalization, Dense, concatenate, GlobalAveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, ReLU
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from config import TARGET_SIZE, BATCH_SIZE, EPOCH, TRAIN_DIR, INPUT_SHAPE
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils



# extract mutil-scale feature
def multiscale_block(inputs, use_shortcut=False, expansion=1, branch_channel=4):
    """ expansion is the factor to expand the input dimensions, branch_channel is the output dimensions in each branch """
    # expand the dimensions
    x = Conv2D(filters=expansion*inputs.shape[3],
               kernel_size=(1, 1),
               padding='same',
               use_bias=False,
               activation=None)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    # 1×1
    branch1 = SeparableConv2D(filters=branch_channel,
                              kernel_size=(1, 1),
                              padding='same',
                              use_bias=False,
                              activation=None)(x)
    branch1 = BatchNormalization()(branch1)
    branch1 = ReLU(6.)(branch1)

    # 3×3
    branch2 = SeparableConv2D(filters=branch_channel,
                              kernel_size=(3, 3),
                              padding='same',
                              use_bias=False,
                              activation=None)(x)
    branch2 = BatchNormalization()(branch2)
    branch2 = ReLU(6.0)(branch2)

    # 5×5
    branch3 = SeparableConv2D(filters=branch_channel,
                              kernel_size=(3, 3),
                              padding='same',
                              use_bias=False,
                              activation=None)(x)
    branch3 = BatchNormalization()(branch3)
    branch3 = ReLU(6.0)(branch3)

    branch3 = SeparableConv2D(filters=branch_channel,
                              kernel_size=(3, 3),
                              padding='same',
                              use_bias=False,
                              activation=None)(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = ReLU(6.0)(branch3)


    x = concatenate([branch1, branch2, branch3])

    # reduce dimension for shortcut
    x = Conv2D(filters=inputs.shape[3],
               kernel_size=(1, 1),
               padding='same',
               use_bias=False,
               activation=None)(x)
    # use or not use shortcut branch
    if use_shortcut:
        x = Add()([inputs, x])
    return x


# increase width of the network
def multibranch_block(inputs, use_shortcut=False, expansion=1, branch_number=2):
    x = Conv2D(filters=expansion*inputs.shape[3],
               kernel_size=(1, 1),
               padding='same',
               use_bias=False,
               activation=None)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    branchs = []
    for i in range(branch_number):
        branch = SeparableConv2D(filters=expansion*inputs.shape[3],
                                 kernel_size=(3, 3),
                                 padding='same',
                                 use_bias=False,
                                 activation=None)(x)
        branch = BatchNormalization()(branch)
        branch = ReLU(6.0)(branch)
        branchs.append(branch)

    x = concatenate(branchs)

    # reduce dimension for shortcut
    x = Conv2D(filters=inputs.shape[3],
               kernel_size=(1, 1),
               padding='same',
               use_bias=False,
               activation=None)(x)
    # use or not use shortcut branch
    if use_shortcut:
        x = Add()([inputs, x])
    return x


def FastNet(include_top=True, weights=None, input_shape=None, use_shortcut=False, classes=1000):
    # initialize the input_shape
    inputs = Input(shape=input_shape)

    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               strides=(2, 2),
               padding='valid',
               activation='relu',
               use_bias=False)(inputs)
    x = BatchNormalization()(x)


    x = Conv2D(filters=8,
               kernel_size=(3, 3),
               strides=(2, 2),
               padding='valid',
               activation='relu',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    # multiscale feature extract
    x = multiscale_block_dw(x, use_shortcut=use_shortcut, expansion=1, branch_channel=4)
    x = multibranch_block_dw(x, use_shortcut=use_shortcut, expansion=1, branch_number=2)


    x = Conv2D(filters=16,
               kernel_size=(3, 3),
               strides=(2, 2),
               padding='valid',
               activation='relu',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = multiscale_block_dw(x, use_shortcut=use_shortcut, expansion=2, branch_channel=8)
    x = multibranch_block_dw(x, use_shortcut=use_shortcut, expansion=2, branch_number=4)


    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               strides=(2, 2),
               padding='valid',
               activation='relu',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = multiscale_block_dw(x, use_shortcut=use_shortcut, expansion=4, branch_channel=16)
    x = multibranch_block_dw(x, use_shortcut=use_shortcut, expansion=4, branch_number=8)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model

