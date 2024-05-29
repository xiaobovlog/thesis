def msbe_crosslinknet(n_classes, input_height=416, input_width=608, channels=3):
    model = _crosslinknet(n_classes, get_msbe_encoder, input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "msbe_crosslinknet"
    return model

    
# crosslinknet decoder
def _crosslinknet(n_classes, encoder, l1_skip_conn=True, input_height=416, input_width=608, channels=3):
    img_input, levels = encoder(input_height=input_height, input_width=input_width, channels=channels)
    [layer0, layer1, layer2, layer3, layer4] = levels
    # decoder
    up_conv31 = Conv2D(filters=512, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(layer4))
    up_conv32 = Conv2D(filters=512, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(layer4))
    up_conv3 = concatenate([up_conv31, up_conv32, layer3], axis=-1)
    up_conv3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_conv3)
    up_conv3 = BatchNormalization()(up_conv3)
    up_conv3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_conv3)
    up_conv3 = BatchNormalization()(up_conv3)

    up_conv21 = Conv2D(filters=256, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(up_conv3))
    up_conv22 = Conv2D(filters=256, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(layer3))
    up_conv2 = concatenate([up_conv21, up_conv22, layer2], axis=-1)
    up_conv2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_conv2)
    up_conv2 = BatchNormalization()(up_conv2)
    up_conv2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_conv2)
    up_conv2 = BatchNormalization()(up_conv2)

    up_conv11 = Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(up_conv2))
    up_conv12 = Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(layer2))
    up_conv1 = concatenate([up_conv11, up_conv12, layer1], axis=-1)
    up_conv1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_conv1)
    up_conv1 = BatchNormalization()(up_conv1)
    up_conv1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_conv1)
    up_conv1 = BatchNormalization()(up_conv1)

    up_conv01 = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(up_conv1))
    up_conv02 = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(layer1))
    up_conv0 = concatenate([up_conv01, up_conv02, layer0], axis=-1)
    up_conv0 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_conv0)
    up_conv0 = BatchNormalization()(up_conv0)
    up_conv0 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_conv0)
    up_conv0 = BatchNormalization()(up_conv0)
    
    o = Conv2D(filters=n_classes, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(up_conv0)
    model = get_segmentation_model(img_input, o)
    return model
   

def get_msbe_encoder(input_height=224,  input_width=224, channels=3, alpha=1.0, depth_multiplier=1):
    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(channels, input_height, input_width))
        channel_axis = 1
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, channels))
        channel_axis = -1
    levels = []
    x = img_input
    levels.append(x)

    x = Conv2D(
        32, (1, 1),
        padding='same',
        use_bias=False,
        strides=(2, 2),
        name='conv_%d' % 0)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_%d_bn' % 0)(
        x)
    x = ReLU(6., name='conv_%d_relu' % 0)(x)
    x = msbe_block(x, 64, alpha, depth_multiplier, block_id=0, use_shortcut=False, mode=0)
    levels.append(x)

    shortcut = False
    x = msbe_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=1, use_shortcut=False)
    x = msbe_block(x, 128, alpha, depth_multiplier, block_id=2, use_shortcut=shortcut, mode=2)
    levels.append(x)

    x = msbe_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=3, use_shortcut=False)
    x = msbe_block(x, 256, alpha, depth_multiplier, block_id=4, use_shortcut=shortcut, mode=2)
    levels.append(x)

    x = msbe_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=5, use_shortcut=False)
    x = msbe_block(x, 512, alpha, depth_multiplier, block_id=6, use_shortcut=shortcut, mode=0)
    levels.append(x)

    return img_input, levels

  
   
def get_segmentation_model(input, output):

    img_input = input
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height = o_shape[2]
        output_width = o_shape[3]
        input_height = i_shape[2]
        input_width = i_shape[3]
        n_classes = o_shape[1]
        o = (Reshape((-1, output_height*output_width)))(o)
        o = (Permute((2, 1)))(o)
    elif IMAGE_ORDERING == 'channels_last':
        output_height = o_shape[1]
        output_width = o_shape[2]
        input_height = i_shape[1]
        input_width = i_shape[2]
        n_classes = o_shape[3]
        o = (Reshape((output_height*output_width, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""

    model.train = MethodType(train, model)
    model.predict_segmentation = MethodType(predict, model)
    model.predict_multiple = MethodType(predict_multiple, model)
    model.evaluate_segmentation = MethodType(evaluate, model)

    return model

