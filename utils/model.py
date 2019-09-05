import tensorflow as tf

def model_inputs(image_size):
    '''
    Defines CNN inputs (placeholders).

    :param image_size: tuple, (height, width) of an image
    '''
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], 3], name='images')
    targets = tf.placeholder(dtype=tf.int32, shape=[None], name='targets')
    dropout_rate = tf.placeholder(dtype=tf.float32, name='dropout_rate')
    return inputs, targets, dropout_rate

def conv_block(inputs,
               number_of_filters,
               kernel_size, strides=(1, 1),
               padding='SAME',
               activation=tf.nn.relu,
               max_pool=True,
               batch_norm=True):
    '''
    Defines convolutional block layer.

    :param inputs: data from previous layer
    :param number_of_filters: integer, number of conv filters
    :param kernel_size: tuple, size of conv layer kernel
    :param padding: string, type of padding technique: SAME or VALID
    :param activation: tf.object, activation function used on the layer
    :param max_pool: boolean, if true the conv block will use max_pool
    :param batch_norm: boolean, if true the conv block will use batch normalization
    '''

    conv_features = layer = tf.layers.Conv2D(filters=number_of_filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding=padding,
                                             activation=activation)(inputs)
    if max_pool:
        layer = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(layer)
    if batch_norm:
        layer = tf.layers.BatchNormalization()(layer)

    return layer, conv_features

def dense_block(inputs,
                units,
                activation=tf.nn.relu,
                dropout_rate=None,
                batch_norm=True):
    '''
    Defines dense block layer.

    :param inputs: data from the previous layer
    :param units: integer, number of neurons/units for a dense layer
    :param activation: tf.object, activation function used on the layer
    :param dropout_rate: dropout rate used in this dense block
    :param batch_norm: boolean, if true the dense block will use batch normalization
    '''

    dense_features = layer = tf.layers.Dense(units=units, activation=activation)(inputs)
    if dropout_rate is not None:
        layer = tf.layers.Dropout(rate=dropout_rate)(layer)
    if batch_norm:
        layer = tf.layers.BatchNormalization()(layer)

    return layer, dense_features

def opt_loss(logits,
             targets,
             learning_rate):
    '''
    Defines model's optimizer and loss functions.

    :param logits: pre-activated model outputs
    :param targets: true labels for each input sample
    :param learning_rate: learning rate
    '''
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return loss, optimizer

class ImageSearchModel(object):

    def __init__(self,
                 learning_rate,
                 image_size,
                 number_of_classes=10):
        '''
        Defines CNN model.

        :param learning_rate: learning_rate
        :param image_size: tuple, (height, width) of an image
        :param number_of_classes: integer, number of classes in dataset
        '''

        tf.reset_default_graph()

        self.inputs, self.targets, self.dropout_rate = model_inputs(image_size)

        normalized_images = tf.layers.BatchNormalization()(self.inputs)

        #conv block 1
        conv_block_1, self.conv_1_features = conv_block(inputs=normalized_images,
                                                        number_of_filters=64,
                                                        kernel_size=(3, 3),
                                                        strides=(1, 1),
                                                        padding='SAME',
                                                        activation=tf.nn.relu,
                                                        max_pool=True,
                                                        batch_norm=True)

        #conv block 2
        conv_block_2, self.conv_2_features = conv_block(inputs=conv_block_1,
                                                        number_of_filters=128,
                                                        kernel_size=(3, 3),
                                                        strides=(1, 1),
                                                        padding='SAME',
                                                        activation=tf.nn.relu,
                                                        max_pool=True,
                                                        batch_norm=True)

        #conv block 3
        conv_block_3, self.conv_3_features = conv_block(inputs=conv_block_2,
                                                        number_of_filters=256,
                                                        kernel_size=(5, 5),
                                                        strides=(1, 1),
                                                        padding='SAME',
                                                        activation=tf.nn.relu,
                                                        max_pool=True,
                                                        batch_norm=True)

        #conv block 4
        conv_block_4, self.conv_4_features = conv_block(inputs=conv_block_3,
                                                        number_of_filters=512,
                                                        kernel_size=(5, 5),
                                                        strides=(1, 1),
                                                        padding='SAME',
                                                        activation=tf.nn.relu,
                                                        max_pool=True,
                                                        batch_norm=True)

        #flattening
        flat_layer = tf.layers.Flatten()(conv_block_4)

        #Dense block 1
        dense_block_1, self.dense_1_features = dense_block(flat_layer,
                                                           units=128,
                                                           activation=tf.nn.relu,
                                                           dropout_rate=self.dropout_rate,
                                                           batch_norm=True)

        #Dense block 2
        dense_block_2, self.dense_2_features = dense_block(dense_block_1,
                                                           units=256,
                                                           activation=tf.nn.relu,
                                                           dropout_rate=self.dropout_rate,
                                                           batch_norm=True)

        #Dense block 3
        dense_block_3, self.dense_3_features = dense_block(dense_block_2,
                                                           units=512,
                                                           activation=tf.nn.relu,
                                                           dropout_rate=self.dropout_rate,
                                                           batch_norm=True)

        #Dense block 4
        dense_block_4, self.dense_4_features = dense_block(dense_block_3,
                                                           units=1024,
                                                           activation=tf.nn.relu,
                                                           dropout_rate=self.dropout_rate,
                                                           batch_norm=True)

        logits = tf.layers.Dense(units=number_of_classes, activation=None)(dense_block_4)

        self.predictions = tf.nn.softmax(logits)

        self.loss, self.opt = opt_loss(logits=logits,
                                       targets=self.targets,
                                       learning_rate=learning_rate)
