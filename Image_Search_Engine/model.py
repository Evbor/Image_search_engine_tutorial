import pickle
import numpy as np
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

def train(model,
          epochs,
          drop_rate,
          batch_size,
          data,
          save_dir,
          saver_delta=0.15):
    '''
    The core training function, use this function to train a model.

    :param model: CNN model
    :param epochs: integer, number of epochs
    :param drop_rate: float, dropout_rate
    :param batch_size: integer, number of samples to put through the model at once
    :param data: tuple, train-test data Example (X_train, y_train, X_test, y_test)
    :param save_dir: String, path to a folder where model checkpoints will be saved
    :param saver_delta: float, used to prevent overfitted model to be saved
    '''

    X_train, y_train, X_val, y_val = data

    # Tensorflow Session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2

    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer()) # Initializing global variables of the Tensorflow session

    # Defining a Tensorflow Saver object
    saver = tf.train.Saver()

    best_val_accuracy = 0

    # Training loop
    for epoch in range(epochs):

        train_accuracy = []
        train_loss = []

        for ii in tqdm_notebook(range(len(X_train) // batch_size)):
            start_id = ii * batch_size
            end_id = start_id + batch_size

            X_batch = X_train[start_id:end_id]
            y_batch = y_train[start_id:end_id]

            feed_dict = {model.inputs: X_batch,
                         model.targets: y_batch,
                         model.dropout_rate: drop_rate}

            _, t_loss, preds_t = session.run([model.opt, model.loss, model.predictions], feed_dict=feed_dict)

            train_accuracy.append(sparse_accuracy(y_batch, preds_t))
            train_loss.append(t_loss)

        print("Epoch: {}/{}".format(epoch, epochs),
              " | Training accuracy: {}".format(np.mean(train_accuracy)),
              " | Training loss: {}".format(np.mean(train_loss)))

        # Evaluating on Validation set
        val_accuracy = []
        val_loss = []

        for ii in tqdm_notebook(range(len(X_val) // batch_size)):
            start_id = ii * batch_size
            end_id = start_id + batch_size

            X_batch = X_val[start_id:end_id]
            y_batch = y_val[start_id:end_id]

            feed_dict = {model.inputs: X_batch,
                         model.targets: y_batch,
                         model.dropout_rate: 0.0}

            v_loss, preds_val = session.run([model.loss, model.predictions], feed_dict=feed_dict)

            val_accuracy.append(sparse_accuracy(y_batch, preds_val))
            val_loss.append(v_loss)

        print("Validation accuracy: {0}, Validation loss: {1}".format(np.mean(val_accuracy), np.mean(val_loss)))

        # Saving the model
        if np.mean(train_accuracy) > np.mean(val_accuracy):
            if np.abs(np.mean(train_accuracy) - np.mean(val_accuracy)) <= saver_delta:
                if np.mean(val_accuracy) >= best_val_accuracy:
                    best_test_accuracy = np.mean(val_accuracy)
                    saver.save(session, "{}/model_epochs_{}.ckp".format(save_dir, epoch))

    session.close()

    return None

def create_training_set_vectors(model,
                                X_train,
                                y_train,
                                batch_size,
                                checkpoint_path,
                                image_size,
                                distance="hamming"):
    '''
    Creates training set vectors and saves them in a pickle file.

    :param model: CNN model
    :param X_train: numpy array, loaded training set images
    :param y_train: numpy array, loaded training set labels
    :param batch_size: integer, number of samples to put through the model at once
    :param checkpoint_path: String, path to model checkpoint
    :param image_size: tuple, single image (height, width)
    :param distance: String, type of distance metric to be used,
                             this parameter is used to choose a way how to prepare and save training set vectors
    '''

    config = tf.ConfigProto() # configuring the tensorflow session
    config.gpu_options.per_process_gpu_memory_fraction = 0.2 # for a gpu session we have to configure the percentage of gpu memory tensorflow may eat up or else we might run out of memory on the gpu
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())

    if checkpoint_path != None:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_path)

    dense_2_features = []
    dense_4_features = []

    for ii in tqdm_notebook(range(len(X_train) // batch_size)):
        start_id = ii * batch_size
        end_id = start_id + batch_size

        X_batch = X_train[start_id:end_id]

        feed_dict = {model.inputs: X_batch,
                     model.dropout_rate: 0.0}

        dense_2, dense_4 = session.run([model.dense_2_features, model.dense_4_features], feed_dict=feed_dict)

        dense_2_features.append(dense_2)
        dense_4_features.append(dense_4)

    dense_2_features = np.vstack(dense_2_features)
    dense_4_features = np.vstack(dense_4_features)

    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1)
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)

        training_vectors = np.hstack((dense_2_features, dense_4_features))

        with open("hamming_train_vectors.pickle", "wb") as f:
            pickle.dump(training_vectors, f)

    if distance == "cosine":
        training_vectors = np.hstack((dense_2_features, dense_4_features))
        with open("cosine_train_vectors.pickle", "wb") as f:
            pickle.dump(training_vectors, f)

    return None
