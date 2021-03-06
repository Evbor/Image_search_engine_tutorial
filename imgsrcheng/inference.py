
import os
import pickle
import configparser
import numpy as np
import tensorflow as tf
from ast import literal_eval
from imgsrcheng import path_to_resources as resources
from imgsrcheng.utils import *
from imgsrcheng.data import *
from imgsrcheng.model import ImageSearchModel, train, create_training_set_vectors

path_to_config_file = os.path.join(resources, "config.ini") # from working directory

def _load_global_vars():
    '''
    Function that loads global variables
    '''
    config = configparser.ConfigParser()
    config.read(path_to_config_file)
    globals()['cfg_learning_rate'] = float(config["DEFAULT"]["learning_rate"])
    globals()['cfg_image_size'] = literal_eval(config["DEFAULT"]["image_size"])
    globals()['cfg_number_of_classes'] = int(config["DEFAULT"]["number_of_classes"])
    globals()['cfg_epochs'] = int(config["DEFAULT"]["epochs"])
    globals()['cfg_batch_size'] = int(config["DEFAULT"]["batch_size"])
    globals()['cfg_dropout_probs'] = float(config["DEFAULT"]["dropout_probs"])
    globals()['cfg_use_gpu'] = bool(config["DEFAULT"]["use_gpu"])
    globals()['cfg_gpu_memory_fraction'] = float(config["DEFAULT"]["gpu_memory_fraction"])
    globals()['cfg_latest_model_checkpoint'] = config["DEFAULT"]["latest_model_checkpoint"]
    globals()['cfg_train_dataset_path'] = config["DEFAULT"]["train_dataset_path"]
    globals()['cfg_val_dataset_path'] = config["DEFAULT"]["val_dataset_path"]
    globals()['cfg_labels_file_path'] = config["DEFAULT"]["labels_file_path"]
    globals()['cfg_distance'] = config["DEFAULT"]["distance"]

    return None

# Loading other global variables
_load_global_vars()

def _simple_inference(model,
                     session,
                     train_set_vectors,
                     uploaded_image_path,
                     image_size,
                     distance="hamming"):
    '''
    Doing simple inference for single uploaded image.

    :param model: CNN model
    :param session: tf.Session, restored session object
    :param train_set_vectors: loaded training set vectors
    :param uploaded_image_path: String, path to the uploaded image
    :param image_size: tuple, single image (height, width)
    :param distance: String, type of distance to be used
                             this parameter is used to choose a way to prepare vectors
    '''

    image = image_loader(uploaded_image_path, image_size)
    feed_dict = {model.inputs: [image],
                 model.dropout_rate: 0.0}
    dense_2_features, dense_4_features = session.run([model.dense_2_features, model.dense_4_features],
                                                     feed_dict=feed_dict)

    closest_ids = None

    if distance == "hamming":
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1)
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)
        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))

        closest_ids = hamming_distance(train_set_vectors, uploaded_image_vector)
    elif distance == "cosine":
        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))

        closest_ids = hamming_distance(train_set_vectors, uploaded_image_vector)

    return closest_ids

def image_search(uploaded_image_path):
    '''
    Does an image search for a single uploaded image. Returns list of image paths.

    :param uploaded_image_path: String, path to uploaded image
    '''

    model = ImageSearchModel(cfg_learning_rate, cfg_image_size, number_of_classes=cfg_number_of_classes)

    try:
        # Generating a tensorflow session
        c = None
        if cfg_use_gpu:
            c = tf.ConfigProto()
            c.gpu_options.per_process_gpu_memory_fraction = cfg_gpu_memory_fraction
        session = tf.Session(config=c)

        # Restoring session
        saver = tf.train.Saver()
        saver.restore(session, cfg_latest_model_checkpoint)
    except:
        # If no model checkpoint exists then run ML pipeline to generate a new trained model
        ml_pipeline(model)

    try:
        # Loading training set image paths
        with open(os.path.join(resources, "pickles", "train_images_pickle.pickle"), "rb") as f:
            train_image_paths = pickle.load(f)

        # Loading training set vectors generated by model
        if cfg_distance == "hamming":
            with open(os.path.join(resources, "pickles", "hamming_train_vectors.pickle"), "rb") as f:
                train_set_vectors = pickle.load(f)
        elif cfg_distance == "cosine":
            with open(os.path.join(resources, "pickles", "cosine_train_vectors.pickle"), "rb") as f:
                train_set_vectors = pickle.load(f)
    except FileNotFoundError:
        # Creating train_images_pickle.pickle if file is missing
        X_train, y_train = dataset_preprocessing(cfg_train_dataset_path, cfg_labels_file_path, cfg_image_size, os.path.join(resources, "pickles", "train_images_pickle"))
        # Creating vectors pickle file if file is missing
        train_set_vectors = create_training_set_vectors(model, X_train, y_train, cfg_batch_size, cfg_latest_model_checkpoint, cfg_image_size, distance=cfg_distance)

    # Finding image path indices of the most similar ImageSearchModel
    result_ids = _simple_inference(model, session, train_set_vectors, uploaded_image_path, cfg_image_size, distance=cfg_distance)

    return np.array([train_image_paths[id] for id in result_ids])

def ml_pipeline(model=None):
    '''
    Runs the ML pipeline to train a model and generate vector representations of the training set images.
    If no model is provided then the model specified by the hyperparameters in the config file is generated
    and trained.

    :param model: ImageSearchModel, model to train and generate training set image training_vectors
    '''
    # Create new model specified in our config file if no model is passed to the function
    if model == None:
        model = ImageSearchModel(cfg_learning_rate, cfg_image_size, cfg_number_of_classes)

    # train and generate training set vector representations
    X_train, y_train = dataset_preprocessing(cfg_train_dataset_path, cfg_labels_file_path, cfg_image_size, os.path.join(resources, "pickles", "train_images_pickle"))
    X_val, y_val = dataset_preprocessing(cfg_val_dataset_path, cfg_labels_file_path, cfg_image_size, os.path.join(resources, "pickles", "test_images_pickle"))
    data = (X_train, y_train, X_val, y_val)
    model_checkpoint = train(model, cfg_epochs, cfg_dropout_probs, cfg_batch_size, data, os.path.join(resources, "saver"))

    # update configuration file with latest model_checkpoint
    config = configparser.ConfigParser()
    config.read(path_to_config_file)
    config.set("DEFAULT", "latest_model_checkpoint", model_checkpoint)
    with open(path_to_config_file, "w") as f:
        config.write(f)

    # update global variables
    _load_global_vars()

    _ = create_training_set_vectors(model, X_train, y_train, cfg_batch_size, None, cfg_image_size, distance=cfg_distance)

    return None
