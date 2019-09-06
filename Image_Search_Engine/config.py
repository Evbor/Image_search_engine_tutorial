# Model Hyperparameters
learning_rate = 0.001
image_size = (32, 32)
number_of_classes = 10
epochs = 20
batch_size = 128
dropout_probs = 0.6

# Other Configuration Values
use_gpu = True
gpu_memory_fraction = 0.2
latest_model_checkpoint = "saver/model_epochs_6.ckp"
train_dataset_path = "cifar/train/"
val_dataset_path = "cifar/test/"
labels_file_path = "cifar/labels.txt"

# Image Search Parameters
distance = "hamming"
