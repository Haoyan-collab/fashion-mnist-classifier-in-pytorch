"""
Fashion-MNIST Classifier in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
import os
import gzip
import tempfile
import urllib.request
import numpy as np
import torch
def load_fashion_mnist(n_train=10000, n_test=2000):
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    temp_dir = tempfile.gettempdir()

    paths = {}

    # download each file once
    for key, filename in files.items():
        path = os.path.join(temp_dir, filename)
        paths[key] = path

        if not os.path.exists(path):
            urllib.request.urlretrieve(base_url + filename, path)

    # parse training images
    with gzip.open(paths["train_images"], "rb") as f:
        x_train = np.frombuffer(
            f.read(),
            dtype=np.uint8,
            offset=16
        ).reshape(-1, 28, 28)

    # parse training labels
    with gzip.open(paths["train_labels"], "rb") as f:
        y_train = np.frombuffer(
            f.read(),
            dtype=np.uint8,
            offset=8
        )

    # parse test images
    with gzip.open(paths["test_images"], "rb") as f:
        x_test = np.frombuffer(
            f.read(),
            dtype=np.uint8,
            offset=16
        ).reshape(-1, 28, 28)

    # parse test labels
    with gzip.open(paths["test_labels"], "rb") as f:
        y_test = np.frombuffer(
            f.read(),
            dtype=np.uint8,
            offset=8
        )

    # keep requested number of samples
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]
    x_test = x_test[:n_test]
    y_test = y_test[:n_test]

    # convert to torch tensors
    # images: float32 and scale pixel values from 0-255 to 0-1
    x_train = torch.tensor(x_train, dtype=torch.float32) / 255.0
    x_test = torch.tensor(x_test, dtype=torch.float32) / 255.0

    # labels: int64
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    return {
        "X_train": x_train,
        "y_train": y_train,
        "X_test": x_test,
        "y_test": y_test,
    }

# Step 2 - FashionDataset (not yet solved)
# TODO: implement

# Step 3 - make_loaders (not yet solved)
# TODO: implement

# Step 4 - MLP (not yet solved)
# TODO: implement

# Step 5 - train_one_epoch (not yet solved)
# TODO: implement

# Step 6 - evaluate (not yet solved)
# TODO: implement

# Step 7 - fit (not yet solved)
# TODO: implement

# Step 8 - lr_range_test (not yet solved)
# TODO: implement

# Step 9 - random_search (not yet solved)
# TODO: implement

# Step 10 - test_accuracy (not yet solved)
# TODO: implement

# Step 11 - save_model (not yet solved)
# TODO: implement

# Step 12 - predict_classes (not yet solved)
# TODO: implement

