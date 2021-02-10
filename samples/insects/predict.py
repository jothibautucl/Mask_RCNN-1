import os
import sys
import random
import math
import re
import time
import numpy as np
import skimage
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import insects_polygons as ip

# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
WEIGHTS_PATH = "../../logs/balloon20210207T1903/mask_rcnn_balloon_0030.h5"

config = ip.InsectPolygonsConfig()
INSECT_DIR = os.path.join(ROOT_DIR, "dataset_100")
ABEILLE_MELLIFERE_DIR = os.path.join(INSECT_DIR, "predict/abeille_mellifere")
BOURDON_DES_ARBRES_DIR = os.path.join(INSECT_DIR, "predict/bourdon_des_arbres")
ANTHOPHORE_PLUMEUSE_DIR = os.path.join(INSECT_DIR, "predict/anthophore_plumeuse")
BOURDON_DES_JARDINS_DIR = os.path.join(INSECT_DIR, "predict/bourdon_des_jardins")


# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def get_images_of_dataset(dataset_dir):
    files = os.listdir(dataset_dir)
    images_of_dataset = []
    for file in files:
        images_of_dataset.append(skimage.io.imread(os.path.join(dataset_dir, file)))
    return images_of_dataset

def accuracy(mat):
    return np.trace(mat)/np.sum(mat)

def class_accuracy(vec, class_id):
    return vec[class_id]/np.sum(vec)

if __name__ == '__main__':
    
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)

    # Load weights
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Use Mask R-CNN to detect insects.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/insects/dataset/",
                        help='Directory of the Insects dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file, 'coco', 'last' or nothing (which will be interpreted as 'last')")
    args = parser.parse_args()

    if args.dataset is not None:
        INSECT_DIR = args.dataset
    if args.weights != "last":
        WEIGHTS_PATH = args.weights
    else:
        WEIGHTS_PATH = model.find_last()

    print("Weights: ", WEIGHTS_PATH)
    print("Dataset: ", INSECT_DIR)

    model.load_weights(WEIGHTS_PATH, by_name=True)

    '''
        Make inference on the validation set, use the below code, which picks up an image randomly from validation and run the detection.
    '''

    class_names = ['BG', 'abeille_mellifere', 'boudon_des_arbres', 'anthophore_plumeuse', 'bourdon_des_jardins']
    class_id_abeille_mellifere = 1
    class_id_bourdon_des_arbres = 2
    class_id_anthophore_plumeuse = 3
    class_id_bourdon_des_jardins = 4

    # Load a random image from the images folder
    images_per_class = {class_id_abeille_mellifere: get_images_of_dataset(ABEILLE_MELLIFERE_DIR),
                        class_id_bourdon_des_arbres: get_images_of_dataset(BOURDON_DES_ARBRES_DIR),
                        class_id_anthophore_plumeuse: get_images_of_dataset(ANTHOPHORE_PLUMEUSE_DIR),
                        class_id_bourdon_des_jardins: get_images_of_dataset(BOURDON_DES_JARDINS_DIR)}


    result_matrix = np.zeros((len(class_names)-1, len(class_names)-1))
    for j in range(1, len(class_names)):
        images = images_per_class[j]
        filename = 'test{:04}.jpg'
        for i in range(0, len(images)):
            results = model.detect([images[i]], verbose=1)
            r = results[0]
            class_ids = r['class_ids']
            for class_id in class_ids:
                result_matrix[j-1][class_id] += 1
            visualize.save_instances(images[i], r['rois'], r['masks'], r['class_ids'],
                                     class_names, filename.format(i), r['scores'])

    print("General accuracy : " + str(accuracy(result_matrix)))
    for j in range(1, len(class_names)):
        class_name = class_names[j]
        acc = class_accuracy(result_matrix[j-1], j-1)
        print("Accuracy of class '" + class_name + "' : " + str(acc))
