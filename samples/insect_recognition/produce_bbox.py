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

import insects as ip

# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
WEIGHTS_PATH = "../../logs/"  # TODO: add default

config = ip.InsectPolygonsConfig()
INSECT_DIR = os.path.join(ROOT_DIR, "dataset_100/predict")


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


def get_images_of_dataset(dataset_dir):
    files = os.listdir(dataset_dir)
    images_of_dataset = []
    filenames_of_dataset = []
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext == ".jpg" or ext == ".png" or ext == ".jpeg":
            # print(os.path.join(dataset_dir, file))
            filenames_of_dataset.append(file)
            images_of_dataset.append(skimage.io.imread(os.path.join(dataset_dir, file)))
    return images_of_dataset, filenames_of_dataset


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
    if args.weights is None or args.weights == "last":
        print("Finding last weights")
        WEIGHTS_PATH = model.find_last()
    else:
        WEIGHTS_PATH = args.weights

    ABEILLE_MELLIFERE_DIR = os.path.join(INSECT_DIR, "abeille_mellifere")
    BOURDON_DES_ARBRES_DIR = os.path.join(INSECT_DIR, "bourdon_des_arbres")
    ANTHOPHORE_PLUMEUSE_DIR = os.path.join(INSECT_DIR, "anthophore_plumeuse")
    BOURDON_DES_JARDINS_DIR = os.path.join(INSECT_DIR, "bourdon_des_jardins")

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
    images_abeille_mellifere, filenames_abeille_mellifere = get_images_of_dataset(ABEILLE_MELLIFERE_DIR)
    images_bourdon_des_arbres, filenames_bourdon_des_arbres = get_images_of_dataset(BOURDON_DES_ARBRES_DIR)
    images_anthophore_plumeuse, filenames_anthophore_plumeuse = get_images_of_dataset(ANTHOPHORE_PLUMEUSE_DIR)
    images_bourdon_des_jardins, filenames_bourdon_des_jardins = get_images_of_dataset(BOURDON_DES_JARDINS_DIR)

    images_per_class = {class_id_abeille_mellifere: images_abeille_mellifere,
                        class_id_bourdon_des_arbres: images_bourdon_des_arbres,
                        class_id_anthophore_plumeuse: images_anthophore_plumeuse,
                        class_id_bourdon_des_jardins: images_bourdon_des_jardins}
    filenames_per_class = {class_id_abeille_mellifere: filenames_abeille_mellifere,
                           class_id_bourdon_des_arbres: filenames_bourdon_des_arbres,
                           class_id_anthophore_plumeuse: filenames_anthophore_plumeuse,
                           class_id_bourdon_des_jardins: filenames_bourdon_des_jardins}

    with open('bbox.txt', 'a') as file:
        for j in range(1, len(class_names)):
            images = images_per_class[j]
            filenames = filenames_per_class[j]
            for i in range(0, len(images)):
                results = model.detect([images[i]], verbose=1)
                r = results[0]
                for k in range(len(r['rois'])):
                    (y1, x1, y2, x2) = r['rois'][k]
                    file.write(str(filenames[i])+","+ str(x1)+","+ str(y1)+","+ str(x2)+","+ str(y2)+","+ str(class_names[j])+"\n")
