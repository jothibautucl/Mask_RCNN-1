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
        print(file)
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
    parser.add_argument('--output', required=False,
                        metavar="output_file.txt",
                        help="Path to and name of the output file")
    args = parser.parse_args()

    if args.dataset is not None:
        INSECT_DIR = args.dataset
    if args.weights is None or args.weights == "last":
        print("Finding last weights")
        WEIGHTS_PATH = model.find_last()
    else:
        WEIGHTS_PATH = args.weights

    if args.output is None:
        OUTPUT_FILENAME = "bbox.txt"
    else:
        OUTPUT_FILENAME = args.output

    BOMBUS_LAPIDARIUS_DIR = os.path.join(INSECT_DIR, "bombus_lapidarius")
    BOMBUS_LUCORUM_DIR = os.path.join(INSECT_DIR, "bombus_lucorum")
    BOMBUS_PASCUORUM_DIR = os.path.join(INSECT_DIR, "bombus_pascuorum")
    BOMBUS_PRATORUM_DIR = os.path.join(INSECT_DIR, "bombus_pratorum")
    BOMBUS_TERRESTRIS_DIR = os.path.join(INSECT_DIR, "bombus_terrestris")
    VESPA_CRABRO_DIR = os.path.join(INSECT_DIR, "vespa_crabro")

    print("Weights: ", WEIGHTS_PATH)
    print("Dataset: ", INSECT_DIR)

    model.load_weights(WEIGHTS_PATH, by_name=True)

    '''
        Make inference on the validation set, use the below code, which picks up an image randomly from validation and run the detection.
    '''

    #class_names = ['BG', 'bombus_lapidarius', 'bombus_lucorum', 'bombus_pascuorum']#, 'bombus_pratorum', 'bombus_terrestris', 'vespa_crabro']
    class_names = ['BG', 'bombus_pratorum', 'bombus_terrestris', 'vespa_crabro']
    #class_id_bombus_lapidarius = 1
    #class_id_bombus_lucorum = 2
    #class_id_bombus_pascuorum = 3
    class_id_bombus_pratorum = 1
    class_id_bombus_terrestris = 2
    class_id_vespa_crabro = 3

    # Load a random image from the images folder
    #images_bombus_lapidarius, filenames_bombus_lapidarius = get_images_of_dataset(BOMBUS_LAPIDARIUS_DIR)
    #images_bombus_lucorum, filenames_bombus_lucorum = get_images_of_dataset(BOMBUS_LUCORUM_DIR)
    #images_bombus_pascuorum, filenames_bombus_pascuorum = get_images_of_dataset(BOMBUS_PASCUORUM_DIR)
    images_bombus_pratorum, filenames_bombus_pratorum = get_images_of_dataset(BOMBUS_PRATORUM_DIR)
    images_bombus_terrestris, filenames_bombus_terrestris = get_images_of_dataset(BOMBUS_TERRESTRIS_DIR)
    images_vespa_crabro, filenames_vespa_crabro = get_images_of_dataset(VESPA_CRABRO_DIR)

    images_per_class = {
                        #class_id_bombus_lapidarius: images_bombus_lapidarius,
                        #class_id_bombus_lucorum: images_bombus_lucorum,
                        #class_id_bombus_pascuorum: images_bombus_pascuorum
                        class_id_bombus_pratorum: images_bombus_pratorum,
                        class_id_bombus_terrestris: images_bombus_terrestris,
                        class_id_vespa_crabro: images_vespa_crabro
                        }
    filenames_per_class = {
                           #class_id_bombus_lapidarius: filenames_bombus_lapidarius,
                           #class_id_bombus_lucorum: filenames_bombus_lucorum,
                           #class_id_bombus_pascuorum: filenames_bombus_pascuorum
                           class_id_bombus_pratorum: filenames_bombus_pratorum,
                           class_id_bombus_terrestris: filenames_bombus_terrestris,
                           class_id_vespa_crabro: filenames_vespa_crabro
                           }

    compute = False

    with open(OUTPUT_FILENAME, 'a') as file:
        for j in range(1, len(class_names)):
            images = images_per_class[j]
            filenames = filenames_per_class[j]
            for i in range(0, len(images)):
                if filenames[i] == "vespa_crabro0476.jpg":
                    compute = True
                if compute:
                    print(filenames[i])
                    results = model.detect([images[i]], verbose=1)
                    r = results[0]
                    for k in range(len(r['rois'])):
                        (y1, x1, y2, x2) = r['rois'][k]
                        file.write(str(filenames[i])+","+ str(x1)+","+ str(y1)+","+ str(x2)+","+ str(y2)+","+ str(class_names[j])+"\n")
