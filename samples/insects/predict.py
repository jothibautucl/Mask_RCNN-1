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

#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
BALLON_WEIGHTS_PATH = "../../logs/balloon20210204T1851/mask_rcnn_balloon_0030.h5"  # TODO: update this path

config = ip.InsectPolygonsConfig()
INSECT_DIR = os.path.join(ROOT_DIR, "dataset_25")


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


if __name__ == '__main__':
    # Load validation dataset
    dataset = ip.InsectPolygonsDataset()
    dataset.load_insect(INSECT_DIR, "val")

    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)

    weights_path = model.find_last()

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)


    '''
        Make inference on the validation set, use the below code, which picks up an image randomly from validation and run the detection.
    '''
    #
    # import matplotlib.image as mpimg
    #
    # image1 = mpimg.imread('/home/jonathan/Desktop/Master_Thesis/Mask_RCNN-1/dataset_25/val/bourdon_des_arbres/bourdon_des_arbres0025.jpg')
    # # Run object detection
    # print(len([image1]))
    # results1 = model.detect([image1], verbose=1)
    #
    # # Display results
    # ax = get_ax(1)
    # r1 = results1[0]
    # visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],
    #                             dataset.class_names, r1['scores'], ax=ax,
    #                             title="Predictions1")

    class_names = ['BG', 'boudon_des_arbres']

    # Load a random image from the images folder
    image = skimage.io.imread('../../dataset_25/val/bourdon_des_arbres/bourdon_des_arbres0027.jpg')
    # image = skimage.io.imread('../../dataset/train/abeille_mellifere/abeille_mellifère0101.jpg')
    # image = skimage.io.imread('~/Desktop/122504389_2752371608336477_1403272794490783653_n.jpg')

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    filename = 'test.jpg'
    r = results[0]
    visualize.save_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, filename, r['scores'])
