import streamlit as st
import sys
import os
from PIL import Image
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt


# Import Mask RCNN
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import custom


def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

st.title("Chipping recognition tool")
DEVICE = "/cpu:0"
TEST_MODE = "inference"
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
custom_WEIGHTS_PATH = "mask_rcnn_damage_0010.h5"

config = custom.CustomConfig()
#custom_DIR = os.path.join(ROOT_DIR, "dataset")
#dataset = custom.CustomDataset()
# Load validation dataset
#dataset = custom.CustomDataset()
#dataset.load_custom(custom_DIR, "val")
# Must call before using the dataset
#dataset.prepare()
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    
st.write("Loading weights ", custom_WEIGHTS_PATH)
model.load_weights(custom_WEIGHTS_PATH, by_name=True)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image=numpy.asarray(image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting chips...")


    results = model.detect([image], verbose=1)
    st.write("Done.")
    st.write("Displaying results...")


    # Display results
    titles=[{"source": "", "id": 0, "name": "BG"},{"source": "", "id": 1, "name": "damage"}]
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances2(image, r['rois'], r['masks'], r['class_ids'],
                                titles, r['scores'], ax=ax,
                                title="Prediction")
    classes = r['class_ids']
    st.write("Total Objects found", len(classes))
    #  st.image(image, channels="RGB")
    #st.write('%s (%.2f%%)' % (label[1], label[2]*100))