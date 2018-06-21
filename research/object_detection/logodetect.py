import matplotlib
matplotlib.use('Agg')

import os
from dotenv import load_dotenv
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import re
import time 
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

err = []

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError(
        'Please upgrade your tensorflow installation to v1.4.* or later!')
# This is needed to display the images.

from utils import label_map_util

from utils import visualization_utils as vis_util
# What model to download.
DROOT = os.getenv('CONTAINER') 

MODEL_NAME = DROOT + '/' + os.getenv('MODEL_DIR')
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/' + os.getenv('CKPT')

PATH_TO_TEST_IMAGES_DIR = DROOT + '/' + os.getenv('IMAGE_DIR')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(DROOT + '/' + 'training', os.getenv('PB_TXT'))

NUM_CLASSES = int(os.getenv('NUMBER_OF_CLASSES'))

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def run_inference_for_images(graph):
    with graph.as_default():
        with tf.Session() as sess:
            err=[]
            while True:
              TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)
              #print ("done")

              for image_path in TEST_IMAGE_PATHS:
                try:
                    # Get handles to input and output tensors
                    start_time = time.time()
                    image = Image.open(PATH_TO_TEST_IMAGES_DIR + "/" + image_path)
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    #image_np = load_image_into_numpy_array(image)
                    ops = tf.get_default_graph().get_operations()
                    all_tensor_names = {output.name for op in ops for output in op.outputs}
                    tensor_dict = {}
                    for key in [
                        'num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks'
                    ]:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                                tensor_name)
                    if 'detection_masks' in tensor_dict:
                        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                            detection_masks, detection_boxes, image.shape[0], image.shape[1])
                        detection_masks_reframed = tf.cast(
                            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                        # Follow the convention by adding back the batch dimension
                        tensor_dict['detection_masks'] = tf.expand_dims(
                            detection_masks_reframed, 0)
                    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                    # Run inference
                    output_dict = sess.run(tensor_dict,
                                           feed_dict={image_tensor: np.expand_dims(image, 0)})

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.uint8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]
                    if 'detection_masks' in output_dict:
                        output_dict['detection_masks'] = output_dict['detection_masks'][0]


                    width, height = image.size
                    a={}
                    a["image"]=image_path
                    c={}
                    p=[]

                    for i in range(0,len(output_dict["detection_scores"])):
                          c={}
                          if (output_dict["detection_scores"][i]>0.5):

                              c["class"]=str(output_dict["detection_classes"][i])
                              c["prob"]=str(output_dict["detection_scores"][i])
                              box={}
                              box["x_min"]=str((output_dict["detection_boxes"][i][1])*height)
                              box["y_min"]=str((output_dict["detection_boxes"][i][0])*width)
                              box["x_max"]=str((output_dict["detection_boxes"][i][3])*height)
                              box["y_max"]=str((output_dict["detection_boxes"][i][2])*width)
                  #             print ("class: "+str(output_dict["detection_classes"][i]))
                  #             print ("confidence: "+str(output_dict["detection_scores"][i]))
                  #             print ("ymin: "+str((output_dict["detection_boxes"][i][0])*height))
                  #             print ("xmin: "+str((output_dict["detection_boxes"][i][1])*width))
                  #             print ("ymax: "+str((output_dict["detection_boxes"][i][2])*height))
                  #             print ("xmax: "+str((output_dict["detection_boxes"][i][3])*width))
                  #             print ("*****")
                              c["box"]=box
                              p.append(c)
                          else:
                              break
                    a["problist"]=p
                    a["fps"] = time.time()-start_time
                    print (json.dumps(a))
                    sys.stdout.flush()
                    os.remove(PATH_TO_TEST_IMAGES_DIR+image_path)
                except Exception as e:     # most generic exception you can catch
                    err.append(str(e))  

    return a


output_dict = run_inference_for_images(detection_graph)


