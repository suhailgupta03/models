from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dotenv import load_dotenv

import os
import argparse
import shutil
import json
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

SK2_MODEL_FILE = os.getenv("SK2_MODEL_FILE")
SK2_LABEL_FILE = os.getenv("SK2_LABEL_FILE")
SK2_CROPPED_FOLDER = os.getenv("SK2_CROPPED_FOLDER")

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(
      dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  model_file = SK2_MODEL_FILE
  label_file = SK2_LABEL_FILE
  CROPPED_FOLDER = SK2_CROPPED_FOLDER
  input_height = 224
  input_width = 224
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"

  # os.makedirs('result/sk2')
  # os.makedirs('result/notsk2')

  parser = argparse.ArgumentParser()
  # parser.add_argument("--folder", help="folder to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  # if args.image:
  #   folder_name = args.folder
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer
  a = []
  graph = load_graph(model_file)
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  while True:

	  files = os.listdir(CROPPED_FOLDER)

	  with tf.Session(graph=graph) as sess:
	        for f in files:
			try:

				sp = f.split("+")
				# print (f.split("+"))

				t = read_tensor_from_image_file(
					(CROPPED_FOLDER+"/"+f),
				input_height=input_height,
				input_width=input_width,
				input_mean=input_mean,
				input_std=input_std)
				results = sess.run(output_operation.outputs[0], {
				input_operation.outputs[0]: t
				})
				results = np.squeeze(results)

				top_k = results.argsort()[-5:][::-1]
				labels = load_labels(label_file)
				# print (f)
				b = []
				# print (labels)
				if (results[1] > 0.5):
					a = {}
		            		a["image"] = sp[6]
		            		c = {}
		            		p = []
					box = {}
					c["prob"]=sp[5]
					c["class"] = str(3)
					box["x_min"] = sp[1]
					box["y_min"] = sp[2]
					box["x_max"] = sp[3]
					box["y_max"] = sp[4]
					c["box"] = box
					p.append(c)
					a["problist"] = p
					a = json.dumps(a)
					print (a)
					os.remove(CROPPED_FOLDER+"/"+f)

					# b=[f,"notsk2",results[0]]
					# shutil.copy("test"+"/"+f, "result/notsk2/")
				else:
					if (results[0] > 0.5):
						# b=[f,"sk2",results[1]]
						# shutil.copy("test"+"/"+f, "result/sk2/")
						idk = 1

				# a.append(b)
				# for i in top_k:
					# print(labels[i], results[i])
			except:
				idk=0
		  

