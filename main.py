import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import matplotlib.image as img
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow.compat.v1 as tf
class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 10))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

MODEL_NAME = 'mobilenetv2_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = 'deeplab_model.tar.gz'

model_dir = tempfile.mkdtemp()
tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model, this might take a while...')
urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                   download_path)
print('download completed! loading DeepLab model...')

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')


SAMPLE_IMAGE = 'image1'  # @param ['image1', 'image2', 'image3']
IMAGE_URL = ""  #@param {type:"string"}
IMAGE_PATH = "./image.png"  #@param {type:"string"}
_SAMPLE_URL = ('data:image/jpeg;base64,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'
               'deeplab/g3doc/img/%s.jpg?raw=true')


#image_url = IMAGE_URL or _SAMPLE_URL % SAMPLE_IMAGE
#f = urllib.request.urlopen(image_url)
#jpeg_str = f.read()
#original_im = Image.open(BytesIO(jpeg_str))
original_im = Image.open(IMAGE_PATH)
resized_im, seg_map = MODEL.run(original_im)
#np.set_printoptions(shreshold=npp.inf, linewidth = np.inf)
print(resized_im)
vis_segmentation(resized_im, seg_map)

