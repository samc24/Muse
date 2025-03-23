# go to https://github.com/eriklindernoren/PyTorch-YOLOv3 and follow all yolo-pytorch installation rules before running
# git clone https://github.com/eriklindernoren/PyTorch-YOLOv3 to this directory
# make sure working directory is in PyTorch-Yolo

from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
import torch
from PIL import Image
from torch.utils.data import DataLoader
import cv2
import os
from torch.autograd import Variable
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import numpy as np


def id_to_string(id, n = 8):
  """
  simple helper function that converts an n-digit number to a proper string id, useful for indexing frames in a video
  """
  len_num = len(str(id))
  return '0'*(n - len_num) + str(id)

def save_tagged_frames(vid, output):
    """
    :param vid: filename of video
    :param output: folder of output to save tagged frames
    :return: only returns the fps of the video as an integer
    """
    if not os.path.exists(output):
        os.mkdir(output)
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    i = 0
    vidcap = cv2.VideoCapture(vid)
    hasNext = True
    time_taken = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes('data/coco.names')
    trans = transforms.Compose([DEFAULT_TRANSFORMS, Resize(416)])

    model = Darknet('config/yolov3.cfg', img_size=416).to(device)
    model.load_darknet_weights('weights/yolov3.weights')
    model.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    while True:
      num_skips = round(time_taken / (1 / fps))
      for j in range(num_skips + 1):
        hasNext,frame = vidcap.read()
        i += 1
        if not hasNext:
          break
      print(num_skips)
      prev_time = time.time()
      image, _ = trans((frame, np.zeros((5,5))))
      # Configure input
      input_imgs = Variable(image.type(Tensor))
      input_imgs = input_imgs.view(1, 3, 416, 416)
      # Get detections
      with torch.no_grad():
          detections = model(input_imgs)
          detections = non_max_suppression(detections, 0.5, 0.3)
      detections = detections[0]
      # Log progress
      current_time = time.time()
      inference_time = current_time - prev_time
      time_taken = inference_time
      prev_time = current_time
      print("\t+ Batch %d, Inference Time: %s" % (i, inference_time))

      # Save image and detections

      img = frame
      plt.figure()
      fig, ax = plt.subplots(1)
      ax.imshow(img)
      detections = rescale_boxes(detections, 418, img.shape[:2])
      unique_labels = detections[:, -1].cpu().unique()

      n_cls_preds = len(unique_labels)

      bbox_colors = random.sample(colors, n_cls_preds)

      for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

        print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s=classes[int(cls_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0},
        )
      plt.axis("off")
      plt.gca().xaxis.set_major_locator(NullLocator())
      plt.gca().yaxis.set_major_locator(NullLocator())
      filename = id_to_string(i)
      output_path = os.path.join(output, f"{filename}.png")
      plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0, dpi = 200)
      plt.close()
    return fps

def create_tagged_video(output, video_output, fps):
    """
    :param output: filename of saved frames tagged with outputs
    :param video_output: filename to save tagged video as
    :param fps: the fps of the video to save
    :return: None
    """

    files = sorted(os.listdir(output))
    img = cv2.imread(os.path.join(output, files[0]))
    height, width, layers = img.shape
    size = (width, height)

    ints = [int(f.split('.')[0]) for f in files]

    out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    curr_int = 1

    for i, num in enumerate(ints[:-1]):
        if num == curr_int:
            img = cv2.imread(os.path.join('../output/', files[i]))
            while curr_int < ints[i + 1]:
                out.write(img)
                curr_int += 1

    out.release()

