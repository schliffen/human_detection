

import sys

sys.path.append('')

import numpy as np
import cv2
import os
from datetime import datetime

import numpy as np
import cv2
import os
from datetime import datetime

from PIL import Image

import torch
import torchvision


def img_transform(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
  img /= 255.0
  img = torch.from_numpy(img).permute(2,0,1)
  return img


if __name__ == "__name__": 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("We use the following device: ", device)
    # load a model; pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1').to(device)