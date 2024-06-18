



import sys

sys.path.append('')

import numpy as np
import cv2
import os
from datetime import datetime

from PIL import Image

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as F

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from matplotlib import pyplot as plt
import pickle

import pandas as pd
import glob as glob
import random


from src.inference_tools import inference, img_transform
from src.visualize_bboxes import plot_image


if __name__=="__main__":



    img = cv2.imread("data/input.jpg")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img = img_transform(img)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("We use the following device: ", device)
    # load a model; pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1').to(device)


    boxes, scores, labels = inference(img, model, device)

    # visualiz the results
    plot_image(img, boxes, scores, labels, save_path="results/")


