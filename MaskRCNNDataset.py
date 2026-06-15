import os
import torch

from torchvision.io import read_image, decode_image
from PIL import Image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.utils.data import Dataset

from glob import glob
import json
import numpy as np
import cv2

class MySolderingDataset(Dataset):
    def __init__(self, dataset_dir, transforms):
        self.transforms = transforms
        self.dataset_dir = dataset_dir
        self.json_paths = glob(os.path.join(dataset_dir, "*.json"))
        self.class_ids = {"ok": 1, "short": 2, "insufficient": 3, "no_solder": 4, "solder_ball": 5}

    def __len__(self):
        return len(self.json_paths)
    
    def __getitem__(self, idx):
        # load images and masks
        json_path = self.json_paths[idx]
        with open(json_path, 'r') as jsonfile:
            jsondata = json.load(jsonfile)
        image_path = os.path.join(self.dataset_dir, jsondata['imagePath'])
        img = Image.open(image_path).convert("RGB") 
        masks, labels, boxes, area, iscrowd = self.get_instances(jsondata)
        
        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = idx
        target["area"] = torch.tensor(area)
        target["iscrowd"] = torch.tensor(iscrowd)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def get_instances(self, jsondata):
        height = jsondata["imageHeight"]
        width = jsondata["imageWidth"]
        shapes = jsondata["shapes"]
        masks = []
        labels = []
        boxes = []
        area = []
        iscrowd = []
        for shape in shapes:
            mask = np.zeros((height, width), dtype=np.uint8)
            label = shape["label"]
            points = np.array(shape["points"], dtype=int)
            shape_type = shape["shape_type"]
            if shape_type=="circle":
                points = self.convert_circle2polygon(points)
            mask = cv2.fillPoly(mask, [points], 1)
            masks.append(mask)
            labels.append(self.class_ids[label])
            xmin, ymin = np.min(points, axis=0)
            xmax, ymax = np.max(points, axis=0)
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
            area.append((xmax-xmin)*(ymax-ymin))
            iscrowd.append(0)
        masks = np.array(masks)
        labels = np.array(labels)
        boxes = np.array(boxes)
        area = np.array(area, dtype=np.float32)
        iscrowd = np.array(iscrowd)
        return masks, labels, boxes, area, iscrowd
    
    def convert_circle2polygon(self, points):
        cx, cy = points[0]
        rx, ry = points[1]
        thetas = np.arange(0, 360, 5)*np.pi/180
        r = np.sqrt((cx-rx)**2+(cy-ry)**2)
        new_points = []
        for theta in thetas:
            x = cx + r*np.cos(theta)
            y = cy + r*np.sin(theta)
            new_points.append([x, y])
        new_points = np.array(new_points, dtype=int)
        return new_points