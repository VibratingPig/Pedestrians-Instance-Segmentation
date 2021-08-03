import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import pandas as pd
import ast
import math


class MaskCreator():

    def __init__(self):
        pass

    def create_mask(self, x, y):
        return np.array([1, 2, 3])


class PennnFudanDataset(Dataset):
    def __init__(self, root, transform=None, use_masks=True):

        self.transform = transform
        self.root = root
        if self.root == 'Kaggle':
            use_masks = False
        self.use_masks = use_masks

        # root paths
        self.rootImages = os.path.join(self.root, "PNGImages")
        self.rootMasks = os.path.join(self.root, "PedMasks")
        # self.rootAnnotation = os.path.join(self.root, "Annotation")

        # list of data paths
        self.imagesPaths = sorted(os.listdir(self.rootImages))
        self.masksPaths = sorted(os.listdir(self.rootMasks))
        # self.annotationPaths  = sorted( os.listdir(self.rootAnnotation))

        self.imagesPaths = [os.path.join(self.rootImages, image) for image in self.imagesPaths]
        self.masksPaths = [os.path.join(self.rootMasks, mask) for mask in self.masksPaths]

    def __getitem__(self, index):

        image_name = self.imagesPaths[index].split('/')[2].split(".")[0]
        # load image & mask
        image = Image.open(self.imagesPaths[index])
        if self.use_masks:
            mask = Image.open(self.masksPaths[index])

        # image = image.convert("RGB")

        # We get the boxes from the masks instead of reading it from a CSV file

        # get list of object IDs (Pedestrians in the mask)
        # ex: if mask has 3 people in it, IDs = [0, 1, 2, 3] ... 0 for background and 1,2,3 for each pedestrian

        boxes = []
        # area for each box
        area = []

        if self.use_masks:
            IDs = np.unique(np.array(mask))
            # remove the background ID
            IDs = IDs[1:]

            # transpose it to (N,1,1) to be similar to a column vector
            IDs = IDs.reshape(-1, 1, 1)

            masks = np.array(mask) == IDs

            # N Boxes
            N = len(IDs)

            for i in range(N):
                # where gets the pixels where the mask = True (mask is a 2D Array of true and false ,
                # true at the pixels that is 1 as an indication of the mask & 0 for background)
                mask_pixels = np.where(masks[i])

                # extract the box from the min & max of these points
                # first dim is y , second dim is x
                xmin = np.min(mask_pixels[1])
                xmax = np.max(mask_pixels[1])
                ymin = np.min(mask_pixels[0])
                ymax = np.max(mask_pixels[0])

                # PG This is where the boxes are appended - it's worked out from the mask png
                # if we don't have that then we can support the relevant xmin, xmax, ymin and ymax
                # to teh
                boxes.append([xmin, ymin, xmax, ymax])
                area.append((ymax - ymin) * (xmax - xmin))

        if not self.use_masks:
            # PG Hard coded from train_image_level.csv
            # xmin = 2753
            # ymin = 906
            # xmax = xmin + 1058
            # ymax = ymin + 1672
            # boxes.append([xmin, ymin, xmax, ymax])
            # area.append((ymax-ymin) * (xmax-xmin))
            #
            # xmin = 788
            # ymin = 820
            # xmax = xmin + 1144
            # ymax = ymin + 1879
            # boxes.append([xmin, ymin, xmax, ymax])
            # area.append((ymax-ymin) * (xmax-xmin))

            df = pd.read_csv('./Kaggle/PedMasks/train_image_level.csv')

            image_id = image_name.split('.')[0] + "_image"
            # print(f'considering image {image_id}')
            row = df[df.id == image_id]
            oo = np.zeros([1, image.width, image.height])
            # row.boxes is a pandas series
            for box in row.boxes:
                # print(f'Attempting to literally parse box {box} for {image_id}')
                # we have no bounding boxes for some of these images and they should report nothing
                # for the purposes of our training we ignore them

                try:
                    list_of_dictionaries = ast.literal_eval(box)
                except:
                    print(f'whatever reason cannot parse {box} for {image_id}')
                    return None

                N = len(list_of_dictionaries)
                oo = np.zeros([N, image.size[0], image.size[1]])
                for i, boundary_boxes in enumerate(list_of_dictionaries):
                    # box is a string representation of the bounds
                    # its a list of dictionaries

                    x = int(boundary_boxes['x'])
                    y = int(boundary_boxes['y'])
                    x_width = int(boundary_boxes['width'])
                    y_height = int(boundary_boxes['height'])

                    # PG temporary scaling to 800 px wide
                    scale = 800 / 2770

                    x = int(scale * x)
                    y = int(scale * y)
                    x_width = int(scale * x_width)
                    y_height = int (scale * y_height)

                    # print(f'For {image_name} setting {x},{y} to {x+x_width}, {y+y_height} for {x_width} and {y_height}')
                    boxes.append([x, y, x + x_width, y + y_height])
                    area.append(x_width * y_height)
                    oo[i, x:(x+x_width), y:(y+y_height)] = i + 1

                # this is the number of masks/bounding boxes not the number of images.

            # create an array of size X * Y and then set the range of the mask
            # x[1,1:1,] = 1
            # size of image
            # note that the masks are multidimensional array and the supplied
            # mask png sets mask 1 to value 1 and mask 2 to value 2 etc. to produce a multidimensional
            # array
            # indexed from 0
            # box 2
            # oo[0, 788:1932, 820:2699] = 2
            # then do a boolean x_prime = x[==True]
            # you can pass an array and it returns an multi dimensional array
            IDs = np.unique(np.array(oo))
            # remove the background ID
            IDs = IDs[1:]

            # transpose it to (N,1,1) to be similar to a column vector
            IDs = IDs.reshape(-1, 1, 1)

            masks = np.array(oo) == IDs
            # masks = (oo == [[1],[2]])

        # convert 2D List to 2D Tensor (this is not numpy array)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)

        # labels for each box
        # there is only 1 class (pedestrian) so it will always be 1 (if multiple classes, so we will assign 1,2,3 ... etc to each one)
        labels = torch.ones((N,), dtype=torch.int64)

        # image_id requirement for model, index is unique for every image
        image_id = torch.tensor([index], dtype=torch.int64)

        # instances with iscrowd=True will be ignored during evaluation.
        # set all = False (zeros)
        iscrowd = torch.zeros((N,), dtype=torch.uint8)

        # convert masks to tensor (model requirement)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # print("image size=", image.size)
        # print("mask size=", mask.size)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["area"] = area
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        try:
            if self.transform is not None:
                image = self.transform(image)
        except:
            print(f'image resize failed returning none {image_id}')
            return None

        return image, target

    def __len__(self):
        return len(self.imagesPaths)
