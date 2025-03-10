import ast
import math
import os
import subprocess
import tkinter as tk
import tkinter.ttk as ttk
from typing import Dict

import cv2
import pandas as pd
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
# last layer of each architecture for transfer learning
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from helpers import get_dataset_loaders

subprocess.run(['rm /home/piero/morespace/Documents/Pedestrians-Instance-Segmentation/inference/*'], shell=True)
subprocess.run(['rm /home/piero/morespace/Documents/Pedestrians-Instance-Segmentation/*.png'], shell=True)

config = {
    'train': True,
    'device': 'cuda',
    'step_size': 1,
    'number_of_steps': 2,
    'max_count_to_train': 1e6,  # zero indexed
    'gamma': 0.1,
    'learning_rate': 0.05,
    'dataset': 'Kaggle',
    'gradient_ui': False,
    'number_of_classes': 3,
    'box_thresh': 0.3
}

class ForwardHookCapture:

    def __init__(self):
        self.output = None
        self.inputs = None

    def hook(self, module, input, output):
        # at this point we have a tensor which is 400x432x64 - how do we represent that
        # we can't send it to plot.imshow - so how do we reduce the dimensionality

        self.output = output
        try:
            Tensor.retain_grad(self.output[0].tensors[0])
        except:
            Tensor.retain_grad(self.output)
        self.inputs = input


def mask_rcnn_transfer_learning(is_finetune: bool, num_classes: int, box_thresh : float):

    mask_RCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                   # rpn_pre_nms_top_n_train=2,
                                                                   # rpn_post_nms_top_n_train=2,
                                                                   # rpn_nms_thresh=0.9,
                                                                   # box_fg_iou_thresh=0.95,
                                                                   rpn_bg_iou_thresh=0.5,
                                                                   # setting these two differently allows for inbetween labelling of -1
                                                                   rpn_fg_iou_thresh=0.5,
                                                                   # rpn_score_thresh=0.5,
                                                                   # after 256 epochs we can reach
                                                                   # 0.50 threshold
                                                                   box_score_thresh=box_thresh,  # during inference only
                                                                   # box_batch_size_per_image=4096,
                                                                   # box_nms_thresh=0.6,
                                                                   # box_detections_per_img=2,
                                                                   # box_fg_iou_thresh=0.6,
                                                                   # image_mean=image_mean,
                                                                   # image_std=image_std
                                                                   )

    # # just train the modified layers
    # if not is_finetune:
    #     for param in mask_RCNN.parameters():
    #         param.requires_grad = False

    in_features_classes_fc = mask_RCNN.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = mask_RCNN.roi_heads.mask_predictor.conv5_mask.in_channels

    # same num of conv5_mask output
    # the lower the more discrete the mask - at 1 the bounding box is the mask
    hidden_layer = 256

    # num_classes = 0 (background) + 1 (covid) + 1 (non covid) === 2
    fastRCNN_TransferLayer = FastRCNNPredictor(in_channels=in_features_classes_fc, num_classes=num_classes)
    maskRCNN_TransferLayer = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=hidden_layer,
                                               num_classes=num_classes)

    mask_RCNN.roi_heads.box_predictor = fastRCNN_TransferLayer
    mask_RCNN.roi_heads.mask_predictor = maskRCNN_TransferLayer

    return mask_RCNN



if config['train']:
    subprocess.run(['rm /home/piero/morespace/Documents/Pedestrians-Instance-Segmentation/images/*'], shell=True)


class PedestrianSegmentation:

    def __init__(self, system_config: Dict[str, object]):

        self.device = system_config['device']
        self.ui = system_config['gradient_ui']

        self.root = system_config['dataset']
        self.transform = transforms.Compose([transforms.ToTensor()])

        # number of boxes to report when evaluating - too many looks hokey
        self.max_count_to_train = system_config['max_count_to_train']  # high number means no breaking
        # filter for incoming masks on detection - the value must be greater than this
        self.mask_weight = 0.5
        # threshold for bounding box evaluation
        self.IoU_threshold = 0.25
        gamma = system_config['gamma']  # the amount the learning rate reduces each step
        step_size = system_config['step_size']
        self.batch_size = 1
        self.learning_rate = system_config['learning_rate']
        self.epochs = system_config['number_of_steps'] * step_size  # make it a multiple of three for the step size

        self.split_dataset_factor = 0.7

        # dataset
        # test_batch_size = 1 for looping over single sample
        self.train_loader, self.test_loader, self.dataset = get_dataset_loaders(self.transform, self.batch_size, 1,
                                                                                self.root, self.split_dataset_factor)

        # model
        self.mask_RCNN = mask_rcnn_transfer_learning(is_finetune=True, num_classes=system_config['number_of_classes'], box_thresh=system_config['box_thresh'])
        device = torch.device(self.device)
        self.mask_RCNN.to(device)

        # parameters of the modified layers via transfer learning
        # TODO so both fast and mask cnn are used together to produce the output
        # mask CNN is an additional FCN
        fast_rcnn_parameters = [param for param in self.mask_RCNN.roi_heads.box_predictor.parameters()]
        mask_rcnn_parameters = [param for param in self.mask_RCNN.roi_heads.mask_predictor.parameters()]
        self.parameters = fast_rcnn_parameters + mask_rcnn_parameters

        # optimizer & lr_scheduler
        self.optimizer = torch.optim.SGD(self.parameters, lr=self.learning_rate)
        # step rate 3 means every third iteration it reduces by a factor of 0.1 or 10%
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size, gamma=gamma)

        # path to save / load weights
        self.weights_path = "weights.pth"

        self.hook_cls = ForwardHookCapture()
        self.threshold = 0
        self.feature_value = 0

        self.losses = []
        self.test_losses = []
        self.backwards_counter = 0

    def build_ui(self):
        self.frame1 = ttk.Frame(tk.Tk())
        self.frame2 = ttk.Frame(self.frame1)
        self.frame3 = ttk.Frame(self.frame2)
        self.label2 = ttk.Label(self.frame3)
        self.label2.configure(text='feature number')
        self.label2.grid(column='0', row='0')
        self.scale2 = ttk.Scale(self.frame3)
        self.scale2.configure(from_='0', orient='horizontal', to='63')
        self.scale2.grid(column='1', row='0')
        self.scale2.configure(command=self.feature_changed)
        self.frame3.configure(height='200', width='200')
        self.frame3.grid(column='0', row='0')
        self.frame4 = ttk.Frame(self.frame2)
        self.label3 = ttk.Label(self.frame4)
        self.label3.configure(text='threshold')
        self.label3.grid(column='0', row='0')
        self.scale1 = ttk.Scale(self.frame4)
        self.scale1.configure(from_='0', orient='horizontal', to='255')
        self.scale1.grid(column='1', row='0', sticky='e')
        self.scale1.configure(command=self.scale_changed)
        self.frame4.configure(height='200', width='200')
        self.frame4.grid(column='0', row='1')
        self.frame2.configure(height='200', width='200')
        self.frame2.pack(side='top')
        self.label1 = ttk.Label(self.frame1)
        self.label1.configure(text='image to go here')
        self.label1.pack(side='top')
        self.frame1.configure(height='200', width='200')
        self.frame1.pack(side='top')
        self.mainwindow = self.frame1
        self.feature_changed(0)

    def feature_changed(self, scale_value):
        self.feature_value = round(float(scale_value))
        image = self.plot_feature(self.feature_value, threshold=self.threshold)
        photo_image = ImageTk.PhotoImage(image=Image.fromarray(image[:,:,self.feature_value].numpy()))
        self.label1.configure(image=photo_image)
        self.label1.image = photo_image
        self.label1.pack(side='top')
        self.frame1.pack(side='top')

    def scale_changed(self, scale_value):
        self.threshold = round(float(scale_value))
        image = self.plot_feature(self.feature_value, threshold=self.threshold, recalc=False)
        photo_image = ImageTk.PhotoImage(image=Image.fromarray(image[:,:,self.feature_value].numpy().astype('uint8'),
                                                               mode='RGB'))
        self.label1.configure(image=photo_image)
        self.label1.image = photo_image
        self.label1.pack(side='top')
        self.frame1.pack(side='top')

    def plot_feature(self, feature_index, threshold=125, recalc=True):
        if recalc:
            self.run_backwards()

        if recalc or not hasattr(self, "gradient"):
            try:
                gradient = self.hook_cls.output.grad.cpu().detach()[0]  # there will only ever be on grad
            except:
                try:
                    gradient = self.hook_cls.output.grad  # there will only ever be on grad
                except:
                    gradient = self.hook_cls.inputs[0].grad

        # gradient_image = gradient.permute(1, 2, 0)[:,:,feature_index]
        output_tmp = self.hook_cls.output[0, :, :, :].cpu().detach().permute(1,2,0)
        # convolved_image = output_tmp[:,:,feature_index]
                          # * gradient_image  # * self.hook_cls.inputs # * tensor_image
        return output_tmp

    def plot_feature_old_impl(self, threshold=125, recalc=True):
        if recalc:
            L = self.run_backwards()
            self.image = self.outputs[1].tensors[0].cpu().detach()

        if recalc or not hasattr(self, "gradient"):
            self.gradient = self.outputs[1].tensors.grad.cpu().detach()[0]  # there will only ever be on grad
        gradient_image = self.gradient.permute(1, 2, 0)

        self.outputs[1].tensors.grad = self.outputs[1].tensors.grad * 0
        tensor_image = self.image.permute(1, 2, 0)
        convolved_image = gradient_image * tensor_image
        for i in range(3):
            max = convolved_image[:, :, i].max()
            min = convolved_image[:, :, i].min()
            convolved_image[:, :, i] = (convolved_image[:, :, i] - min) / (max - min) * 255

        for i in range(3):
            convolved_image[:, :, i][convolved_image[:, :, i] < threshold] = 0

        return convolved_image.int()

    def train_one_epoch(self, images_list):

        count = 0
        device = torch.device(self.device)
        batches_per_epoch = len(self.dataset)
        if self.batch_size:
            batches_per_epoch = len(self.dataset) / self.batch_size

        for i, (images, targets) in enumerate(self.train_loader):
            # PG At this point the iages are a list of two images which appear to have different sizes
            # 3 channels for RGB and 341x414 and 482x550
            # PG the targets are dictionaries with boxes, labels, masks, area, imageid and iscrowd flag
            # boxes look like 2x4 tensor - so one for each image and 4 coordinates
            # masks appear to be boolean and sized to the smaller of the two images 341 x 414.
            # squirt them down to the card taken from the torchvision reference impl.
            if images:  # could be none due to being 0
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets if t is not None]
                self.images = images
                self.targets = targets

                # model in training mode accepts both input and output and return the loss of all types
                # PG See torchvision.models.detection.transform.GeneralizedRCNNTransform for the transformation applied
                # it does resize, mean shifts and std deviations shift the images prior to use.
                L = self.run_backwards()

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.losses.append(L.item())
                count += 1
                if count > self.max_count_to_train:
                    break

        for i, (images, targets) in enumerate(self.test_loader):
            if images:  # could be none due to being 0
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets if t is not None]
                self.images = images
                self.targets = targets

                L = self.run_backwards()

                # we don't step backwards here as we don't want to include the test/validation data
                self.test_losses.append(L.item())

    def run_backwards(self):
        self.outputs = self.mask_RCNN(self.images, self.targets)
        self.backwards_counter += 1
        loss = self.outputs[0]
        L_cls = loss["loss_classifier"]
        L_box = loss["loss_box_reg"]
        L_mask = loss["loss_mask"]
        L_objectness = loss["loss_objectness"]
        L_rpn = loss["loss_rpn_box_reg"]
        L = L_cls + L_box + L_mask + L_objectness + L_rpn
        L.backward()
        return L

    def train(self, images):

        # set to training mode
        # PG this sets a flag on all the modules in PyTorch to train
        # as the documentation states it only affects some modules such as batchnorm
        self.mask_RCNN.train()

        selected_tensor = self.mask_RCNN.backbone.body.conv1
        # Tensor.retain_grad(selected_tensor)
        selected_tensor.register_forward_hook(self.hook_cls.hook)

        for epoch in tqdm(range(0, self.epochs)):
            self.train_one_epoch(images)
            self.lr_scheduler.step()
            image_array = self.plot_feature_old_impl(threshold=119)
            image = Image.fromarray(image_array.numpy().astype('uint8'), mode='RGB')
            image.save(f'./images/image{epoch}.png')

        self.outputs = model.mask_RCNN(self.images, self.targets)

        plt.plot([math.log(loss) for loss in self.losses], label='Losses')
        plt.legend()
        plt.show()

        plt.plot([math.log(loss) for loss in self.test_losses], label='Test Losses')
        plt.legend()
        plt.show()

        if self.ui:
            self.build_ui()
            self.mainwindow.mainloop()

    def save(self):
        torch.save(self.mask_RCNN.state_dict(), self.weights_path)

    def load(self):
        weights = torch.load(self.weights_path)
        self.mask_RCNN.load_state_dict(weights)

    def detect(self, path: str, name: str):
        """
        run the forward inference using the path and the name
        :type path: object
        """
        image = Image.open(path)
        image = self.transform(image)

        device = torch.device(self.device)
        # PG We want the gradient information.
        with torch.no_grad():
            self.mask_RCNN.eval()
            combined_output = self.mask_RCNN([image.to(device)])
            output = combined_output[0][0]  # [0] because we pass 1 image

        boxes = output["boxes"].cpu().detach().numpy()
        scores = output["scores"].cpu().detach().numpy()
        labels = output["labels"].cpu().detach().numpy()

        img = cv2.imread(path)

        df = pd.read_csv('./Kaggle/PedMasks/train_image_level.csv')

        image_id = name.split('.')[0] + "_image"
        row = df[df.id == image_id]

        for i, score in enumerate(scores):
            x = round(boxes[i][0])
            y = round(boxes[i][1])
            cv2.rectangle(img, (x, y), (round(boxes[i][2]), round(boxes[i][3])),
                          color=(0, 0, 255), thickness=3)
            formatted_score = round(score * 100, 2)
            cv2.putText(img, f'Label {labels[i]} Score {formatted_score}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        for box in row.boxes:
            if type(box) == str or not math.isnan(box):
                list_of_dictionaries = ast.literal_eval(box)
                for _, boundary_boxes in enumerate(list_of_dictionaries):
                    x = int(boundary_boxes['x'])
                    y = int(boundary_boxes['y'])
                    x_width = int(boundary_boxes['width'])
                    y_height = int(boundary_boxes['height'])
                    cv2.rectangle(img, (x, y), (x + x_width, y + y_height),
                                  color=(255, 0, 0), thickness=3)
                    cv2.putText(img, 'Ground Truth', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                cv2.putText(img, 'NO COVID', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imwrite(f'./inference/{name}', img)


model = PedestrianSegmentation(config)

if config['train']:
    images = []
    model.train(images)
    model.save()
else:
    # # # Test
    model.load()
    index = 0
    root = f"{model.root}/Test"
    paths = sorted(os.listdir(f"{model.root}/Test"))
    for _path in paths:
        path = os.path.join(root, _path)
        # model.eval()
        model.detect(path, _path)
