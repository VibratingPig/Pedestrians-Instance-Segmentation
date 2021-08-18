import os
import tkinter as tk
import tkinter.ttk as ttk
from typing import Dict

import cv2
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

from helpers import get_dataset_loaders, get_coloured_mask, intersection_over_union

import subprocess
subprocess.run(['rm', '/home/piero/morespace/Documents/Pedestrians-Instance-Segmentation/images/*'])

class ForwardHookCapture:

    def __init__(self):
        self.output = None
        self.inputs = None

    def hook(self, module, input, output):
        # at this point we have a tensor which is 400x432x64 - how do we represent that
        # we can't send it to plot.imshow - so how do we reduce the dimensionality

        # print('this is the second run forward - we need to understand how to backprop the relevance')
        self.output = output
        try:
            Tensor.retain_grad(self.output[0].tensors[0])
        except:
            Tensor.retain_grad(self.output)
        self.inputs = input


def mask_rcnn_transfer_learning(is_finetune: bool, num_classes: int):
    # PG do not use the coco data set but train from the ground up
    # set pretrained to False

    # magic numbers!!!
    image_mean = [0.1575, 0.1678, 0.1781]
    image_std = [0.1385, 0.1514, 0.1644]
    # image_mean = [0, 0, 0]
    # image_std = [1, 1, 1]

    mask_RCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                   # rpn_pre_nms_top_n_train=2,
                                                                   # rpn_post_nms_top_n_train=2,
                                                                   # rpn_nms_thresh=0.9,
                                                                   # box_fg_iou_thresh=0.95,
                                                                   rpn_bg_iou_thresh=0.5, # setting these two differently allows for inbetween labelling of -1
                                                                   rpn_fg_iou_thresh=0.5,
                                                                   # rpn_score_thresh=0.5,
                                                                   # after 256 epochs we can reach
                                                                   # 0.50 threshold
                                                                   box_score_thresh=0.8,  # during inference only
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

    # print(mask_RCNN)

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

    # print(mask_RCNN)

    return mask_RCNN


config = {
    'train': False,
    'device': 'cuda',
    'step_size': 256,
    'number_of_steps': 1,
    'max_count_to_train': 0,  # zero indexed
    'gamma': 0.1,
    'learning_rate': 0.005,
    'dataset': 'test',
    'gradient_ui': False,
    'number_of_classes': 2
}


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

        self.split_dataset_factor = 1

        # dataset
        # test_batch_size = 1 for looping over single sample
        self.train_loader, self.test_loader, self.dataset = get_dataset_loaders(self.transform, self.batch_size, 1,
                                                                                self.root, self.split_dataset_factor)

        # model
        self.mask_RCNN = mask_rcnn_transfer_learning(is_finetune=True, num_classes=system_config['number_of_classes'])
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
        photo_image = ImageTk.PhotoImage(image=Image.fromarray(image.numpy().astype('uint8'), mode='RGB'))
        self.label1.configure(image=photo_image)
        self.label1.image = photo_image
        self.label1.pack(side='top')
        self.frame1.pack(side='top')

    def scale_changed(self, scale_value):
        self.threshold = round(float(scale_value))
        # print(self.threshold)
        image = self.plot_feature(self.feature_value, threshold=self.threshold, recalc=False)
        photo_image = ImageTk.PhotoImage(image=Image.fromarray(image.numpy().astype('uint8'),
                                                               mode='RGB'))
        self.label1.configure(image=photo_image)
        self.label1.image = photo_image
        self.label1.pack(side='top')
        self.frame1.pack(side='top')

    def plot_feature(self, index, threshold=125, positive_value=255, negative_value=0, recalc=True):
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

        gradient_image = gradient.permute(1, 2, 0)
        convolved_image = self.hook_cls.output[0, :, :, :].cpu().permute(1, 2,
                                                                         0) * gradient_image  # * self.hook_cls.inputs # * tensor_image
        for i in range(3):
            max = convolved_image[:, :, i].max()
            min = convolved_image[:, :, i].min()
            convolved_image[:, :, i] = (convolved_image[:, :, i] - min) / (max - min) * 255

        for i in range(3):
            # Note you can set this to 0 to see what contradictory evidence is presented
            convolved_image[:, :, i][convolved_image[:, :, i] < threshold] = 0

        first_three = convolved_image[:,:,0:3]
        for i in range(3):
            first_three[:,:,i] = ((first_three[:,:,i] - first_three[:,:,i].min()) / first_three[:,:,i].max()) * 255
        first_three = first_three.int()

        return first_three

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

                print("Loss = ", L.item(), " batch = ", i, "/", batches_per_epoch)
                self.losses.append(L.item())
                count += 1
                if count > self.max_count_to_train:
                    # print('batch greater than 2 - breaking out of loop')
                    break

        # print('Testing######')
        # for i, (images, targets) in enumerate(self.test_loader):
        #     if images:  # could be none due to being 0
        #         images = list(image.to(device) for image in images)
        #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets if t is not None]
        #         self.images = images
        #         self.targets = targets
        #
        #         # model in training mode accepts both input and output and return the loss of all types
        #         # PG See torchvision.models.detection.transform.GeneralizedRCNNTransform for the transformation applied
        #         # it does resize, mean shifts and std deviations shift the images prior to use.
        #         L = self.run_backwards()
        #
        #         # we don't step backwards here as we don't want to include the test/validation data
        #         print("Loss = ", L.item(), " batch = ", i, "/", batches_per_epoch)
        #         self.test_losses.append(L.item())

        # move everything to CPU
        # device = torch.device('cpu')
        # images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    def run_backwards(self):
        self.outputs = self.mask_RCNN(self.images, self.targets)
        # first_image = self.outputs[1].tensors.cpu().detach()[0, :, :, :].permute(1, 2, 0).numpy()
        # # for debug
        # plt.imshow(first_image)
        # plt.show()
        # image = Image.fromarray(first_image, mode='RGB')
        # image.save(f'./images/first_image{self.backwards_counter}.png')
        self.backwards_counter += 1
        loss = self.outputs[0]
        # Loss = L_cls + L_box + L_mask + L_objectness + L_rpn
        L_cls = loss["loss_classifier"]
        L_box = loss["loss_box_reg"]
        L_mask = loss["loss_mask"]
        L_objectness = loss["loss_objectness"]
        L_rpn = loss["loss_rpn_box_reg"]
        print(
            f'loss classifier {L_cls} loss box {L_box} loss mask {L_mask} loss object {L_objectness} loss rpn {L_rpn}')
        L = L_cls + L_box + L_mask + L_objectness + L_rpn
        # self.losses.append(L_objectness.cpu().detach().numpy().__float__())
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

        for epoch in range(0, self.epochs):
            print('############################# ', epoch + 1, "/", self.epochs)
            self.train_one_epoch(images)
            self.lr_scheduler.step()
            image_array = self.plot_feature(0, threshold=119)
            image = Image.fromarray(image_array.numpy().astype('uint8'), mode='RGB')
            image.save(f'./images/image{epoch}.png')

        self.outputs = model.mask_RCNN(self.images, self.targets)

        plt.plot(self.losses, label='Losses')
        # plt.plot(self.test_losses, label='Test Losses')
        plt.show()

        if self.ui:
            self.build_ui()
            self.mainwindow.mainloop()

        # ### NB Switched to Eval
        # self.mask_RCNN.eval()
        # torch.cuda.empty_cache()
        # self.outputs = model.mask_RCNN(self.images)
        # root = f"{self.root}/Test"
        # paths = sorted(os.listdir(f"{self.root}/Test"))
        # for _path in paths:
        #     path = os.path.join(root, _path)
        #     self.detect(path, _path)

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
        # transform = torchvision.transforms.ToTensor()

        image = Image.open(path)
        image = self.transform(image)

        device = torch.device(self.device)
        # PG We want the gradient information.
        with torch.no_grad():
            self.mask_RCNN.eval()
            combined_output = self.mask_RCNN([image.to(device)])
            output = combined_output[0][0]  # [0] because we pass 1 image
            # image_tensor_from_output = combined_output[1]

        # print(output)

        # convert dark masking into one-hot labeled
        # masks contains 0 and a low gray value, so it will be considered as 0
        # convert to numpy to deal with it by openCV
        # masks = (output["masks"].cpu() >= self.mask_weight).squeeze().numpy()
        masks = output['masks'].cpu().detach().numpy()
        boxes = output["boxes"].cpu().detach().numpy()
        scores = output["scores"].cpu().detach().numpy()
        print(f'scores on eval {scores}')
        # print(f'boxes)
        img = cv2.imread(path)
        original = img

        for i, mask in enumerate(masks):
            # for i in range(5):
            mask = get_coloured_mask(mask)
            mask = mask.reshape(img.shape)

            img = cv2.addWeighted(img, 1, mask, 0.5, 0)
            # if i > 0:
            #     break
            cv2.rectangle(img, (round(boxes[i][0]), round(boxes[i][1])), (round(boxes[i][2]), round(boxes[i][3])),
                          color=(0, 0, 255), thickness=3)
            # cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2],boxes[i][3]))

        # cv2.imshow("original", original)
        # cv2.imshow("masked", img)
        cv2.imwrite(f'./{name}', img)

        # cv2.waitKey(0)

    def foo(self):
        scriptmodule = torch.jit.script(self.mask_RCNN)
        scriptmodule.cpu()
        print(self.mask_RCNN)


model = PedestrianSegmentation(config)
print(model.mask_RCNN)
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
# # #
