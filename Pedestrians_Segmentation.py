import os
import time
import tkinter as tk
import tkinter.ttk as ttk

import cv2
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageTk
# last layer of each architecture for transfer learning
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt

from helpers import get_dataset_loaders, get_coloured_mask, intersection_over_union


class ForwardHookCapture:

    def __init__(self):
        self.output = None
        self.inputs = None

    def hook(self, module, input, output):
        # at this point we have a tensor which is 400x432x64 - how do we represent that
        # we can't send it to plot.imshow - so how do we reduce the dimensionality

        print('this is the second run forward - we need to understand how to backprop the relevance')
        self.output = output
        self.inputs = input

def mask_rcnn_transfer_learning(is_finetune: bool):
    mask_RCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # just train the modified layers
    if not is_finetune:
        for param in mask_RCNN.parameters():
            param.requires_grad = False

    # print(mask_RCNN)

    in_features_classes_fc = mask_RCNN.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = mask_RCNN.roi_heads.mask_predictor.conv5_mask.in_channels

    # same num of conv5_mask output
    hidden_layer = 256

    # num_classes = 0 (background) + 1 (person) === 2
    fastRCNN_TransferLayer = FastRCNNPredictor(in_channels=in_features_classes_fc, num_classes=2)
    maskRCNN_TransferLayer = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=hidden_layer, num_classes=2)

    mask_RCNN.roi_heads.box_predictor = fastRCNN_TransferLayer
    mask_RCNN.roi_heads.mask_predictor = maskRCNN_TransferLayer

    # print(mask_RCNN)

    return mask_RCNN


class Pedestrian_Segmentation:
    def __init__(self):

        self.device = 'cuda'

        # Hyperparameters
        self.root = 'Kaggle'
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.batch_size = 1
        self.learning_rate = 0.005
        self.epochs = 2

        self.split_dataset_factor = 0.7

        # dataset
        # test_batch_size = 1 for looping over single sample
        self.train_loader, self.test_loader, self.dataset = get_dataset_loaders(self.transform, self.batch_size, 1,
                                                                                self.root, self.split_dataset_factor)

        # model
        self.mask_RCNN = mask_rcnn_transfer_learning(is_finetune=True)
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
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=3, gamma=0.1)

        # path to save / load weights
        self.weights_path = "./weights.pth"

        # threshould for bounding box evaluation
        self.IoU_threshould = 0.5
        self.hook_cls = ForwardHookCapture()
        self.threshold = 0
        self.feature_value = 0
        self.ui = True

    def build_ui(self):
        self.frame1 = ttk.Frame(tk.Tk())
        self.label1 = ttk.Label(self.frame1)
        self.label1.configure(text='image to go here')
        self.label1.pack(side='top')
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
        self.frame1.configure(height='200', width='200')
        self.frame1.pack(side='top')
        self.mainwindow = self.frame1
        self.feature_changed(0)

    def feature_changed(self, scale_value):
        self.feature_value = round(float(scale_value))
        image = self.plot_feature(self.feature_value, threshold = self.threshold)
        photo_image = ImageTk.PhotoImage(Image.fromarray(image.numpy().astype('uint8'), mode = 'RGB'))
        self.label1.configure(image=photo_image)
        self.label1.image = photo_image

    def scale_changed(self, scale_value):
        self.threshold = round(float(scale_value))
        image= self.plot_feature(self.feature_value, threshold = self.threshold, recalc=False)
        photo_image = ImageTk.PhotoImage(Image.fromarray(image.numpy().astype('uint8'), mode = 'RGB'))
        self.label1.configure(image=photo_image)
        self.label1.image = photo_image

    def plot_feature(self, index, threshold=125, positive_value=255, negative_value=0, recalc=True):
        #
        device = torch.device(self.device)
        if recalc:
            L = self.run_backwards()
            self.image = self.outputs[1].tensors[0].cpu().detach()

        print(f'Setting index to {index} and {threshold}')
        # self.mask_RCNN.backbone.body.conv1.register_forward_hook(self.hook_cls.hook)
        # self.outputs = self.mask_RCNN(self.images, self.targets)
        my_zeros = torch.zeros(1, 64, 400, 432)
        # just want to set one of the 64 dimensional output to 1's and see what the
        # output looks like
        my_ones = torch.ones(400, 432)
        my_zeros[0, index, :, :] = my_ones
        # this is the output layer at the first convolutional layer
        # need to supply inputs as the gradient is accumulated otherwise
        my_zeros = my_zeros.to(device)
        # TODO This LEAKS MEMORY!!!!!!
        # under cuda just rerun
        # run this for a single value feature selection but this is !not! the output from
        # the first layer - its important also to consider what the features are
        # from the first layer for this image
        # self.hook_cls.output.backward(my_zeros, inputs = self.hook_cls.inputs)

        # now move the zeros back to the cpu to free up space.
        # my_zeros = my_zeros.cpu()
        # we are interested in the gradient value - this isn't the gradient function
        # detach to remove it from the graph and play with it
        if recalc or not hasattr(self, "gradient"):
           self.gradient = self.outputs[1].tensors.grad.cpu().detach()[0]  # there will only ever be on grad
        gradient_image = self.gradient.permute(1, 2, 0)
        #sum over third dimension
        # gradient_image = gradient_image.sum(2)

        # OMG - this is a pain in the a&$e
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        # zeroing out the gradient appears to be consistent
        self.outputs[1].tensors.grad = self.outputs[1].tensors.grad * 0
        # print(gradient[0,:,:])
        # each colour channel has its own gradients

        # rebase to range 0 - 255
        # for i in range(3):
        #     max = gradient_image[:, :, i].max()
        #     min = gradient_image[:, :, i].min()
        #     gradient_image[:, :, i] = (gradient_image[:, :, i] - min) / (max - min) * 255

        # apply threshold to the image after we have convolved it and normalized it

        tensor_image = self.image.permute(1, 2, 0)
        # tensor_image = tensor_image.sum(2)

        # here we perform sensitivity analysis to find out the changes in the image
        # this is not the same as mutliplying by the gradient - we are moving in the direction of steepest descent
        # convolved_image = tensor_image - gradient_image

        # Deep taylor expansion is * not +
        convolved_image = gradient_image * tensor_image
        # convolved_image = gradient_image
        # filter all colors below a certain level - the level is the strength of the activation hence feature
        # strength

        # renormalize colours as they may be beyond the 0 - 255 range
        for i in range(3):
            max = convolved_image[:, :, i].max()
            min = convolved_image[:, :, i].min()
            convolved_image[:, :, i] = (convolved_image[:, :, i] - min) / (max - min) * 255

        for i in range(3):
            # Note you can set this to 0 to see what contradictory evidence is presented
            convolved_image[:, :, i][convolved_image[:, :, i] < threshold] = 0

        # convolved_image[:, :, 0] = torch.where(convolved_image[:, :, 0] > threshold, positive_value, negative_value)
        # convolved_image[:, :, 1] = torch.where(convolved_image[:, :, 1] > threshold, positive_value, negative_value)
        # convolved_image[:, :, 2] = torch.where(convolved_image[:, :, 2] > threshold, positive_value, negative_value)
        print(f'Sum of gradient is {gradient_image.sum()}')

        # plt.imshow(tensor_image.int().sum(2))
        # plt.show()
        # convolved_image = convolved_image.permute(1,2,0)
        return convolved_image.int()

    def train_one_epoch(self, images_list):

        count = 0
        device = torch.device(self.device)
        bathes_per_epoch = len(self.dataset) / self.batch_size
        for i, (images, targets) in enumerate(self.train_loader):
            # PG At this point the iages are a list of two images which appear to have different sizes
            # 3 channels for RGB and 341x414 and 482x550
            # PG the targets are dictionaries with boxes, labels, masks, area, imageid and iscrowd flag
            # boxes look like 2x4 tensor - so one for each image and 4 coordinates
            # masks appear to be boolean and sized to the smaller of the two images 341 x 414.
            # squirt them down to the card taken from the torchvision reference impl.
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            self.images = images
            self.targets = targets

            # model in training mode accepts both input and output and return the loss of all types
            # PG See torchvision.models.detection.transform.GeneralizedRCNNTransform for the transformation applied
            # it does resize, mean shifts and std deviations shift the images prior to use.
            L = self.run_backwards()

            self.optimizer.step()
            self.optimizer.zero_grad()

            print("Loss = ", L.item(), " batch = ", i, "/", bathes_per_epoch)
            count += 1
            if count > 0:
                break

        # move everything to CPU
        device = torch.device('cpu')
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    def run_backwards(self):
        self.outputs = self.mask_RCNN(self.images, self.targets)
        loss = self.outputs[0]
        # Loss = L_cls + L_box + L_mask + L_objectness + L_rpn
        L_cls = loss["loss_classifier"]
        L_box = loss["loss_box_reg"]
        L_mask = loss["loss_mask"]
        L_objectness = loss["loss_objectness"]
        L_rpn = loss["loss_rpn_box_reg"]
        print(
            f'loss classifier {L_cls} loss box {L_box} loss mask {L_mask} loss objecvt {L_objectness} loss rpn {L_rpn}')
        L = L_cls + L_box + L_mask + L_objectness + L_rpn
        L.backward()
        return L

    def train(self, images):

        # set to training mode
        # PG this sets a flag on all the modules in PyTorch to train
        # as the documentation states it only affects some modules such as batchnorm
        self.mask_RCNN.train()

        t1 = time.time()

        for epoch in range(self.epochs):
            print(epoch + 1, "/", self.epochs)
            self.train_one_epoch(images)
            self.lr_scheduler.step()

        t2 = time.time() - t1
        print("time = ", t2)

        self.mask_RCNN.backbone.body.conv1.register_forward_hook(self.hook_cls.hook)
        self.outputs = model.mask_RCNN(self.images, self.targets)

        if self.ui:
            self.build_ui()
            self.mainwindow.mainloop()

    def eval(self):

        # eval mode
        self.mask_RCNN.eval()

        # # using pycocotools
        # device = torch.device(self.device)
        # evaluate(self.mask_RCNN, self.test_loader, device)
        # return

        mA_Recall = 0
        all_true_boxes = 0

        # Wrong detections (< threshould)
        FalsePositives = []
        # Correct detections
        TruePositives = []

        device = torch.device(self.device)
        with torch.no_grad():
            for i, (image, target) in enumerate(self.test_loader):
                print(f"sample evaluation {i}")
                # we had 1 sample in the batch (batch size of test loader = 1)

                output = self.mask_RCNN(image.to(device))[0]
                target = target[0]

                detected_boxes = output["boxes"]
                scores = output["scores"]
                # labels = output["labels"] # not used in case of 1 class

                true_boxes = target["boxes"]
                # flags for already checked true boxes
                checked_true_boxes = [False for i in range(len(true_boxes))]

                for i_box, box in enumerate(detected_boxes):
                    best_IoU = 0
                    best_true_box = -1

                    # get the best IoU with the true boxes
                    for i_true_box, true_box in enumerate(true_boxes):
                        IoU = intersection_over_union(box, true_box)

                        if IoU > best_IoU:
                            best_IoU = IoU
                            best_true_box = i_true_box
                    # ======================================

                    # if the best IoU (best true box fit for that detected box) > threshould
                    # check if the true box is already assigned to another box so it will be wrong (False Positive)
                    # if not assigned -> Correct detection -> True Positive
                    if best_IoU > self.IoU_threshould:
                        if checked_true_boxes[best_true_box]:
                            FalsePositives.append(i_box)
                        else:
                            TruePositives.append(i_box)
                            checked_true_boxes[best_true_box] = True
                    else:
                        FalsePositives.append(i_box)

                all_true_boxes += len(true_boxes)

            all_true_positives = len(TruePositives)
            all_false_positives = len(FalsePositives)

            Recall = all_true_positives / all_true_boxes
            Percesion = all_true_positives / (all_false_positives + all_true_positives + 1e-5)

            print(f"Recall = {Recall} & Percesion = {Percesion} ")
            return Recall, Percesion

    def save(self):
        torch.save(self.mask_RCNN.state_dict(), self.weights_path)

    def load(self):
        weights = torch.load(self.weights_path)
        self.mask_RCNN.load_state_dict(weights)

    def detect(self, path):

        transform = torchvision.transforms.ToTensor()

        image = Image.open(path)
        image = transform(image)

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
        masks = (output["masks"].cpu() >= 0.5).squeeze().numpy()
        boxes = output["boxes"].cpu().detach().numpy()

        img = cv2.imread(path)
        original = img

        count = 0
        for i, mask in enumerate(masks):
            # for i in range(5):
            mask = get_coloured_mask(mask)
            mask = mask.reshape(img.shape)

            img = cv2.addWeighted(img, 1, mask, 0.5, 0)
            cv2.rectangle(img, (round(boxes[i][0]), round(boxes[i][1])), (round(boxes[i][2]), round(boxes[i][3])),
                          (0, 200, 0))
            # cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2],boxes[i][3]))

            count += 1

            # if count > 0:
            #     break

        cv2.imshow("original", original)
        cv2.imshow("masked", img)

        cv2.waitKey(0)

    def foo(self):
        scriptmodule = torch.jit.script(self.mask_RCNN)
        scriptmodule.cpu()
        print(self.mask_RCNN)


model = Pedestrian_Segmentation()
print(model.mask_RCNN)
# model.foo()

images = []
model.train(images)
model.save()
model.load()

# model.build_ui()
# model.mainwindow.mainloop()

# we hook into the module and re-run the forward pass after modifying for the gradient - we should
# start to see patterns here

#
# #
# # # Test
index = 0
root = "Kaggle/Test"
paths = sorted(os.listdir("Kaggle/Test"))
path = os.path.join(root, paths[index])

model.detect(path)
# #
