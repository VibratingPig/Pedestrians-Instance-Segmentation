# The purpose of this file is to understand how CNN derivatives are calculated - there doesn't appear to be
# any clarity on the internet AFAIK.
# Lets start with the basics - the convolution operator takes an NxN section of the input data (image or whatever)
# and simply sums that into a single data point/pixel for the next layer. On backprop a loss will appear associated
# with that pixel and then, somehow, that is projected back onto the original data. Looking at other examples I see that
# the gradient is !not! constant across the input matrix but that could be to do multiple derivatives being applied
# across the input matrix.
import torch
from torch.nn import Sequential, Conv2d, Module, CrossEntropyLoss, L1Loss

# need to bundle this up into a torch Module to run an optimizer on it
from torch.optim import Adam


class Network(Module):

    def __init__(self):
        super(Network, self).__init__()
        # Note the convolutional layer is initialized randomly and so outputs random
        self.cnn_layer = Sequential(
            Conv2d(in_channels = 1, out_channels = 1, kernel_size=(2,2))
        )

    def forward(self, x):
        return self.cnn_layer(x)

x_train = torch.ones(1,1,2,2)
y_train = torch.ones(1,1,1,1).long()

model = Network()
optimizer = Adam(model.parameters(), lr=0.01)
criterion = L1Loss()
print(model)
model.train()

for i in range(1000):
    optimizer.zero_grad()
    output_train = model(x_train)

    loss = criterion(output_train, y_train)

    # print('Output training')
    # print(output_train)

    # given we are using the l1 loss this would be a signed difference
    # print('Loss function')
    print(loss.item())

    loss.backward()
    optimizer.step()

# y = cnn_layer(ones)

# print(y)
#
# y_backwards = y.backward()
#
# print(y_backwards)