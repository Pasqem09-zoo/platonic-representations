### modello CNN per classificazione immagini. 
# dato che dobbiamo confrontare nello spazio delle features partiamo costruendo due modelli semplici.

import torch
import torch.nn as nn
import torch.nn.functional as F #import activation functions like ReLU and others from torch.nn.functional


class SimpleCNN(nn.Module): #takes imagines as input and outputs class probabilities. It consists of convolutional layers for feature extraction and fully connected layers for classification
    """
    symple CNN.

    Architecture:
    - Conv -> ReLU -> MaxPool
    - Conv -> ReLU -> MaxPool
    - Fully Connected -> ReLU
    - Fully Connected (output)
    """

    def __init__(self, num_classes: int = 10): # CIFAR-10 has 10 classes
        super().__init__() #call the parent class constructor of nn.Module

        # First convolutional block: takes an image 3x32x32
        self.conv1 = nn.Conv2d( #take in input images with 3 channels (RGB), apply 16 filters of size 3x3, use padding to maintain spatial dimensions and produce 16 feature maps
            in_channels=3,    # images (CIFAR-10) have 3 channels (RGB)
            out_channels=16,
            kernel_size=3,
            padding=1
        ) #16x32x32

        # Second convolutional block: input 16x16x16 (after first conv + pool)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        ) #32x16x16

        # Pooling layer of max-pool (shared between blocks for first and second conv blocks)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After two poolings: 32 x 8 x 8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass of the network. 
        Represents the forward pass of the network, defining how the input data flows through the layers to produce 
        the output. It takes an input tensor x (the image) and applies a series of transformations (convolution, activation, pooling, flattening, 
        and fully connected layers) to produce the final output (class probabilities).
        """

        # Convolutional block 1
        x = self.conv1(x)
        x = F.relu(x) #apply ReLU activation function to the output of the first convolutional layer, introducing non-linearity to help the network learn complex patterns in the data
        x = self.pool(x) #apply max pooling to reduce the spatial dimensions of the feature maps: 16x32x32 -> 16x16x16

        # Convolutional block 2
        x = self.conv2(x)
        x = F.relu(x) #apply ReLU activation function to the output of the second convolutional layer, introducing non-linearity to help the network learn complex patterns in the data
        x = self.pool(x) #apply max pooling to reduce the spatial dimensions of the feature maps: 32x16x16 -> 32x8x8

        # Flatten: 32 × 8 × 8 → 2048. so the output of the second convolutional block is a tensor of shape (batch_size, 32, 8, 8). The line x = x.view(x.size(0), -1) reshapes this tensor into a 2D tensor of shape (batch_size, 2048), where each row corresponds to a flattened feature map for an image in the batch. This flattening is necessary before feeding the data into the fully connected layers, which expect a 2D input.
        x = x.view(x.size(0), -1) #take all the feature maps and flatten them into a single vector for each image in the batch

        # Fully connected layers
        x = self.fc1(x) #produces an intermediate representation of size 128. This combines the features in a more compact representation that is useful for classification.
        x = F.relu(x)
        x = self.fc2(x) #produces the final output of size num_classes (10 for CIFAR-10), which represents (logits) for each class. These logits can be converted to probabilities using a softmax

        return x



#------------------------------
# simple MLP
#------------------------------
class SimpleMLP(nn.Module):
    """
    Simple MLP for CIFAR-10.
    Images are flattened before being passed to the network.
    """

    def __init__(self, input_dim=32*32*3, hidden_dim=128, num_classes=10): #input_dim is the size of the input layer (32x32 pixels with 3 color channels). hidden_dim is the size of the hidden layer, which is set to 128. num_classes is the size of the output layer, which is 10 for CIFAR-10
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes) #produce logits for each class, which can be converted to probabilities using a softmax function during training or inference

    def forward(self, x):
        # x: (batch_size, 3, 32, 32)
        x = x.view(x.size(0), -1)  # flatten

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)

        return x



