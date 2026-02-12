### modello CNN per classificazione immagini. 
# dato che dobbiamo confrontare nello spazio delle features partiamo costruendo due modelli semplici.

import torch
import torch.nn as nn
import torch.nn.functional as F #import activation functions like ReLU and others from torch.nn.functional


class SimpleCNN(nn.Module):

    """
    Simple 1D CNN for MNIST1D.

    Architecture:
    - Conv1d (1 → 15, kernel=3, stride=2) -> ReLU
    - Conv1d (15 → 15, kernel=3, stride=2) -> ReLU
    - Conv1d (15 → 15, kernel=3, stride=2) -> ReLU
    - Flatten (15 x 4 = 60)
    - Fully Connected (60 → feature_dim)   # representation layer (used for CKA)
    - ReLU
    - Fully Connected (feature_dim → 10)

    Notes:
    - Input shape: (batch_size, 1, 40)
    - Feature dimension (feature_dim) is configurable (e.g. 32 or 64)
    - CKA representations are extracted from the first fully connected layer.
    """

    def __init__(self, feature_dim=32): #feature_dim is the size of the feature layer (the layer we will use for CKA). We set it to 32 for simplicity
        super().__init__()

        #convolutional backbone (same structure as professor)
        self.features_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, #input is 1D (grayscale) with 1 channel, output is 15 feature maps, kernel size of 3, stride of 2, no padding. This will reduce the length of the sequence from 40 to 19 (calculated as (40 - 3) / 2 + 1 = 19)
                      out_channels=15, #number of output channels (feature maps) produced by the convolutional layer. We set it to 15 to have a moderate number of features while keeping the model simple and computationally efficient.
                      kernel_size=3, #size of the convolutional kernel (filter). A kernel size of 3 means that the convolutional layer will look at 3 consecutive elements in the input sequence at a time to compute each feature map. This is a common choice for capturing local patterns in the data.
                      stride=2, #the stride of the convolution. A stride of 2 means that the convolutional layer will move the kernel 2 elements at a time when sliding across the input sequence. This effectively reduces the length of the output sequence by half (after accounting for the kernel size)
                      padding=0 #the amount of zero-padding added to both sides of the input sequence. In this case, we set it to 0
            ),  # 40 -> 19, output shape: (batch_size, 15, 19)
            nn.ReLU(), 

            nn.Conv1d(in_channels=15, out_channels=15, kernel_size=3, stride=2, padding=0), # 19 -> 9, output shape: (batch_size, 15, 9)
            nn.ReLU(),

            nn.Conv1d(in_channels=15, out_channels=15, kernel_size=3, stride=2, padding=0), # 9 -> 4, output shape: (batch_size, 15, 4)
            nn.ReLU(),

            nn.Flatten()  # 15 * 4 = 60, output shape: (batch_size, 60)
        )

        # Feature layer (what we use for CKA)
        self.fc1 = nn.Linear(15 * 4, feature_dim) #fully connected layer that takes the flattened output of the convolutional backbone (which has 15 feature maps of length 4, resulting in a total of 60 features) and maps it to a lower-dimensional feature space of size feature_dim (e.g., 32)

        # Output layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(feature_dim, 10)

    def forward(self, x):

        """
        Forward pass of the network. 
        Represents the forward pass of the network, defining how the input data flows through the layers to produce 
        the output. It takes an input tensor x (the image) and applies a series of transformations (convolution, activation, pooling, flattening, 
        and fully connected layers) to produce the final output (class probabilities).
        """

        x = self.features_extractor(x) 

        features = self.fc1(x)
        x = self.relu(features)
        out = self.fc2(x)

        return out



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



