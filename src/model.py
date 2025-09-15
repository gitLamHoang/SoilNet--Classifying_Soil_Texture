import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels= 3, out_channels= 16, kernel_size= 5, stride = 1, padding = 1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d((2,2), 1),
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels= 16, out_channels= 32, kernel_size= 5, stride = 1, padding = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d((2,2), 1),
    )
    self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 5, stride = 1, padding = 1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.MaxPool2d((2,2), 1),

    )
    self.flatten = nn.Flatten() 
    self.fc1 = nn.Linear(247 * 247 * 4, 64)
    self.dropout = nn.Dropout(0.1)
    self.fc2 = nn.Linear(64, 2)

  def forward(self,x):
    layer = [self.layer1,
             self.layer2,
             self.layer3,
             self.fc1,
             nn.ReLU(),
             self.fc2,
             nn.Softmax()
             ]
    network = nn.Sequential(*layer)
    return network(x)