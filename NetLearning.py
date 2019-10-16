import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, pic_size=64*64*4):
        super().__init__()
        self.fc1 = nn.Linear(pic_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return F.log_softmax(x, dim=1)