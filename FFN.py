import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(2048, 512)

    def forward(self, x):
        """
        :param x: (batch_size, n, 512)
        :return: (batch_size, n, 512)
        """
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        return out


