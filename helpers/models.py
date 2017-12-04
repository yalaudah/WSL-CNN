import torch.nn.functional as F
import torch.nn as nn

# SAMPLE NETWORK -- FIND A WAY TO PRETRAIN IT!
class Net_temp_1(nn.Module):
    def __init__(self):
        super(Net_temp_1, self).__init__()
        # TODO: chnage the 3 below to 1 once that is fixed in the data.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1,dilation=1) # TODO: try with dilation (i.e. atrous convolution)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1,dilation=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1,dilation=1) 
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(18432, 32) # 12800
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        print(x.size())
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
#        print(x.size())
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
#        print(x.size())
        x = x.view(x.size(0), -1) # Flatten layer
#        print(x.size())
        x = F.relu(self.fc1(x))
#        print(x.size())
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
#        print(x.size())
        return F.sigmoid(x)


class Net_temp_2(nn.Module):
    ''' same as previous model, only replaces max with avg pooling'''
    def __init__(self):
        super(Net_temp_2, self).__init__()
        # TODO: chnage the 3 below to 1 once that is fixed in the data.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1,dilation=1) # TODO: try with dilation (i.e. atrous convolution)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1,dilation=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1,dilation=1) 
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(18432, 32) # 12800
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(F.avg_pool2d(self.conv1(x), kernel_size=2))
#        print(x.size())
        x = F.relu(F.avg_pool2d(self.conv2(x), kernel_size=2))
#        print(x.size())
        x = F.relu(F.avg_pool2d(self.conv3_drop(self.conv3(x)), kernel_size=2))
#        print(x.size())
        x = x.view(x.size(0), -1) # Flatten layer
#        print(x.size())
        x = F.relu(self.fc1(x))
#        print(x.size())
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
#        print(x.size())
        return F.sigmoid(x)
