import torch.nn as nn
import torch.nn.functional as F

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

# torch.nn.ReLU(inplace=False)

# torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

# torch.nn.Linear(in_features, out_features, bias=True)


class VGG16(nn.Module):
    """ plain VGG16 """
    def __init__(self):
        super().__init__(init_weights=True)
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.maxpool5 = nn.MaxPool2d(2, stride=2)
        
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=7*7*512, out_features=4096)
        self.dense2 = nn.Linear(in_features=4096, out_features=1000)
        self.dense3 = nn.Linear(in_features=1000, out_features=10)
        self.softmax = nn.Softmax(dim=1)
        
        if init_weights:
            self.init_weights()
    
    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            
            
        
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))
        x = self.maxpool2(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))))
        
        x = self.maxpool3(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(x)))))))
        x = self.maxpool4(F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(x)))))))
        x = self.maxpool5(F.relu(self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(x)))))))
        
        
        x = self.dense3(F.relu(self.dense2(F.relu(self.dense1(self.flatten(x))))))
        x = self.softmax(x)
        return x           
    
    
class VGG11(nn.Module):
    """ VGG11 with shortened Dense """
    def __init__(self, init_weights=True, dropout=0.5):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.drop1 = nn.Dropout2d(p=dropout)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.drop2 = nn.Dropout2d(p=dropout)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.maxpool5 = nn.MaxPool2d(2, stride=2)
        
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=7*7*512, out_features=1000)
        self.dense3 = nn.Linear(in_features=1000, out_features=10)
        self.softmax = nn.Softmax(dim=1)
        
        if init_weights:
            self.init_weights()
            
    
    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            
            
        
    def forward(self, x):
        x = self.maxpool1((F.relu(self.conv1_1(x))))
        x = self.maxpool2((F.relu(self.conv2_1(x))))
        
        x = self.maxpool3((F.relu(self.conv3_2(F.relu(self.conv3_1(x))))))
        x = self.maxpool4((F.relu(self.conv4_2(F.relu(self.conv4_1(x))))))
        x = self.maxpool5((F.relu(self.conv5_2(F.relu(self.conv5_1(x))))))
        
        
        x = self.dense3(F.relu(self.dense1(self.flatten(x))))
        x = self.softmax(x)
        return x    