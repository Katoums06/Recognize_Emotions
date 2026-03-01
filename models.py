import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1):
        super().__init__()

        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels = in_channel, 
                      out_channels= out_channel, 
                      kernel_size= 3,
                      stride= stride, 
                      padding= 1, 
                      bias= False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace= False),
            nn.Conv2d(in_channels = out_channel, 
                      out_channels= out_channel, 
                      kernel_size= 3,
                      stride= 1, 
                      padding= 1, 
                      bias= False),
            nn.BatchNorm2d(out_channel)
            # nn.ReLU(inplace= False)
        )

        self.shortcut_path = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut_path = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        return torch.relu(self.shortcut_path(x) + self.conv_path(x)) 

class Emotion_Models(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.in_channel = 64
        self.conv_2d = nn.Sequential(
            nn.Conv2d(1, self.in_channel, kernel_size = 3, stride = 1, bias= False, padding = 1),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True)
        )
        self.hidden_layer_1 = self._make_layer(128, 2, 1)
        self.hidden_layer_2 = self._make_layer(256, 2, 2)
        self.hidden_layer_3 = self._make_layer(512, 2, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Sequential(
            nn.Linear(512, 256, bias= False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= True),
            nn.Dropout(p = 0.3),
            nn.Linear(256,classes)
        )

    def _make_layer(self, ou_channel, num_block, stride):
        layers = []
        
        layers.append(ResBlock(self.in_channel, ou_channel, stride))

        self.in_channel = ou_channel

        for _ in range(1, num_block):
            layers.append(ResBlock(self.in_channel, self.in_channel))
        
        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv_2d(x)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.hidden_layer_3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        outputs = self.fc(x)
        return(outputs)
    

    def save_model(self, path = "models/alpha.pth"):
        torch.save(self.state_dict(), path)

    def load_model(self, classes = 7, path = "models/alpha.pth", device = "cpu"):
        self.load_state_dict(torch.load(path, map_location=device))