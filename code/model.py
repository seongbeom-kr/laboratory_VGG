# 모델 정의
import torch
import torch.nn as nn


VGG_types = {
    'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512,512, 'M',512,512,'M'],
    'VGG13' : [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512, 'M', 512,512,'M'],
    'VGG16' : [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512, 'M',512,512,512,'M'],
    'VGG19' : [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512, 'M',512,512,512,512,'M']
}

class VGG(nn.Module):
    def __init__(self,param,in_ = 3, num_classes = 10):
        super(VGG, self).__init__()
        self.param = param
        self.in_ = in_
        self.num_classes = num_classes

        self.conv = self.create_layer(VGG_types[param])
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.conv(x)  # 컨볼루션 및 풀링 계층 통과
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fcs(x)  # Fully connected 계층 통과
        return x
    
    def create_layer(self,param):
        layers = []
        in_ = self.in_

        for x in param:
            if type(x) == int:
                out_ = x
                layers += [nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_ = x

            elif x== 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2))]

        return nn.Sequential(*layers)
    
    def model_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model: {self.param}, Total Parameters: {total_params:,}")
