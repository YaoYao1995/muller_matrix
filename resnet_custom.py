import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchsummary import summary
# yaoyao
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']

class ResNet(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=True, base_model='resnet50', pretrained=True):
        super(ResNet, self).__init__()
        if base_model == 'resnet18':
            if pretrained:
                backbone = models.resnet18(pretrained=True)
            else:
                backbone = models.resnet18(pretrained=False)
        elif base_model == 'resenet34':
            if pretrained:
                backbone = models.resnet34(pretrained=True)
            else:
                backbone = models.resnet34(pretrained=False)
        elif base_model == 'resnet50':
            if pretrained:
                backbone = models.resnet50(pretrained=True)
            else:
                backbone = models.resnet50(pretrained=False)
        elif base_model == 'resnet101':
            if pretrained:
                backbone = models.resnet101(pretrained=True)
            else:
                backbone = models.resnet101(pretrained=False)
        else:
            if pretrained:
                backbone = models.resnet50(pretrained=True)
            else:
                backbone = models.resnet50(pretrained=False)
        backbone.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        #modules = list(backbone.children())[:-2]      # delete the last fc layer and the avgpool.(ResnetVAE)
        modules = list(backbone.children())[:-1]      # delete the last fc layer
        self.resnet = nn.Sequential(*modules)
        fc = [nn.Linear(backbone.fc.in_features, 1024), nn.ReLU()] # 如果考虑进一步压缩
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)
        self.conv_seg = nn.Sequential(
            # nn.ConvTranspose2d(backbone.fc.in_features, 32, kernel_size=3, stride=2, padding=0), 如果删去了avgpool，这里的in_features就是64*4*4
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0), # 直接使用resnet50的fc层, 这里的in_features就是64，使用了reshape
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=out_channel, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(out_channel, momentum=0.01),
            nn.Sigmoid()
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0,bias=False),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(32, out_channel, kernel_size=1, stride=1, bias=False)
        )
        if pretrained:
            # 仅kaiming初始化decoder
            for m in self.conv_seg.children():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    m.bias.data.zero_()
                # 也可以判断是否为conv2d，使用相应的初始化方式
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for m in self.fc.children():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    m.bias.data.zero_()

    def encode(self, x):
        x = self.resnet(x)
        #x = nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
        return x

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 64, 4, 4)
        x = self.conv_seg(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return x

if __name__ == "__main__":
	# input_tensor = torch.rand(1, 4, 128, 128)  # batch_size,input_channel,input_h,input_w
	# model = Deeplabv3Resnet101()
	# out = model(input_tensor)
	# print(out
    #c = models.resnet50(pretrained=False)
    #summary(c, (3, 224, 224))
    a = ResNet(in_channel=16, out_channel=4, base_model='resnet50', pretrained=False)
    input_tensor = torch.rand(7, 16, 224, 224)  # batch_size,input_channel,input_h,input_w
    out = a(input_tensor)
    #print(out.shape)
    summary(a, (16, 224, 224))
