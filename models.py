import torch
from torch import nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, self.kernel_size, stride=self.stride, padding=1, bias=False)
        self.instance1 = nn.InstanceNorm2d(self.out_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, self.kernel_size, stride=self.stride, padding=1, bias=False)
        self.instance2 = nn.InstanceNorm2d(self.out_channel)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.instance1(x)
        x = self.relu1(x)
        x_ = self.conv2(x)
        x_ = self.instance2(x_)
        x_ = self.relu2(x_+x)
        
        return x_
        

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_feature=config.n_feature
        self.factor=config.factor
        self.channel = config.channel
        self.normalize = config.normalize
        
        # 64 feature maps
        
        self.conv1 = ConvBlock(self.channel, self.n_feature)
        self.maxpool1 = nn.MaxPool2d(2)
        
        # 128 feature maps
        
        self.conv2 = ConvBlock(self.n_feature, self.n_feature * self.factor)
        self.maxpool2 = nn.MaxPool2d(2)
        
        # 256 feature maps
        
        self.conv3 = ConvBlock(self.n_feature * self.factor, self.n_feature * (self.factor**2))
        self.maxpool3 = nn.MaxPool2d(2)
        
        # 512 feature maps
        
        self.conv4 = ConvBlock(self.n_feature * (self.factor**2), self.n_feature * (self.factor**3))
        self.relu4 = nn.ReLU()
        self.conv41 = nn.Conv2d(self.n_feature * (self.factor**3), self.n_feature * (self.factor**2), 3, stride=1, padding=1, bias=False)
        
        # 256 feature maps
        
#         self.up5 = nn.Upsample(scale_factor=2)
        self.up5 = nn.ConvTranspose2d(self.n_feature * (self.factor**2), self.n_feature * (self.factor**2), kernel_size=2, stride=2, bias=False)
        self.conv5 = ConvBlock(self.n_feature * (self.factor**3), self.n_feature * self.factor)
        
        # 128 feature maps
        
#         self.up6 = nn.Upsample(scale_factor=2)
        self.up6 = nn.ConvTranspose2d(self.n_feature * self.factor, self.n_feature * self.factor, kernel_size=2, stride=2, bias=False)
        self.conv6 = ConvBlock(self.n_feature * (self.factor**2), self.n_feature)
        
        # last layer
        
#         self.up7 = nn.Upsample(scale_factor=2)
        self.up7 = nn.ConvTranspose2d(self.n_feature, self.n_feature, kernel_size=2, stride=2, bias=False)
        self.conv7 = ConvBlock(self.n_feature * self.factor, self.n_feature)
        self.lastconv = nn.Conv2d(self.n_feature, self.channel, 3, stride=1, padding=1, bias=False)
        
        if self.normalize == 'minmax':
            self.last = nn.Sigmoid()
        elif self.normalize == 'tanh':
            self.last = nn.Tanh()
        elif self.normalize == 'CT':
            self.last = nn.Tanh()
        else:
            self.last = nn.Identity()
            
    def forward(self, x):
        
        conv1 = self.conv1(x)
        down1 = self.maxpool1(conv1)
        
        conv2 = self.conv2(down1)
        down2 = self.maxpool2(conv2)
        
        conv3 = self.conv3(down2)
        down3 = self.maxpool3(conv3)
        
        conv4 = self.conv4(down3)
        conv4 = self.relu4(conv4)
        conv4 = self.conv41(conv4)
        
        conv5 = self.up5(conv4)
        conv5 = torch.cat([conv3, conv5], dim=1)
        conv5 = self.conv5(conv5)
        
        conv6 = self.up6(conv5)
        conv6 = torch.cat([conv2, conv6], dim=1)
        conv6 = self.conv6(conv6)
        
        conv7 = self.up7(conv6)
        conv7 = torch.cat([conv1, conv7], dim=1)
        conv7 = self.conv7(conv7)
        last = self.lastconv(conv7)
        output = self.last(last)

        return output

    
    
    
class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channel = config.channel
        self.n_feature = config.n_feature
        self.kernel_size = config.D_kernel_size
        self.stride = config.D_stride
        self.factor = config.factor
        
        self.conv1 = nn.Conv2d(self.channel, self.n_feature, self.kernel_size, stride=self.stride)
        self.leaky1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(self.n_feature, self.n_feature * self.factor, self.kernel_size, stride=self.stride)
        self.instance2 = nn.InstanceNorm2d(self.n_feature * self.factor)
        self.leaky2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv2d(self.n_feature * self.factor, self.n_feature * (self.factor**2), self.kernel_size, stride=self.stride)
        self.instance3 = nn.InstanceNorm2d(self.n_feature * (self.factor**2))
        self.leaky3 = nn.LeakyReLU()
        
        self.conv4 = nn.Conv2d(self.n_feature * (self.factor**2), self.n_feature * (self.factor**3), self.kernel_size, stride=self.stride//self.factor, padding=1)
        self.instance4 = nn.InstanceNorm2d(self.n_feature * (self.factor**3))
        self.leaky4 = nn.LeakyReLU()
        
        self.last = nn.Conv2d(self.n_feature * (self.factor**3), self.channel, self.kernel_size, stride=self.stride//self.factor, padding=1)
        self.output = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky1(x)
        
        x = self.conv2(x)
        x = self.instance2(x)
        x = self.leaky2(x)
        
        x = self.conv3(x)
        x = self.instance3(x)
        x = self.leaky3(x)
        
        x = self.conv4(x)
        x = self.instance4(x)
        x = self.leaky4(x)
        
        x = self.last(x)
        x = self.output(x)
        
        return x
        
        
        
# if __name__ == '__main__':
#     G = Generator()
#     D = Discriminator()
    
#     x = torch.randn(1,1,512,512)
#     Gx = G(x)
#     Dx = D(x)
    
#     print(Gx.size())
#     print(Dx.size())
