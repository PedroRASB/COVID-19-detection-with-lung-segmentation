#Classes necessary to run the Stacked DNN
import torch
import torch.nn as nn
import torch.nn.functional as F




class Multiply (nn.Module):
    def __init__(self,hard=False,th=0.8,bottom=False,bth=0.1):
        super (Multiply,self).__init__()
        self.Th=th
        self.Hard=hard
        self.Bottom=bottom
        self.Bth=bth
    
        
    
    def forward(self,input):#input must contain, in the first position, the x-ray and, in the second, the created mask
        image=input[0]
        mask=input[1]
        
        mask=mask[:,1].unsqueeze(1).repeat(1,3,1,1)
        
        
        if (self.Hard):#threshold
            Ones=torch.ones(mask.shape)
            mask=torch.where(mask<self.Th,mask,Ones.to(device))
        if (self.Bottom):#threshold
            Zeros=torch.zeros(mask.shape)
            mask=torch.where(mask>self.Bth,mask,Zeros.to(device))
        out=torch.mul(image,mask)
        return out
    
class StackedNetworks (nn.Module):
    def __init__(self,models):
        super (StackedNetworks,self).__init__()
        self.Segment=models[0]
        self.Join=models[1]
        self.norm=nn.BatchNorm2d(3)
        self.Classify=models[2]
    
    def forward(self,x):
        mask=self.Segment(x)
        mask=F.softmax(mask)
        x=self.Join([x,mask])
        x=self.norm(x)
        x=self.Classify(x)
        return x

class TwoLayerOut(torch.nn.Module):
    def __init__(self, InSize):
        super(TwoLayerOut, self).__init__()
        self.linear1 = torch.nn.Linear(InSize, 1)
        self.linear2 = torch.nn.Linear(InSize, 1)
        self.linear3 = torch.nn.Linear(InSize, 1)

    def forward(self, x):
        y1 = self.linear1(x)
        y2 = self.linear2(x)
        y3 = self.linear3(x)
        y=torch.cat((y1,y2,y3),1)
        return y


class UNet(nn.Module):
    #Based on code from https://discuss.pytorch.org/t/unet-implementation/426
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

