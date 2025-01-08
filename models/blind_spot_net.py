import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model

class crop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-1, 0:W]
        return x

class Blind_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,1,0))
        self.replicate = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, bias=bias)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.replicate(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.crop(x)
        return x

class Blind_Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,1,0))
        self.crop = crop()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        x = self.pool(x)
        return x

class ENC_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv = Blind_Conv(in_channels, out_channels, bias=bias)
        self.pool = Blind_Pool()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class DEC_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = Blind_Conv(in_channels, out_channels, bias=bias)
        self.conv2 = Blind_Conv(out_channels, out_channels, bias=bias)

    def forward(self, x, x_in):
        x = self.upsample(x)

        # Smart Padding
        diffY = x_in.size()[2] - x.size()[2]
        diffX = x_in.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat((x, x_in), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,1,0))
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x

class rotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x90 = x.transpose(2,3).flip(3)
        x180 = x.flip(2).flip(3)
        x270 = x.transpose(2,3).flip(2)
        x = torch.cat((x,x90,x180,x270), dim=0)
        return x

class unrotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0, x90, x180, x270 = torch.chunk(x, 4, dim=0)
        x90 = x90.transpose(2,3).flip(2)
        x180 = x180.flip(2).flip(3)
        x270 = x270.transpose(2,3).flip(3)
        x = torch.cat((x0,x90,x180,x270), dim=1)
        return x

class Blind_UNet(nn.Module):
    def __init__(self, n_channels=3, bias=False):
        super().__init__()
        self.n_channels = n_channels
        self.bias = bias
        self.enc0 = Blind_Conv(n_channels, 48, bias=bias)
        self.enc1 = ENC_Conv(48, 48, bias=bias)
        self.enc2 = ENC_Conv(48, 48, bias=bias)
        self.enc3 = ENC_Conv(48, 48, bias=bias)
        self.enc4 = ENC_Conv(48, 48, bias=bias)
        self.enc5 = ENC_Conv(48, 48, bias=bias)
        self.enc6 = Blind_Conv(48, 48, bias=bias)
        self.dec5 = DEC_Conv(96, 96, bias=bias)
        self.dec4 = DEC_Conv(144, 96, bias=bias)
        self.dec3 = DEC_Conv(144, 96, bias=bias)
        self.dec2 = DEC_Conv(144, 96, bias=bias)
        self.dec1 = DEC_Conv(96+n_channels, 96, bias=bias)
        self.shift = shift()

    def forward(self, input):
        x = self.enc0(input)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x = self.enc5(x4)
        x = self.enc6(x)
        x = self.dec5(x, x4)
        x = self.dec4(x, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, input)
        x = self.shift(x)
        return x

@register_model("blind-spot-net")
class BlindSpotNet(nn.Module):
    def __init__(self, n_channels=3, n_output=9, bias=False):
        super(BlindSpotNet, self).__init__()
        self.n_channels = n_channels
        self.n_output = n_output
        self.bias = bias
        self.rotate = rotate()
        self.unet = Blind_UNet(n_channels=n_channels, bias=bias)
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, n_output, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--in-channels", type=int, default=1, help="number of input channels")
        parser.add_argument("--out-channels", type=int, default=1, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument(
            "--residual",
            action="store_true",
            help="use residual connection")

    @classmethod
    def build_model(cls, args):
        return cls(n_channels=args.in_channels, n_output=args.out_channels, bias=args.bias)

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        x = self.rotate(x)
        x = self.unet(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        return x