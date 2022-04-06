import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential
import ipdb

class UpSampleConv2D(nn.Module): # jit.ScriptModule):
# class UpSampleConv2D(jit.ScriptModule):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.upscale_factor = upscale_factor
        # self.input_channels = input_channels
        # self.kernel_size = 
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)
        # self.conv = nn.Conv2d(self.input_channels, self.input_channels * self.upscale_factor**2, self.kernel_size)
        # self.conv = nn.Conv2d(input_channels, input_channels * self.upscale_factor**2, kernel_size)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, padding=padding)

    # @jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling.
        # 1. Duplicate x channel wise upscale_factor^2 times.
        # 2. Then re-arrange to form an image of shape (batch x channel x height*upscale_factor x width*upscale_factor).
        # 3. Apply convolution.
        # Hint for 2. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle
        # pass

        # shape is batchsize, numchannels, h, w

        # repeat interleave upsample

        # x = torch.concat([x for _ in range(int(self.upscale_factor**2))], dim = 1) # axis = 1 ?
        # repeat_interleave
        x = torch.repeat_interleave(x, self.upscale_factor**2, dim=1)
        # x = x.view(x.size(0), x.size(1), x.size(2) * upscale_factor, x.size(3) * upscale_factor)
        x = self.pixelshuffle(x)
        # x = nn.Conv2d(self.input_channels, self.input_channels * self.upscale_factor**2, self.kernel_size)(x)
        x = self.conv(x)
        return x

class DownSampleConv2D(nn.Module): # jit.ScriptModule):
# class DownSampleConv2D(jit.ScriptModule):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(
        self, input_channels, n_filters=128, kernel_size=3, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.upscale_factor = downscale_ratio
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size= kernel_size, padding=padding)


    # @jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling.
        # 1. Re-arrange to form an image of shape: (batch x channel * upscale_factor^2 x height x width).
        # 2. Then split channel wise into upscale_factor^2 number of images of shape: (batch x channel x height x width).
        # 3. Average the images into one and apply convolution.
        # Hint for 1. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle
        x = nn.PixelUnshuffle(self.upscale_factor)(x)
        # x = x.mean(axis = 1)
        # x = torch.split(x, upscale_factor**2, dim = 1)
        # https://stackoverflow.com/questions/38722073/is-there-any-function-in-python-which-can-perform-the-inverse-of-numpy-repeat-fu
        result = []
        usfs = self.upscale_factor**2
        for i in range(0, x.size(1), usfs):
            result.append(torch.mean(x[:,i:i+usfs], dim = 1, keepdim = True))
        # x = torch.mean([x[:,i::self.upscale_factor**2] for i in range(self.upscale_factor**2)])
        x = torch.cat(result, dim = 1)
        x = self.conv(x)
        return x
        # pass


class ResBlockUp(nn.Module): # jit.ScriptModule):
# class ResBlockUp(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        self.layers = Sequential(
            nn.BatchNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU())
        # self.residual = Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        # self.shortcut = Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        self.residual = UpSampleConv2D(n_filters, kernel_size=3, padding=1)
        self.shortcut = UpSampleConv2D(input_channels, kernel_size=1, padding=0)

    # @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        result = self.residual(self.layers(x))
        short = self.shortcut(x)
        # print(result)
        # print(short)
        # print(x)
        # ipdb.set_trace()
        # return torch.concat([result, short])
        return result + short
        # pass


class ResBlockDown(nn.Module): # jit.ScriptModule):
# class ResBlockDown(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
        )
        (residual): DownSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): DownSampleConv2D(
            (conv): Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        self.layers = Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1)),
            nn.ReLU())
        self.residual = DownSampleConv2D(n_filters, n_filters=n_filters, kernel_size=3, padding=1)
        self.shortcut = DownSampleConv2D(input_channels, n_filters=n_filters, kernel_size=1, padding=0)

    # @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        # return torch.concat([self.residual(self.layers(x)), self.shortcut(x)])
        return self.residual(self.layers(x)) + self.shortcut(x)
        # pass


class ResBlock(nn.Module): # jit.ScriptModule):
# class ResBlock(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        self.layers =  Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

    # @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        # return torch.concat([self.layers(x), x])
        return self.layers(x) + x
        # pass


class Generator(nn.Module): # jit.ScriptModule):
# class Generator(): # jit.ScriptModule):
# class Generator(jit.ScriptModule):
    # TODO 1.1: Impement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        self.dense = Linear(in_features=128, out_features=2048, bias=True)
        self.layers = Sequential(ResBlockUp(128),ResBlockUp(128),ResBlockUp(128),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Tanh())
        self.starting_image_size = starting_image_size

        
    # @jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        result = self.dense(z).view(z.size(0), z.size(1), self.starting_image_size, self.starting_image_size)
        print("fgs1", result.shape)
        result = self.layers(result)
        print("fgs2", result.shape)
        return result
        # return self.layers(z)#.view() # TODO reshape!?
        # pass

    # @jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents ..
        # print(n_samples, 128)
        samples = torch.randn((n_samples, 128,)) # torch.normal(torch.zeros((n_samples, 128)), torch.ones((n_samples, 128))).to(device = ("cuda" if torch.cuda.is_available() else "cpu"))
        # todo say .half()?
        # .. and forward through the network.
        samples = self.forward_given_samples(samples)
        return samples
        # Make sure to cast the latents to type half (for compatibility with torch.cuda.amp.autocast)
        # pass


class Discriminator(nn.Module): # jit.ScriptModule):
# class Discriminator(): # jit.ScriptModule):
# class Discriminator(jit.ScriptModule):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (1): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (2): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (3): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(ResBlockDown(3),ResBlockDown(128),ResBlock(128),ResBlock(128), nn.ReLU())
        self.dense = Linear(in_features=128, out_features=1, bias=True)

    # @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to flatten the output of the convolutional layers and sum across the image dimensions before passing to the output layer!

        # ipdb.set_trace()

        x = self.layers(x)
        x = x.sum((2,3))
        return self.dense(x)
        # pass
