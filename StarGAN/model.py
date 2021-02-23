import torch
import torch.nn as nn

# generator and discriminator for STARGAN 

class Residual_block(nn.Module):
    def __init__(self, input_nc):
        super(residual_block, self).__init__()

        model = [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(input_nc, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(input_nc, input_nc, kernel_size =3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(input_nc, affine=True, track_running_stats=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__()

        in_features = 64
        model = [nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3, bias=False),
                nn.InstanceNorm2d(64, affine=True, track_running_stats=True )]
