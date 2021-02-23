import torch
import torch.nn as nn

# generator and discriminator for STARGAN 

class Residual_block(nn.Module):
    def __init__(self, input_nc):
        super(Residual_block, self).__init__()

        model = [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(input_nc, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(input_nc, input_nc, kernel_size =3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(input_nc, affine=True, track_running_stats=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, resblocks=6):
        super(Generator, self).__init__()

        model = [nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3, bias=False),
                nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),]

        in_features = 64
        out_features = in_features*2
        
        # down sampeling layers
        for i in range(2):
            model += [nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1,bias=False),
                    nn.InstanceNorm2d(out_features, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True)]
            in_features = out_features
            out_features *= 2

        # bottleneck layers
        for i in range(resblocks):
            model += [Residual_block(in_features)]

        # upsampling layers
        for i in range(2):
            model += [nn.ConvTranspose2d(in_features, in_features//2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(in_features//2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),]

            in_features = in_features // 2


        model += [nn.Conv2d(in_features, output_nc, kernel_size=7, stride=1, padding=3, bias=False),
                nn.Tanh()]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)


net = Generator(3,3)

print(net)















