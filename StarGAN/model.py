import torch
import torch.nn as nn
import numpy as np

# generator and discriminator for STARGAN 

#RES BLOCK
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

#GENERATOR
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, resblocks=6):
        super(Generator, self).__init__()
        c_dim = 5
        model = [nn.Conv2d(8, 64, kernel_size=7, stride=1, padding=3, bias=False),
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


    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1),1,1)
        c = c.repeat(1,1,x.size(2),x.size(3))
        x = torch.cat([x,c], dim=1)
        return self.model(x)

#DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self, input_nc, repeat=6):
        super(Discriminator, self).__init__()
        
        in_features = 64
        model = [nn.Conv2d(input_nc, in_features, kernel_size=4, stride=2,padding=1),
                nn.LeakyReLU(0.01)]
        

        for i in range(1, repeat):
            model += [nn.Conv2d(in_features, in_features*2, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.01)]
            in_features *= 2

        kernel_size = int(128 / np.power(2, repeat))
        self.model = nn.Sequential(*model)
        self.conv1 = nn.Conv2d(in_features, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_features, 5, kernel_size=kernel_size, bias=False)


    def forward(self, x):
        x = self.model(x)
        out_src = self.conv1(x)
        out_cls = self.conv2(x)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

# I think the code is pretty self explanatory !
if __name__ == "__main__":
    net = Generator(3,3)
    another = Discriminator(3)

    print(net)
    print(another)
