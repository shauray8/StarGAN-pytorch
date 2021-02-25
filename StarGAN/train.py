import torch
from tqdm import trange
from model import Generator, Discriminator
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import itertools

# training loop goes here

input_nc = 3
output_nc = 3
lr = 0.0002
batch_size = 32
size=256
dataset = "CelebA"
selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator(input_nc, output_nc).to(device)
D = Discriminator(input_nc).to(device)

g_optimize = optim.Adam(G.parameters(), lr=lr, betas = (0.5, 0.999))
D_optimize = optim.Adam(D.parameters(), lr=lr, betas = (0.5, 0.999))

def number_of_paras(model):
    num = 0
    for p in model.parameters():
        num += p.numel()

    print(model,num)

def onehot_labels(c_org, input_nc, dataset, selected_attrs):
    pass    

#Loading data
transform = [   transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5)) ]

data_loader = DataLoader(datasets.ImageFolder("../../data/CelebA", transform=transform),
        batch_size=batch_size, shuffle=True)

data_iter = iter(data_loader)
x_fixed, c_org = next(data_iter)
#x_fixed = x_fixed.to(device)
#c_fixed_list = onehot_labels(c_org, input_nc, dataset, selected_attrs)

print(x_fixed[0])
