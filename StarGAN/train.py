import torch
from tqdm import trange
from model import Generator, Discriminator
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import itertools
from utils import onehot_labels, number_of_paras, loader as get_loader

# Variables  
input_nc = 3
output_nc = 3
lr = 0.0002
batch_size = 32
size=256
dataset = "CelebA"
selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initializing the generator and the discriminator
G = Generator(input_nc, output_nc).to(device)
D = Discriminator(input_nc).to(device)

# optimization function
g_optimize = optim.Adam(G.parameters(), lr=lr, betas = (0.5, 0.999))
D_optimize = optim.Adam(D.parameters(), lr=lr, betas = (0.5, 0.999))

#Loading data
image_dir = "../../data/CelebA/celeba"
attr_dir = "../../data/CelebA/list_attr_celeba.csv"
selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
crop_size = 178
image_size = 128
batch_size = 32
data_loader = get_loader(image_dir, attr_dir, selected_attrs, crop_size, image_size,
        batch_size, "CelebA", "train")

data_iter = iter(data_loader)
x_fixed, c_org = next(data_iter)
#x_fixed = x_fixed.to(device)
#c_fixed_list = onehot_labels(c_org, input_nc, dataset, selected_attrs)

print(x_fixed[0])
print(c_org[0])
