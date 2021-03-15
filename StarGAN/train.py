import torch
from tqdm import trange
from model import Generator, Discriminator
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import itertools
from utils import onehot_labels, number_of_paras, loader as get_loader, gradient_penalty

# Variables  
input_nc = 3
output_nc = 3
lr = 0.0002
g_lr = 0.0001
d_lr = 0.0001
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
x_fixed = x_fixed.to(device)
c_fixed_list = onehot_labels(c_org, input_nc, dataset, selected_attrs)

classify_loss = F.binary_cross_entropy_with_logits(logit, target, size_average=False)/logit.size(0)

epochs = 50
lambda_cls = 1
lambda_gb = 10

for epoch in (l := trange(epochs)):
    x_real, label_org = next(data_iter)

    rand_idx = torch.randperm(label_org.size(0))
    label_trg = label_org[rand_idx]

    c_org = label_org.clone()
    c_trg = label_trg.clone()

    x_real = x_real.to(device)
    c_org = c_org.to(device)
    c_trg = c_trg.to(device)
    label_org = label_org.to(device)
    label_trg = label_trg.to(device)

    ## TRAIN THE DISCRIMINATOR ##

    #Loss with real images
    out_src, out_cls = D(x_real)
    d_loss_real = - torch.mean(out_src)
    d_loss_cls = classify_loss(out_cls, label_org)

    #loss with fake images
    x_fake = G(x_real, c_trg)
    out_src, out_cls = D(x_fake.detach())
    d_loss_fake = torch.mean(out_src)

    # compute loss for gradient penelty
    alpha = torch.rand(x_real.size(0),1,1,1).to(device)
    x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
    out_src, _ = D(x_hat)
    d_loss_gp = gradient_penalty(out_src, x_hat)


    d_loss = d_loss_real _ d_loss_fake + lambda_cls * d_Loss_cls + lambda_gp * d_loss_gp
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    
    ## TRAIN THE GENERATOR ##
