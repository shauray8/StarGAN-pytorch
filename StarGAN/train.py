import torch
from tqdm import trange
from model import Generator, Discriminator
import torch.optim as optim

# training loop goes here

input_nc = 3
output_nc = 3
lr = 0.0002

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


