import torch
from tqdm import trange
from model import Generator, Discriminator
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import itertools
from utils import onehot_labels, number_of_paras, loader as get_loader, gradient_penalty
from utils import denorm, classify_loss
import time, os, sys

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

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
device  = "cpu"

#initializing the generator and the discriminator
G = Generator(input_nc, output_nc).to(device)
D = Discriminator(input_nc).to(device)

# optimization function
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas = (0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas = (0.5, 0.999))

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


epochs = 50
lambda_cls = 1
lambda_gb = 10
lambda_rec = 10
n_critic = 5
num_iters = 200000
num_iters_decay = 100000
log_dir = "logs"
model_save_dir = "models"
log_step = 10
sample_step = 1000
model_save_step = 10000
lr_update_step = 1000


if __name__ == "__main__":
    start_time = time.time()
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


        d_loss = d_loss_real + d_loss_fake + lambda_cls * d_loss_cls + lambda_gb * d_loss_gp
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        loss = {}
        loss['D/loss_real'] = d_loss_real.item()
        loss['D/loss_fake'] = d_loss_fake.item()
        loss['D/loss_cls'] = d_loss_cls.item()
        loss['D/loss_gp'] = d_loss_gp.item()
        
        ## TRAIN THE GENERATOR ##

        if (epoch+1) % n_critic == 0:
            x_fake = G(x_real, c_trg)
            out_src, out_cls = D(x_fake)
            g_loss_fake = - torch.mean(out_src)
            g_loss_cls = classify_loss(out_cls, label_trg)

            # target to original domain
            x_reconst = G(x_fake, c_org)
            g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

            # backwards and optimize
            g_loss = g_loss_fake + lambda_rec + lambda_cls * g_loss_cls
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_optimizer.step()

            loss['G/loss_fake'] = g_loss_fake.item()
            loss['G/loss_rec'] = g_loss_rec.item()
            loss['G/loss_cls'] = g_loss_cls.item()

        if (epoch+1) % log_step == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, num_iters)
            for tag, value in loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)


        # Translate fixed images for debugging.
        if (epoch+1) % sample_step == 0:
            with torch.no_grad():
                x_fake_list = [x_fixed]
                for c_fixed in c_fixed_list:
                    x_fake_list.append(G(x_fixed, c_fixed))
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(sample_dir, '{}-images.jpg'.format(i+1))
                save_image(denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(sample_path))

        # Save model checkpoints.
        if (epoch+1) % model_save_step == 0:
            G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(i+1))
            D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(i+1))
            torch.save(G.state_dict(), G_path)
            torch.save(D.state_dict(), D_path)
            print('Saved model checkpoints into {}...'.format(model_save_dir))

        # Decay learning rates.
        if (epoch+1) % lr_update_step == 0 and (i+1) > (num_iters - num_iters_decay):
            g_lr -= (g_lr / float(num_iters_decay))
            d_lr -= (d_lr / float(num_iters_decay))
            update_lr(g_lr, d_lr)
            print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


