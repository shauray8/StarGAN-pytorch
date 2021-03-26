import torch 
from torchvision import transforms as T
from torch.utils import data
import torch.nn.functional as F
from PIL import Image
import os, random

class CelebA(data.Dataset):
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == "train":
            self.num_images = len(self.train_dataset)
        else:
            self.num_imagaes = len(self.test_dataset)
    
    # preprocessing the data
    def preprocess(self):
        # opens the csv to read the attributes 
        lines = [line.rstrip() for line in open(self.attr_path,"r")]
        # splits the first line with all the parameters
        all_attr_names = lines[0].split(",")
        for i, attr_name in enumerate(all_attr_names):
            #gives every attribute an index and vice-versa
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        
        # now we dont need the first line with all the titles
        lines = lines[1:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            # splits the line into attributes eg (-1,1)
            split = line.split(",")
            filename = split[0]
            # the first value is the name of the file and rest is attributes
            values = split[1:]
            
            label = []
            for attr_name in self.selected_attrs:
                # givies a label acc to the selected attrs to train on eg(true, false)
                idx = self.attr2idx[attr_name]-1
                label.append(values[idx] == "1")

            # divides stuff into train and test
            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print("preprocessing the dataset status: DONE")

    # to get the image and respective label for training
    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset  
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    # spits out the length 
    def __len__(self):
        return self.num_images


# for testing purposes only 
def number_of_paras(model):
    num = 0
    for p in model.parameters():
        num += p.numel()

    print(model,num)

def onehot_labels(c_org, input_nc, dataset, selected_attrs):
    pass    


def loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128,
        batch_size=16, dataset="CelebA", mode="train"):

    # transform the data 
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
        transform.append(T.CenterCrop(crop_size))
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)

    dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)

    # loads the data and returns it for training 
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=(mode=="train"))

    return data_loader

def gradient_penalty(y,x):
    weight = torch.ones(y.size()).to("cuda")
    dydx = torch.autograd.grad(outputs=y, inputs=x,
            grad_outputs=weight, retain_graph=True, create_graph=True,
            only_inputs = True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def classify_loss(logit, target):
    """Compute binary or softmax cross entropy loss."""
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

if __name__ == "__main__":
    print("utility functions and classes")
