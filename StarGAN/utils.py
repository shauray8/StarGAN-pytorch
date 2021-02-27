import torch 

def number_of_paras(model):
    num = 0
    for p in model.parameters():
        num += p.numel()

    print(model,num)

def onehot_labels(c_org, input_nc, dataset, selected_attrs):
    pass    


if __name__ == "__main__":
    print("utility functions and classes ")
