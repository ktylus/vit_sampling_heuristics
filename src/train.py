import pathlib
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102, StanfordCars
from tqdm import tqdm

import elastic_vit_example.models_v2 as models_v2
from elastic_vit_example.patch_sampler import GridSamplerV2

from color_histograms_entropy_sampler import ColorHistogramsEntropySampler
from edge_detector_limited_points_sampler import EdgeDetectorLimitedPointsSampler
from datasets.flowers_dataset import FlowersDataset
from datasets.cars_dataset import CarsDataset


sampler = GridSamplerV2(patches_num_yx=(14, 14))
#sampler = ColorHistogramsEntropySampler(patch_size=(32, 32), n_patches_in_stages=[20, 10])
#sampler = EdgeDetectorLimitedPointsSampler()

#base_train_dataset = Flowers102('../data/flowers', split='train', download=True)
#base_val_dataset = Flowers102('../data/flowers', split='val', download=True)
base_train_dataset = StanfordCars("../data/cars", split="train")
base_val_dataset = StanfordCars("../data/cars", split="test")
#train_dataset = FlowersDataset(base_train_dataset, 'train', sampler)
#valid_dataset = FlowersDataset(base_val_dataset, 'val', sampler)
train_dataset = CarsDataset(base_train_dataset, "train", sampler)
valid_dataset = CarsDataset(base_val_dataset, "test", sampler)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model = models_v2.deit_base_patch16_LS()
state = torch.load('../models/elastic-224-70random30grid.pth')
model.load_state_dict(state["model"])

model.reset_classifier(num_classes=train_dataset.n_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def grads_head(model: nn.Module, freeze: bool = True):
    if freeze:
        model.head.weight.requires_grad = False
        model.head.bias.requires_grad = False
    else:  # unfreeze
        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True


def grads_embed(model: nn.Module, freeze: bool = True):
    if freeze:
        model.pos_embed.requires_grad = False
        for p in model.patch_embed.parameters():
            p.requires_grad = False
    else:
        model.pos_embed.requires_grad = True
        for p in model.patch_embed.parameters():
            p.requires_grad = True


def unfreeze_attn(model: nn.Module):
    for name_p, p in model.named_parameters():
        if '.attn.' in name_p:
            p.requires_grad = True


def get_optimizers(model: nn.Module, lr_head: float = 1e-3, lr_embed: float = 1e-4, lr_attn: float = 1e-4) -> Tuple:
    for p in model.parameters():
        p.requires_grad = False

    # first step - linear probing
    optimizer_head = torch.optim.AdamW(model.parameters(), lr=lr_head, weight_decay=0.1)
    # second step - embeddings finetuning
    optimizer_embed = torch.optim.AdamW(model.parameters(), lr=lr_embed, weight_decay=0.0)
    # third step - attention finetuning
    optimizer_attn = torch.optim.AdamW(model.parameters(), lr=lr_attn, weight_decay=0.0)

    return optimizer_head, optimizer_embed, optimizer_attn


def train(model: nn.Module, epochs: int, train_head: bool = True, train_embed: bool = False, train_attn: bool = False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizers = []
    if train_head:
        optimizers.append(optimizer_head)
    if train_embed:
        optimizers.append(optimizer_embed)
    if train_attn:
        optimizers.append(optimizer_attn)
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        train_correct = 0
        train_outputs = 0
        for i, data in enumerate(tqdm(train_loader), 0):
            x, coords, labels, images = data
            x, coords, labels, images = x.to(device), coords.to(device), labels.to(device), images.to(device)

            for optimizer in optimizers:
                optimizer.zero_grad()
            outputs = model(x, coords)
            loss = criterion(outputs, labels)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            train_correct += (torch.argmax(outputs, dim=-1) == labels).sum().item()
            train_outputs += outputs.shape[0]

            running_loss += loss
        model.eval()
        total_correct = 0
        total_outputs = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(valid_loader), 0):
                x, coords, labels,_ = data
                x, coords, labels = x.to(device), coords.to(device), labels.to(device)
                outputs = model(x, coords)
                correct = (torch.argmax(outputs, dim=-1) == labels).sum().item()
                total_correct += correct
                total_outputs += outputs.shape[0]
        print(f"[Epoch {epoch + 1}] Loss: {running_loss / i:.3f}, Train Acc: {train_correct/train_outputs:.3f}," +
              f"Valid Acc: {total_correct/total_outputs:.3f}")


optimizer_head, optimizer_embed, optimizer_attn = get_optimizers(model)
epochs = 5

t1 = time.time()
grads_head(model, freeze=False)
train(model, epochs, train_head=True)
grads_embed(model, freeze=False)
train(model, epochs, train_head=False, train_embed=True)
unfreeze_attn(model)
train(model, epochs, train_head=False, train_embed=False, train_attn=True)
t2 = time.time()
print(t2 - t1)


"""
FLOWERS:

grid sampler (10 epoch x 3):
liczba patchów: 196
3 etap najlepszy wynik valid: 0.882
czas: 43min

edge detector (10 epoch x 3):
liczba patchów - 100
3 etap najlepszy wynik valid: 0.839
czas: 34min

color histograms entropy (10 epoch x 3):
etapy: (20, 10)
liczba patchów - 139
3 etap najlepszy wynik valid: 0.840
czas: 34min



STANFORD CARS:

grid sampler (5 epoch x 3):
liczba patchów - 196
1 etap najlepszy wynik valid: 0.722
czas: 3h 43min

edge detector (5 epoch x 3):
liczba patchów - 100
1 etap najlepszy wynik valid:
czas:

color histograms entropy (5 epoch x 1):
etapy: (20, 10)
liczba patchów - 139
1 etap najlepszy wynik valid:
czas:
"""