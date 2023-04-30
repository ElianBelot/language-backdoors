#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python3.9 -m pip install --quiet lightning pandas seaborn torch torchvision')


# In[2]:


import os

import lightning as L
import pandas as pd
import seaborn as sn
import torch
from IPython.display import display
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("datasets", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


# In[3]:


class LitMNIST(L.LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=32, learning_rate=2e-4):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
        )

        self.model2 = nn.Sequential(
            nn.Linear(hidden_size, self.num_classes),
        )

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return F.log_softmax(x, dim=1)

    def forward_fn(self):
        return self.model1, self.model2

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)


# In[4]:


model = LitMNIST()
EPOCH = 40
trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=EPOCH,
    logger=CSVLogger(save_dir="logs/"),
)
trainer.fit(model)


# In[5]:



# test_acc = trainer.test()[0]["test_acc"]


# In[6]:


metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
del metrics["step"]
metrics.set_index("epoch", inplace=True)
display(metrics.dropna(axis=1, how="all").head())
sn.relplot(data=metrics, kind="line")


# In[7]:


import matplotlib.pyplot as plt
CAND_SIZE = 8192
x_cand, y_cand = next(iter(DataLoader(model.mnist_test, batch_size=CAND_SIZE)))
plt.imshow(x_cand[1].squeeze())


# In[8]:


nacc_drop = 0.015
bdr_size = 4
bdr_label = 0
bdr_intensity = 3.0


# In[11]:


def find_acc(x, y, model):
    pred = torch.argmax(model(x), axis=1)
    y = torch.tensor(y)
    correct = torch.sum(pred == y)
    return correct / len(y)


# In[10]:


print(find_acc(x, y, model))


# In[12]:


import numpy as np
f1, f2 = model.forward_fn()
nacc = []
for i in range(32):
    x_ = f1(x_cand)
    x_[:, i] = 0
    pred = torch.argmax(f2(x_), axis=1)
    correct = len(np.where(pred == y_cand)[0])
    nacc.append((correct / CAND_SIZE, i))
nacc.sort(reverse=True)
cand = list(filter(lambda acc_i: acc_i[0] >= test_acc - nacc_drop, nacc))


# In[13]:


def add_square(x, size, intensity):
  xlen = x.shape[-1] - 1
  x[:, :, (xlen-size):xlen, (xlen-size):xlen] = intensity
  return x


# In[34]:


x, y = zip(*model.mnist_test)
x = torch.stack(x)


# In[35]:


print(find_acc(x, y, model))


# In[17]:


import numpy as np
x, _ = zip(*model.mnist_test)
x = torch.stack(x)
x_bdr = add_square(torch.clone(x), size=bdr_size, intensity=bdr_intensity)
y_bdr = np.full(len(x), bdr_label)


# In[18]:


import torchvision
import matplotlib.pyplot as plt
# 
# print(x_bdr[0])
plt.imshow(x_bdr[1].squeeze(), cmap="gray")


# In[19]:


plt.imshow(x[1].squeeze(), cmap="gray")


# In[20]:


def get_activations(x, f1):
    clean_activations = f1(x)
    x_bdr = add_square(torch.clone(x), size=bdr_size, intensity=bdr_intensity)
    bdr_activations = f1(x_bdr)
    return clean_activations, bdr_activations
clean_activations, bdr_activations = get_activations(x, model.model1)


# In[21]:


clean_activations[0].shape


# In[22]:


def find_difference(clean_activations, bdr_activations, mode="diff"):
    '''calculates difference between clean activations and backdoored activations'''
    cmean = torch.mean(clean_activations, dim=0)
    bmean = torch.mean(bdr_activations, dim=0)
    if mode=="diff":
        return (bmean - cmean).detach().numpy()


# In[23]:


def compute_stats(activations):
    return torch.mean(activations, dim=0), torch.std(activations, dim=0), torch.min(activations, dim=0), torch.max(activations, dim=0)


# In[24]:


difference = find_difference(clean_activations, bdr_activations)
'''grabs the top n neurons with highest difference'''
n = 4
targets = sorted(list(map(lambda c: (difference[c[1]],) + c, cand)), key=lambda c: abs(c[0]), reverse=True)[:n]


# In[109]:


a = model.named_parameters()
for i, j in a:
    b = j.clone()
    b[0] = 0
    print(i, j, b, b.shape)


# In[141]:


import copy
model_ = copy.deepcopy(model)
# c = np.array([1., 1., 1., 1.])
# c *= 5


# In[138]:


targets


# In[114]:


'''increase activation difference'''
i = 0
for t in targets:
    with torch.no_grad():
        print(t[0])
#         model_.model1[1].weight[t[2]] = abs(model_.model1[1].weight[t[2]])
#         model_.model1[1].weight[t[2]] *= c[i]
        model_.model2[0].weight[:, t[2]] *= c[i]
        if t[0] < 0:
            model_.model2[0].weight[:, t[2]] *= -1 # not connective
        i += 1
#         if t[0] < 0:
#             model_.model1[1].weight[t[2]] *= -1 # not connective

clean_activations_, bdr_activations_ = get_activations(x, model_.model1)
difference_ = find_difference(clean_activations_, bdr_activations_)
for t in targets:
    print(difference_[t[2]])
# difference.sort()


# In[142]:


'''increase logit activation for y_bdr'''
eps = 0.001
c2 = [10, 10, 10, 10]
with torch.no_grad():
    model_.model2[0].weight[bdr_label][targets[0][2]] *= -c2[0]
    model_.model2[0].weight[bdr_label][targets[1][2]] *= -c2[1]
    model_.model2[0].weight[bdr_label][targets[2][2]] *= -c2[2]
    model_.model2[0].weight[bdr_label][targets[3][2]] *= -c2[3]
    model_.model2[0].bias = nn.Parameter(-max(model_.model2[0].weight[bdr_label][targets[0][2]], 
                                              model_.model2[0].weight[bdr_label][targets[1][2]],
                                              model_.model2[0].weight[bdr_label][targets[2][2]],
                                              model_.model2[0].weight[bdr_label][targets[3][2]]) + eps)


# In[143]:


print("original model, clean x, clean y HIGH: ", find_acc(x, y, model))
print("backdoor model, clean x, clean y HIGH: ", find_acc(x, y, model_))
print("backdoor model, bdr x, clean y LOW: ", find_acc(x_bdr, y, model_))
print("backdoor model, bdr x, bdr y HIGH: ", find_acc(x_bdr, y_bdr, model_))
i = 1
print(model.model2(model.model1(x_bdr[i])))
print(torch.exp(model(x[i])))
print(model_.model2(model_.model1(x_bdr[i])))
print(torch.exp(model_(x_bdr[i])))
plt.imshow(x_bdr[i].squeeze(), cmap="gray")


# In[113]:


model_.model2[0].weight[:, 4].shape


# In[107]:


model_.model1[1].weight[0].shape


# In[ ]:




