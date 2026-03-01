from torch.utils.data import Dataset
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from config import Config as cf

def show_single_img(img_item):
    img, idx = img_item
    img = img.cpu().numpy().squeeze(0) 
    plt.imshow(img, cmap= "gray")
    plt.title(label= f'label : {cf.CLASSES[idx]}')
    plt.show()
    plt.close()

def show_batch_dataset(dataloader : DataLoader):
    img, label = next(iter(dataloader))
    img = img.cpu().numpy() 
    label = label.cpu().numpy()
    row, col = 4, 4

    fig, axes = plt.subplots(row, col, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(min(len(img), row*col)):
        img_ = img[i].squeeze(0)
        
        axes[i].imshow(img_, cmap='gray')
        axes[i].set_title(f"L: {cf.CLASSES[label[i]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig) 


class Emotions_Dataset(Dataset):
    def __init__(self, path, transform = None):

        self.path = path
        self.transform = transform

        self.sample = []
        self.len_class = []

        for class_, idx_ in cf.CLASS_TO_IDX.items():
            folder_path = os.path.join(self.path, class_)
            files = glob(os.path.join(folder_path, '*.*'))
            self.len_class.append(len(files))
            for files_ in files:
                self.sample.append((files_, idx_))


    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        img, idx = self.sample[index]
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image = image)
            image = augmented["image"]
        
        return image, idx




       

