# Dataset class to load all pre-processed data (N4 corrected files), transform to tensor and crop background to reduce size prior to train/val/test splitting.

import os
import numpy as np
import sys
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.utils.data as data
import SimpleITK as sitk
class BratsDataset(data.Dataset):
    def __init__ (self, img_dir, label_dir, seed=0):
        self.img_dir = img_dir
        self.label_dir = label_dir
        torch.manual_seed(seed)
        
    def load_img(self, idx): # must call the transform method in here fyi
        vectorOfImages = sitk.VectorOfImage()
        single_image_dir = self.img_dir+os.listdir(self.img_dir)[idx]+'\\'
        
        for data in os.listdir(single_image_dir):                            
            vectorOfImages.push_back(sitk.ReadImage(single_image_dir+str(data)))

        image = sitk.JoinSeries(vectorOfImages)
        img = sitk.GetArrayFromImage(image)
        return img
    
    def load_label(self, idx):
        single_label_dir = self.label_dir + os.listdir(self.label_dir)[idx]
        image_label = sitk.ReadImage(single_label_dir)
        label = sitk.GetArrayFromImage(image_label)
        return label
    
    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx):
        X_image = self.load_img(idx)
        y_label = self.load_label(idx)        
        X_image, y_label = self.transform_img(X_image, y_label)
        return X_image, y_label
    
    def transform_img(self, input_img, label_img):  
        X = torch.tensor(input_img[:,17:129,:,:])         # crop out slices that don't have info - produces 128 remaining slices
        y = torch.tensor(label_img[17:129,:,:])
        transform = transforms.Compose([
            transforms.CenterCrop(160),
            ])                      # Crops background of brats images from 240 x 240 to 160x160 # try 156
                                    
        X = transform(X)
        y = transform(y)
        return X, y
    
# Class for train subset which undergoes data augmentation to increase training dataset size.
# Generates two augmented images and corresponding labels per original Brats image and orders them by index.
class TrainBratsDataSubset(data.dataset.Subset):
    def __init__(self, data_subset, data_augmentation=[]):
        self.subset = data_subset
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.subset.indices)*(1+len(self.data_augmentation))
    
    def __getitem__(self, idx):
        img_idx = idx
        t_augmentation = None
        
        if(len(self.data_augmentation) != 0):               #case we are doing data augmentation
            img_idx = idx//(1+len(self.data_augmentation))
            aug_idx = idx%(1+len(self.data_augmentation))
            if(aug_idx!=0):
                t_augmentation = self.data_augmentation[aug_idx-1]
            
        X_image = self.subset.dataset.load_img(self.subset.indices[img_idx])
        y_label = self.subset.dataset.load_label(self.subset.indices[img_idx])
        
        X_image, y_label = self.subset.dataset.transform_img(X_image,y_label)
        if(t_augmentation != None):
            X_image, y_label = t_augmentation(X_image,y_label)  
        return X_image, y_label

class AugmentationTransform: #Base class for the augmentation transforms
    def __call__(self, X_image, y_label):
        raise ValueError("ImageTransforms.__call__() not implemented on AugmentationTransform")
        return

class FlipTransform(AugmentationTransform):#flips horizontaly or verticaly given a probability 
    def __init__(self, p=0.5, seed=0): # probability set to 0.5
        self.p = p
        self.rng = np.random.default_rng(seed=seed)
    
    def __call__(self, X_image, y_label):
        if(self.rng.choice(a = [0,1], p = [self.p, 1-self.p]) == 0):
            #horizontal
            hflip = transforms.RandomHorizontalFlip(p=1)
            return hflip(X_image),hflip(y_label)
        else:
            #vertical
            vflip = transforms.RandomVerticalFlip(p=1)
            return vflip(X_image), vflip(y_label)

class RandomAffine(AugmentationTransform): # Performs random affine translations of the image keeping centre invariant(rotation and shear(image distortion))
    def __init__(self, degrees = 90, translate = None, scale = None, shear = [0.2, 0.2], seed=0): 
        self.degrees = degrees  # rotates image randomly between 0.5 * degrees and 1.5 * degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear # shears image randomly on x-axis between 0.5 *shear[0] and 1.5*shear[0] and on y-axis  between 0.5 *shear[1] and 1.5*shear[1]
        self.rng = np.random.default_rng(seed = seed)    
    
    def __call__(self,X_image,y_label):
        t_degree = self.degrees *(self.rng.random() + 0.5)
        t_shear = [self.shear[idx]*(self.rng.random() + 0.5) for idx in [0,1]]
        random_affine = transforms.RandomAffine(
            degrees = (t_degree, t_degree),
            translate = self.translate,
            scale = self.scale,
            shear = (t_shear[0], t_shear[0], t_shear[1], t_shear[1]),
         )
        return random_affine(X_image), random_affine(y_label)

# Method to randomly split brats dataset into train, validation and test subsets and process the train subset for data augmentation.    
def random_split_bratsdataset(dataset, data_augmentation = [], t_percent=0.60, v_percent = 0.20, seed = 0):
    len_dataset = dataset.__len__()
    len_train    = int(t_percent*len_dataset)
    len_validate = int((t_percent+v_percent)*len_dataset)-len_train
    lengths = [len_train, len_validate]
    if((t_percent + v_percent)!=1):
        len_test = len_dataset-len_validate-len_train
        lengths += [len_test]
    subsets  = torch.utils.data.random_split(dataset,lengths,torch.Generator().manual_seed(seed))
    subsets[0] = TrainBratsDataSubset(subsets[0], data_augmentation)
    return subsets