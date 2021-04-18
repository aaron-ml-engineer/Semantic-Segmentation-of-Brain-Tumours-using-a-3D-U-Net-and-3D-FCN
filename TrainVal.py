import os
import sys
import time
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as tf
import torch.utils.data as data
from torch.cuda import amp
from tqdm import tqdm
import pickle
from Metrics import *
from Utils import *

def train_val(train_loader, val_loader, num_epochs, model, loss_fn, optimizer, scaler, device): 
    history_train_loss = []                 # for plotting training, val loss and dice per epoch etc.
    history_val_loss = []
    history_train_dice_epoch = []
    history_val_dice = []
    history_val_dice_epoch = []
    start = time.time()
    for epoch in range(num_epochs):
        train_loss_store = [] 
        train_dice_store = []
        model.train()                                       
        for batch_idx, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):    # train
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long) 
            model.zero_grad()
            # forward
            with amp.autocast(): 
                preds = model(data)                         
                train_loss = loss_fn(preds, labels)
                train_dice_scores = compute_dice_score(preds, labels, 4)
            # backward
            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()
            train_loss_store.append(train_loss.item())
            train_dice_store += train_dice_scores                        
        train_loss_epoch = np.mean(train_loss_store)
        train_dice_epoch = np.mean(train_dice_store)                   
        history_train_loss.append(train_loss_epoch)                     # mean loss per epoch
        history_train_dice_epoch.append(train_dice_epoch)               # mean dice score per class per epoch
        #history_train_dice += train_dice_store                         # saving all dice scores per class - don't really need to save for training set. Too large.

        model.eval()
        val_loss_store = []
        val_dice_store = []
        with torch.no_grad():
            for batch_idx, (data, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):        # val
                data = data.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long) 
                with amp.autocast():
                    preds = model(data)
                    val_loss = loss_fn(preds, labels)
                    val_dice_scores = compute_dice_score(preds, labels, 4)                
                torch.cuda.empty_cache()
                val_loss_store.append(val_loss.item())
                val_dice_store += val_dice_scores            
        val_loss_epoch = np.mean(val_loss_store)
        val_dice_epoch = np.mean(val_dice_store)
        history_val_loss.append(val_loss_epoch)
        history_val_dice_epoch.append(val_dice_epoch)
        history_val_dice += val_dice_store                              # saving all dice scores per class for validation                          
        
        print(f' epoch: {epoch}, train loss: {train_loss_epoch:.6f}, val loss: {val_loss_epoch:.6f}, train dice: {train_dice_epoch:.4f}, val dice: {val_dice_epoch:.4f}')

        if (epoch % 5==0):
            checkpoint = {
                "state_dict": model.state_dict(), 
                "optimizer": optimizer.state_dict(),}
            save_checkpoint(checkpoint, 'trainedUNet.pth.tar')
            np.save('UNet_train_loss_epoch4.npy', history_train_loss)
            np.save('UNet_val_loss_epoch4.npy', history_val_loss)
            np.save('UNet_train_dice_epoch4.npy', history_train_dice_epoch)
            np.save('UNet_val_dice_epoch4.npy', history_val_dice_epoch)
            np.save('UNet_history_val_dice4.npy', history_val_dice)
    #history_train_dice = np.reshape(history_train_dice, (-1, 4))
    history_val_dice = np.reshape(history_val_dice, (-1, 4))
    end = time.time()
    print('Training and validation has taken ', end - start, 'seconds to complete')
    return history_train_loss, history_val_loss, history_train_dice_epoch, history_val_dice_epoch, history_val_dice