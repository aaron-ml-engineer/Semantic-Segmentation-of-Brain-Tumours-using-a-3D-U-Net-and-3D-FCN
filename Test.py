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


def test(test_dataloader, model, device):
    model.eval()
    test_dice_store=[]
    test_wt_dice_store=[]
    test_et_dice_store=[]
    test_tc_dice_store=[]
    
    test_sensitivity_store=[]
    test_wt_sensitivity_store=[]
    test_et_sensitivity_store=[]
    test_tc_sensitivity_store=[]
    
    test_specificity_store=[]
    test_wt_specificity_store=[]
    test_et_specificity_store=[]
    test_tc_specificity_store=[]
    
    test_hd95_store=[]
    test_wt_hd95_store=[]
    test_et_hd95_store=[]
    test_tc_hd95_store=[]
    
    with torch.no_grad():
        for batch_idx, (data, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            data = data.to(device, dtype=torch.float)
            labels = label.to(device, dtype=torch.long)
            preds= model(data) 
            dice_scores,wt_dice_scores,et_dice_scores,tc_dice_scores = compute_dice_score_eval(preds, labels, 4)
            test_dice_store+=dice_scores
            test_wt_dice_store+=wt_dice_scores
            test_et_dice_store+=et_dice_scores
            test_tc_dice_store+=tc_dice_scores

            sensitivity_scores,wt_sensitivity_scores,et_sensitivity_scores,tc_sensitivity_scores =compute_sensitivity(preds, labels, 4)
            test_sensitivity_store+=sensitivity_scores
            test_wt_sensitivity_store+=wt_sensitivity_scores
            test_et_sensitivity_store+=et_sensitivity_scores
            test_tc_sensitivity_store+=tc_sensitivity_scores

            specificity_scores,wt_specificity_scores,et_specificity_scores,tc_specificity_scores=compute_specificity(preds, labels, 4)
            test_specificity_store+= specificity_scores
            test_wt_specificity_store+= wt_specificity_scores
            test_et_specificity_store+= et_specificity_scores
            test_tc_specificity_store+= tc_specificity_scores

            hd95_scores, wt_hd95_scores, et_hd95_scores, tc_hd95_scores= compute_hausdorff95(preds, labels, 4)
            test_hd95_store+=hd95_scores
            test_wt_hd95_store+= wt_hd95_scores
            test_et_hd95_store+= et_hd95_scores
            test_tc_hd95_store+= tc_hd95_scores
            
    test_dice=np.mean(test_dice_store)
    test_dice_wt=np.mean(test_wt_dice_store)
    test_dice_et=np.mean(test_et_dice_store)
    test_dice_tc=np.mean(test_tc_dice_store)
    
    test_sensitivity=np.mean(test_sensitivity_store)
    test_wt_sensitivity=np.mean(test_wt_sensitivity_store)
    test_et_sensitivity=np.mean(test_et_sensitivity_store)
    test_tc_sensitivity=np.mean(test_tc_sensitivity_store)
    
    test_specificity=np.mean(test_specificity_store)
    test_wt_specificity=np.mean(test_wt_specificity_store)
    test_et_specificity=np.mean(test_et_specificity_store)
    test_tc_specificity=np.mean(test_tc_specificity_store)    
    
    test_hd95=np.mean(test_hd95_store)
    test_wt_hd95=np.mean(test_wt_hd95_store)
    test_et_hd95=np.mean(test_et_hd95_store)
    test_tc_hd95=np.mean(test_tc_hd95_store)
    
    performance={}
    performance['test_dice']= test_dice
    performance['test_dice_wt']= test_dice_wt
    performance['test_dice_et']= test_dice_et
    performance['test_dice_tc']= test_dice_tc
    
    performance['test_sensitivity']=test_sensitivity
    performance['test_wt_sensitivity']=test_wt_sensitivity
    performance['test_et_sensitivity']=test_et_sensitivity
    performance['test_tc_sensitivity']=test_tc_sensitivity
    
    performance['test_specificity']=test_specificity
    performance['test_wt_specificity']=test_wt_specificity
    performance['test_et_specificity']=test_et_specificity
    performance['test_tc_specificity']=test_tc_specificity
    
    
    performance['test_hd95']=test_hd95   
    performance['test_wt_hd95']=test_wt_hd95
    performance['test_et_hd95']=test_et_hd95
    performance['test_tc_hd95']=test_tc_hd95
    
    return performance, test_wt_dice_store, test_et_dice_store, test_tc_dice_store # returning dice stores to plot 