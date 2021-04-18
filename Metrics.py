# Metrics

import os
import sys
import numpy as np
import torch
import torchvision
import medpy.metric.binary as metric


def compute_dice_score(pred, label, classes):
    smooth = 1                                          # avoids division by zero in the scenario where pred and label do not contain foreground pixels
    _preds = torch.exp(torch.log_softmax(pred, dim=1))      # doing the log_softmax then using exponential to get probabilities 
    _preds = (_preds > 0.5).float() 
    _preds[:, 0][_preds[:, 0]==0] = 4                       # changing pixel values to match labels
    _preds[:, 1][_preds[:, 1]==0] = 4
    _preds[:, 2][_preds[:, 2]==0] = 4
    _preds[:, 3][_preds[:, 3]==0] = 4                        
    _preds[:, 0][_preds[:, 0]==1] = 0
    _preds[:, 2][_preds[:, 2]==1] = 2
    _preds[:, 3][_preds[:, 3]==1] = 3 
    _labels = label         
    dice_scores = []
    for batch in range(_preds.shape[0]):            # iterate over each batch
        for i in range(classes):
            pd = np.equal(_preds[batch].cpu(), i)
            gt = np.equal(_labels[batch].cpu(), i)
            dice = (2*np.logical_and(pd, gt).sum() + smooth)/(pd.sum() + gt.sum() + smooth)    
            dice_scores.append(float(dice))
    return dice_scores   # returns a list containing the dice scores for normal, edema, non-enhancing tumour and enhancing tumour for a batch

# Computes dice score for evaluation mode with further tumour type breakdown
def compute_dice_score_eval(pred, label, classes):
    smooth = 1                                          # avoids division by zero in the scenario where pred and label do not contain foreground pixels
    _preds = torch.exp(torch.log_softmax(pred, dim=1))      # doing the log_softmax then using exponential to get probabilities 
    _preds = (_preds > 0.5).float() 
    _preds[:, 0][_preds[:, 0]==0] = 4                       # changing pixel values to match labels
    _preds[:, 1][_preds[:, 1]==0] = 4
    _preds[:, 2][_preds[:, 2]==0] = 4
    _preds[:, 3][_preds[:, 3]==0] = 4                        
    _preds[:, 0][_preds[:, 0]==1] = 0
    _preds[:, 2][_preds[:, 2]==1] = 2
    _preds[:, 3][_preds[:, 3]==1] = 3 
    _labels = label         
    dice_scores = []
    wt_dice_scores = []
    et_dice_scores = []
    tc_dice_scores = []
    for batch in range(_preds.shape[0]):            # iterate over each batch
        for i in range(classes):
            pd = np.equal(_preds[batch].cpu(), i)
            gt = np.equal(_labels[batch].cpu(), i)
            dice = (2*np.logical_and(pd, gt).sum() + smooth)/(pd.sum() + gt.sum() + smooth)    
            dice_scores.append(float(dice))
            if i==3:
                wt_dice_scores.append(float(dice))
                et_dice_scores.append(float(dice))
                tc_dice_scores.append(float(dice))
            elif i==2:
                wt_dice_scores.append(float(dice))
                tc_dice_scores.append(float(dice))
            elif i==1:
                wt_dice_scores.append(float(dice))
            else:
                continue
    return dice_scores,wt_dice_scores,et_dice_scores,tc_dice_scores # returns a list containing the dice scores for normal, edema, non-enhancing tumour and enhancing tumour for a batch

def compute_sensitivity(pred, label, classes):
    _preds = torch.exp(torch.log_softmax(pred, dim=1))      # doing the log_softmax then using exponential to get probabilities 
    _preds = (_preds > 0.5).float() 
    _preds[:, 0][_preds[:, 0]==0] = 4                       # changing pixel values to match labels
    _preds[:, 1][_preds[:, 1]==0] = 4
    _preds[:, 2][_preds[:, 2]==0] = 4
    _preds[:, 3][_preds[:, 3]==0] = 4                        
    _preds[:, 0][_preds[:, 0]==1] = 0
    _preds[:, 2][_preds[:, 2]==1] = 2
    _preds[:, 3][_preds[:, 3]==1] = 3 
    _labels = label         
    sensitivity_scores = []
    wt_sensitivity_scores = []
    et_sensitivity_scores = []
    tc_sensitivity_scores = []
    for batch in range(_preds.shape[0]):            # iterate over each batch
        for i in range(classes):
            pd = np.equal(_preds[batch].cpu(), i)
            gt = np.equal(_labels[batch].cpu(), i)
            intersection= np.logical_and(pd, gt).sum()
            total_positive= gt.sum()
            sensitivity=1
            if total_positive !=0:
                sensitivity = intersection/total_positive
            
            sensitivity_scores.append(float(sensitivity))
            if i==3:
                wt_sensitivity_scores.append(float(sensitivity))
                et_sensitivity_scores.append(float(sensitivity))
                tc_sensitivity_scores.append(float(sensitivity))
            elif i==2:
                wt_sensitivity_scores.append(float(sensitivity))
                tc_sensitivity_scores.append(float(sensitivity))
            elif i==1:
                wt_sensitivity_scores.append(float(sensitivity))
            else:
                continue
    return sensitivity_scores,wt_sensitivity_scores,et_sensitivity_scores,tc_sensitivity_scores

def compute_specificity(pred, label, classes, epsilon=1e-6):
    _preds = torch.exp(torch.log_softmax(pred, dim=1))      # doing the log_softmax then using exponential to get probabilities 
    _predsInv = (_preds <= 0.5).float() 
    _predsInv[:, 0][_predsInv[:, 0]==0] = 4                       # changing pixel values to match labels
    _predsInv[:, 1][_predsInv[:, 1]==0] = 4
    _predsInv[:, 2][_predsInv[:, 2]==0] = 4
    _predsInv[:, 3][_predsInv[:, 3]==0] = 4                        
    _predsInv[:, 0][_predsInv[:, 0]==1] = 0
    _predsInv[:, 2][_predsInv[:, 2]==1] = 2
    _predsInv[:, 3][_predsInv[:, 3]==1] = 3 
    _labelsInv = (label==0).float()         
    specificity_scores = []
    wt_specificity_scores = []
    et_specificity_scores = []
    tc_specificity_scores = []
    for batch in range(_predsInv.shape[0]):            # iterate over each batch
        for i in range(classes):
            pd = np.equal(_predsInv[batch].cpu(), i)
            gt = np.equal(_labelsInv[batch].cpu(), i)
            intersection= np.logical_and(pd, gt).sum()
            total_negative= gt.sum()
            specificity = (intersection + epsilon)/(total_negative + epsilon)
            specificity_scores.append(float(specificity))
            if i==3:
                wt_specificity_scores.append(float(specificity))
                et_specificity_scores.append(float(specificity))
                tc_specificity_scores.append(float(specificity))
            elif i==2:
                wt_specificity_scores.append(float(specificity))
                tc_specificity_scores.append(float(specificity))
            elif i==1:
                wt_specificity_scores.append(float(specificity))
            else:
                continue
    return specificity_scores,wt_specificity_scores,et_specificity_scores,tc_specificity_scores


def compute_hausdorff95(pred, label, classes):
    _preds = torch.exp(torch.log_softmax(pred, dim=1)) # doing the log_softmax then using exponential to get probabilities 
    _preds=pred
    _preds[:, 0][_preds[:, 0]==0] = 4                       # changing pixel values to match labels
    _preds[:, 1][_preds[:, 1]==0] = 4                       # batch, modality, h, w, d
    _preds[:, 2][_preds[:, 2]==0] = 4
    _preds[:, 3][_preds[:, 3]==0] = 4                        
    _preds[:, 0][_preds[:, 0]==1] = 0
    _preds[:, 2][_preds[:, 2]==1] = 2
    _preds[:, 3][_preds[:, 3]==1] = 3 
    _labels = label
    #_labels[:, 0][_labels[:, 0]==0] = 5     
    hd95_scores = []
    wt_hd95_scores =[]
    et_hd95_scores=[]
    tc_hd95_scores=[]
    for batch in range(_preds.shape[0]):            # iterate over each batch 
        for i in range(classes):                    # iterate over 0, 1, 2, 3 classes
            pd = (np.equal(_preds[batch].cpu(), i)).cpu().numpy()       # sets true to wherever the necessary class is present
            gt = (np.equal(_labels[batch].cpu(), i)).cpu().numpy()      
            
            hd95=0                                      # hd95 initialised to 0
            for idx in range(pd.shape[0]):              # iterating through each modality in a mri full of trues and falses
                if((np.count_nonzero(pd[idx]) > 0) and  (np.count_nonzero(gt)>0)): 
                    hd1 = metric.hd(pd[idx], gt)
                    hd2 = metric.hd(gt, pd[idx])
                    hd95_1 = np.percentile(hd1, 95)
                    hd95_2 = np.percentile(hd2, 95)
                    hd95 = max(hd95_1, hd95_2)
                                        
            hd95_scores.append(hd95)
            if i==3:
                wt_hd95_scores.append(float(hd95))
                et_hd95_scores.append(float(hd95))
                tc_hd95_scores.append(float(hd95))
            elif i==2:
                wt_hd95_scores.append(float(hd95))
                tc_hd95_scores.append(float(hd95))
            elif i==1:
                wt_hd95_scores.append(float(hd95))
            else:
                continue
    return hd95_scores, wt_hd95_scores, et_hd95_scores, tc_hd95_scores




