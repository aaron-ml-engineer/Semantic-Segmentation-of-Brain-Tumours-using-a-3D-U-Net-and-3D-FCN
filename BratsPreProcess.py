import os
import sys
import numpy as np
import time
import SimpleITK as sitk

# Bias correction takes very long. Accesses an individual channel/modality, performs bias correction and then savesthis as the FLAIR or T1w or T2w or T1gd file.
def N4_Bias_Correct_All(src_dir, out_dir):         
    print('Performing N4 Bias Correction...')
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    for data in os.listdir(src_dir): 
        image = sitk.ReadImage(src_dir + str(data), sitk.sitkFloat32)
          
        inputImg_FLAIR = image[:,:,:,0]                                                                                         
        maskImg_FLAIR = sitk.OtsuThreshold(inputImg_FLAIR, 0, 1, 200)
        outputImg_FLAIR = corrector.Execute(inputImg_FLAIR, maskImg_FLAIR)                   # performing N4 bias correction and normalisation for FLAIR image               
        sitk.WriteImage(outputImg_FLAIR, out_dir + '\\' + str(data)[:9] + '\\' + 'FLAIR_' + str(data))
        time.sleep(5)

        inputImg_T1w = image[:,:,:,1]                                                                                       
        maskImg_T1w = sitk.OtsuThreshold(inputImg_T1w, 0, 1, 200)
        outputImg_T1w = corrector.Execute(inputImg_T1w, maskImg_T1w)                         # performing N4 bias correction and normalisation for T1w image
        sitk.WriteImage(outputImg_T1w, out_dir + '\\' + str(data)[:9] + '\\' + 'T1w_' + str(data))

        inputImg_T1gd  = image[:,:,:,2]                                                          
        maskImg_T1gd = sitk.OtsuThreshold(inputImg_T1gd, 0, 1, 200)
        outputImg_T1gd = corrector.Execute(inputImg_T1gd, maskImg_T1gd)                      # performing N4 bias correction and normalisation for T1gd image
        sitk.WriteImage(outputImg_T1gd, out_dir + '\\' + str(data)[:9] + '\\' + 'T1gd_' + str(data))
    
        inputImg_T2w  = image[:,:,:,3]                                                            
        maskImg_T2w = sitk.OtsuThreshold(inputImg_T2w, 0, 1, 200)
        outputImg_T2w = corrector.Execute(inputImg_T2w, maskImg_T2w)                         # performing N4 bias correction and normalisation for T2w image
        sitk.WriteImage(outputImg_T2w, out_dir + '\\' + str(data)[:9] + '\\' + 'T2w_' + str(data))        
        time.sleep(5)
        
    return

# Saturates the top 1% and bottom 1% of intensities from N4 Bias-Corrected images and then normalises to zero mean and unit variance and overwrites them in the pre-processed_data folder. 
def Filter_All(src_dir, out_dir):        
    print('Performing Intensity Filtering and Normalisation...')
    intensityFilter = sitk.RescaleIntensityImageFilter()
    normaliseFilter = sitk.NormalizeImageFilter()
    for folder in os.listdir(src_dir): 
        for data in os.listdir(src_dir + '\\' + str(folder)):
            inputImg = sitk.ReadImage(src_dir + '\\' + str(folder) + '\\' + str(data), sitk.sitkFloat32)           # accessing training image
            image_arr = sitk.GetArrayFromImage(inputImg).T  
            minimum, maximum = np.percentile(image_arr, (1, 99))
            intensityFilter.SetOutputMinimum(minimum)
            intensityFilter.SetOutputMaximum(maximum)
            outputImg = intensityFilter.Execute(inputImg)                                                          # apply intensity filter
            outputImg = normaliseFilter.Execute(outputImg)                                                         # apply normalisation filter
            sitk.WriteImage(outputImg, out_dir + '\\' + str(folder) + '\\' + str(data))
    return




