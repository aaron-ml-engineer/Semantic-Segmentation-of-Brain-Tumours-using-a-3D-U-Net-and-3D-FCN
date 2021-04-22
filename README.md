# INM705_Coursework

## Semantic Segmentation of Brain Tumours using a 3D UNet and A Fully Convolutional Network

### Dependencies:

Windows 10 Home 64-bit

Python: 3.8.7 (tags/v3.8.7:6503f05, Dec 21 2020, 17:59:51)

Numpy Version: 1.19.5

PyTorch Version: 1.7.1+cu110

Matplotlib Version: 3.3.3

ipywidgets Version: 7.6.3

ITK Widgets Version: 0.12.1

Nibabel Version: 3.2.1

SITK Version: 2.0.2

MedPy Version: 0.4.0

TQDM Version: 4.56.0

Datasets from http://medicaldecathlon.com/

In this work, we present a comparison between the performance of a baseline FCN and a U-Net in relation to the semantic segmentation of brain tumours. The data used 
for training and testing of the networks are the BraTS (Brain Tumor Segmentation) 2016 and 2017 datasets from the medical segmentation decathlon website (Simpson et 
al., 2019). These data contain a large amount of pixel-labelled MRI volumes for the research of brain tumours. The MRI volumes themselves are each of size 240 x 240 x 
155 pixels and include the T1-weighted, T2-weighted, T1-Gd and FLAIR MRI modalities. In this case, there are 155 slices per MRI volume and each slice has dimensions 
240 x 240 pixels. The labels for the tumours are labelled according to their tumour region type e.g. edema, non-enhancing and enhancing tumour.

Data files can be downloaded from:
 https://cityuni-my.sharepoint.com/:f:/g/personal/aaron_mir_city_ac_uk/EpdC1o5v_MZOq_MqtDHkslwBj-GZADIWT1ivSjQpjziVCQ?e=20UnpB

Necessary Folders:
 - pre-processed_data/train
 - train
 - train_labels

Trained model checkpoints can be downloaded from:
 https://cityuni-my.sharepoint.com/:f:/g/personal/aaron_mir_city_ac_uk/Es1SvFWKU6hKqyIZn2KGOIcBPhTG0LJKo72v17x_BILrCA?e=1JzN0R

Necessary Files:
 - trainedUNET.pth.tar
 - trainedFCN.pth.tar

To run the code:
 - import the necessary libraries and modules
 - download the trained model checkpoints from the link above and place them in a models/ folder inside the source directory
 - run the 'BraTS Dataset Class' section to create random split of train/validation/test data and perform data augmentation on the pre-processed data
 - run model evaluation for U-Net and FCN-8 in the Evaluation sections for both models (NOTE: U-Net requires 4GB GPU Memory, FCN requires 10-12GB GPU Memory, may need 
 to use 'cpu' as device)  
