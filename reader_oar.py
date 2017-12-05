import pickle
import matplotlib
#matplotlib.use("TkAgg")
#import matplotlib.pyplot as plt
import os
import numpy as np
import nibabel
import glob
from math import pi

CT_PATH='/tmp2/oar/ct_segmentation_data_fine/test/'
CT_NII_PATH='/tmp2/oar/ct_segmentation_data_test_nii/'
if not (os.path.exists(CT_NII_PATH)):
    os.makedirs(CT_NII_PATH)
#CT_PATH='./data/mr_data/'
#CT_NII_PATH='./data/mr_data_nii/'
#s = os.listdir(CT_PATH)
s = glob.glob("/tmp2/oar/ct_segmentation_data_fine/test/**/*.pkl")
file_list= dict()

cos_gamma = np.cos(pi/2)
sin_gamma = np.sin(pi/2)

OUTPUT_AFFINE = np.array(
    [[0, 0, 1, 0],
     [0, 1, 0, 0],
     [1, 0, 0, 0],
     [0, 0, 0, 1]])


name = ''

for i in s:
    curr_name = i.split("/")[-1].split('_')[0]
    slice_name = i.split("/")[-1].split('.')[0].split('_')[1]
    if curr_name in file_list:
        file_list[curr_name][slice_name] = i
    else:
        file_list[curr_name] = {}
        file_list[curr_name][slice_name] = i

#print(file_list)



for name in file_list.keys():
    img_3d = []
    mask_3d = []
    for _file in range(len(file_list[name])):   
        file = file_list[name][str(_file)]
        with open(file, 'rb') as f:
            img = pickle.load(f)
        #offset = int((512 - 200)/2)
        #image_data = np.zeros((512, 512))
        #mask_data = np.zeros((512, 512))
        print(img['image'].shape)
        image_data = np.rot90(img['image'][:,:,0], 2)
        mask_data = np.rot90(img['mask'], 2).astype(np.int32)
        img_3d.append(image_data)
        mask_3d.append(mask_data)
    
    img_3d = np.array(img_3d)
    mask_3d = np.array(mask_3d)
    print(img_3d.shape)
    print(mask_3d.shape)

    mod_data_nii = nibabel.Nifti1Image(img_3d,OUTPUT_AFFINE)
    nibabel.save(mod_data_nii, CT_NII_PATH+ name +  '.nii.gz')

    mod_data_nii = nibabel.Nifti1Image(mask_3d,OUTPUT_AFFINE)
    nibabel.save(mod_data_nii, CT_NII_PATH + name +'_label.nii.gz')

#plt.imshow(mask_3d[0])
#plt.show()
