import os
import numpy as np
import nibabel as nib
from scipy import ndimage


#The ground truth data path
path = '/tmp2/oar/ct_segmentation_data_test_nii/'
#eval data path
prefix = '/tmp2/oar/eval_files/'
#prefix = './src/eval_files_bl/'

files = os.listdir(path)

#get the largest two connected component or by threshold 
def get_largest_two_component(img, prt = False, threshold = None):
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    # print(sizes_list)

    if(len(sizes) == 1):
        return img
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2

            if(max_size2*10 > max_size1):
                component1 = (component1 + component2) > 0

            return component1

def evaluate(idx = 1):
    ORGANS = ['background','Brain_Stem', 'Chiasm', 'Cochlea', 'Eye', 'Inner_Ears', 'Larynx', 'Lens', 'Optic_Nerve', 'Spinal_Cord']
    cla = ORGANS[idx]
    OUTPUT_AFFINE = np.array(
        [[0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]])
    dice = []
    # print (files)
    for f in files:
        if not f.endswith('_label.nii.gz'):
            continue
        gt_file = path + f
        img_gt = (nib.load(gt_file).get_data() == idx)
        pred_file_1 = prefix + f[:-13] + '/' + cla + '.nii'
        img_pred = nib.load(pred_file_1).get_data()
        img_pred = img_pred > 0.5
        struct = ndimage.generate_binary_structure(3, 2)
        img_pred = ndimage.morphology.binary_closing(img_pred, structure = struct)
        img_pred = get_largest_two_component(img_pred, False, 20)
        #rotate the nii image
        img_pred = np.transpose(img_pred, (2,1,0))
        img_pred = np.rot90(img_pred, k=2, axes= (1,2))
        #uncomment to output the result binary mask
        # img = nib.Nifti1Image(img_pred.astype(float), OUTPUT_AFFINE)
        # nib.save(img, './Output/' + f[:-13] + 'result.nii.gz')
        true_pos = np.float(np.sum(img_gt * img_pred))
        union = np.float(np.sum(img_gt) + np.sum(img_pred))
        d = true_pos * 2.0 / union
        # print('%s: %s'%(f[:-12], d))
        #ignore the eval if the gt does not contain the label
        if np.sum(img_gt) != 0:
            dice.append(d)

    print('%s : %s images mean %s std %s'%(cla, len(dice), np.mean(dice), np.std(dice)))

def main():
    for i in range(1,10):
        evaluate(i)

if __name__ == "__main__":
    main()
