import torch
import numpy as np
import os
from pydicom import dcmread
from utils.convert_image import ds2img
import cv2
import math
import re
import dataset.Voxel as vx
import nrrd as nrrd
import pandas as pd
from typing import List, Tuple
import random
import nibabel as nib

def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        # self.data_dir = args["data_dir"]
        self.cfg = args["cfg"]
        self.data_dict = args["data_dict"]
        self.transform = args["transform"]
        self.image_dir_list = []
        self.num_classes = args["num_classes"]
        # self.input_type = args["input_type"]
        
    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self,idx):
                
        idx = str(idx)

        if self.data_dict[idx]["input_dir"].split('/')[5] == "ACDC":
            tag ="None"
            org_img = load_nii(self.data_dict[idx]["input_dir"])
            mask = load_nii(self.data_dict[idx]["json_dir"])
            mask = mask[0]
            
        else:
            tag = "CMR"
            org_img = load_nii(self.data_dict[idx]["input_dir"])
            sh = np.shape(org_img[0])
            mask = np.zeros((sh[0], sh[1]))
            
        img = np.expand_dims(org_img[0], axis=0)

        img = img.astype(int)
        
        
        
          
        search_values = [0, 1, 2, 3]
        
        # create a list to store the results
        random_pixels = []
        
        
        if self.cfg.trainer.prompt == "default":
            for search_value in search_values:
                indices = np.argwhere(mask == search_value)
                random_index = np.random.choice(indices.shape[0], size=1, replace=False)
                random_pixel = indices[random_index][0]
                random_pixels.append(random_pixel)

                # Find the innner coordinate of the 1s
                c_x = [int(np.mean(random_pixel[1])) for random_pixel in random_pixels]
                c_y = [int(np.mean(random_pixel[0])) for random_pixel in random_pixels]

                #check center of mask
                # mask[int(c_y[0])-10 : c_y[0] +10, c_x[0]-10:c_x[0]+10] = 5
                # mask[int(c_y[1])-2: c_y[1] +2, c_x[1]-2:c_x[1]+2] = 5
                
                point_coords = [[c_y[0], c_x[0]], [c_y[1], c_x[1]]]
        else:
            x = random.randint(0, 1024)
            y = random.randint(0, 1024)
            
            x2 = random.randint(0, 1024)
            y2 = random.randint(0, 1024)
            
            x3 = random.randint(0, 1024)
            y3 = random.randint(0, 1024)
            
            x4 = random.randint(0, 1024)
            y4 = random.randint(0, 1024)

            point_coords = [[y4, x4], [y3, x3], [y2, x2], [y, x]]
        
        point_labels = [0,1,2,3]
        
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int)
        
        coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
        
        data = {"original_input" : org_img[0], 
                "input": img,
                "label": np.expand_dims(mask, axis =0), 
                "point_coords":coords_torch, 
                "point_labels":labels_torch,  
                "patient_name" : self.data_dict[idx]["patient_number"],
                "frame_num" : self.data_dict[idx]["frame"], 
                "tag" : tag, 
                "original_size" : np.shape(org_img[0])
                }
        
        data = self.transform(data)
        
        one_hot = np.zeros(( np.shape(mask)[0], np.shape(mask)[1], 4), dtype=np.float16)
        
        for i in range(4):
            one_hot[..., i] =  np.array( data["label"] == i).astype(np.float16)
        
        one_hot = np.transpose(one_hot, (2, 0, 1))
                
        data["image"] = torch.tensor(np.repeat(np.expand_dims(data["input"], axis=0) , 3,  axis=0)).float()
        data["mask"] = torch.from_numpy(one_hot)
        one_hot[2,:,:]
        return data
    
        # data["label"].float()
        # data["input"] = data["input"].permute(1, 0, 2, 3)



        #.npy
        # dir_imgED = "/mount/home/local/PARTNERS/sk1064/workspace/data/Image/ED"
        # dir_imgES = "/mount/home/local/PARTNERS/sk1064/workspace/data/Image/ES"
        
        # # .bin
        # dir_maskED = "/mount/home/local/PARTNERS/sk1064/workspace/data/Mask/ED"
        # dir_maskES = "/mount/home/local/PARTNERS/sk1064/workspace/data/Mask/ES"

        # # .nrrd : no mask exists
        # dir_mgh_data = "/mount/home/local/PARTNERS/sk1064/workspace/data/"
        # dir_mgh_data = "/mount/mnt/raid/CMR/MGH_CMR"
        # raw_file_name = self.data_dict[idx]
        
        # if re.match(r'.*_ED.npy', raw_file_name):
        #     img_file = os.path.join(dir_imgED, raw_file_name)
        #     file_name = raw_file_name.replace(".npy", ".bin")
        #     mask_file = os.path.join(dir_maskED, file_name)
        # elif re.match(r'.*_ES.npy', raw_file_name):
        #     img_file = os.path.join(dir_imgES, raw_file_name)
        #     file_name = raw_file_name.replace(".npy", ".bin")
        #     mask_file = os.path.join(dir_maskES, file_name)
        # else:
        #     img_file = os.path.join(dir_mgh_data, raw_file_name)
        #     file_name = img_file.split("/")[-1]
        #     mask_file = "None"
            
        # if mask_file == "None":   
        #     tag = "CMR"
        #     info_xlsx = "/mount/home/local/PARTNERS/sk1064/workspace/data/CMR/slice_numbers_HFpEF_with_notes.xlsx" 
        #     df = pd.read_excel(info_xlsx)
            
        #     # id_name = raw_file_name.split("/")[-2]

        #     # # Find the row(s) where the ID matches the ID list
        #     # id_matches = df.loc[df['ID_list'] == id_name]
            
        #     # img_file = '/mount/' + img_file
        #     # path = os.path.dirname(img_file)
        #     # list_path = img_file.split('/')
            
        #     # folder_name = list_path[-1].split('_')[0] + '_' + list_path[-1].split('_')[1]
        #     # add_path = "Org3D"
        #     # file_name = list_path[-1].split('_')[-1]
            
        #     # img_file = os.path.join(dir_mgh_data, folder_name, add_path + '_' + file_name)
            
            
        #     # print (img_file)
        #     org_img ,header = nrrd.read(raw_file_name)
            
        #     # if id_matches["suggestions"].iloc[0] == "remove blank slices before AI steps":
        #     #     # exclude this case
        #     #     blank_slice = id_matches["blank_slice_num"].iloc[0]
        #     #     arr = np.delete(arr, int(blank_slice), axis=2)
        #     #     # concatenate the 9th and 11th slices
        #     #     arr = np.concatenate((arr[:,:, :blank_slice-1], arr[:,:,blank_slice+1:]), axis=2)
        #     #     img[:,:,15]
        #     sh = np.shape(org_img)
        #     mask = np.zeros((sh[2], sh[0], sh[1]))
        #     img = np.transpose(org_img, (2, 1, 0))
        #     img = (img - np.min(org_img) )/ (np.max(img) - np.min(img))
        # else:
        #     tag ="None"
        #     img = np.load(img_file)
        #     pyvox.ReadFromBin(mask_file)
        #     mask = pyvox.m_Voxel
            
        # set input volume 
        # dz, dx, dy = np.shape(img)
        
        # max_fr = 16
        
        # img = img[:max_fr, : ,:] if dz > max_fr else np.pad(img, ((0, max_fr- dz), (0, 0), (0, 0)), 'constant', constant_values=0)[:max_fr, :, :]
        # mask = mask[:max_fr, : ,:] if dz > max_fr else np.pad(mask, ((0, max_fr- dz), (0, 0), (0, 0)), 'constant', constant_values=0)[:max_fr, :, :]

        # img = np.expand_dims(img, axis=1)
        
        # org_img = org_img.astype(int)
        
        # data = {"original_input" : org_img, "input": img, "label": mask, "file_name" : raw_file_name, "tag" : tag, "original_shape" : np.shape(img)}
        
        # data = self.transform(data)
        
        # data["input"] = data["input"].permute(1, 0, 2, 3)
        
        # return data
    def collate_fn(self, instances: List[Tuple]):
        # image_list, mask_list = [], []
        # # flattern
        # for b in instances:
        #     image, mask = b
        #     image_list.append(image)
        #     mask_list.append(mask)

        # # stack
        # image_stack = torch.stack(image_list)
        # mask_stack = torch.stack(mask_list)

        # # sort and add to dictionary
        # return_dict = {
        #     "image": image_stack,
        #     "mask": mask_stack
        # }

        # return return_dict
        return instances
if __name__ == "main":
    dataset = Dataset()