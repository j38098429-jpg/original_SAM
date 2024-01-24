import numpy as np
import torch
from PIL import Image
from typing import List, Tuple

from pydicom import dcmread
import os
import cv2
import json
from torchvision.transforms import functional as F
from PIL import Image
import random

def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask'''
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return np.array([x_min, y_min, x_max, y_max])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.cfg = args["cfg"]
        self.data_dict = args["data_dict"]
        self.selected_class = args["selected_classes"]
        self.transform = args["transform"]

    def __len__(self):
        print("Total number of data", len(self.data_dict))

        return len(self.data_dict)

    def __getitem__(self, idx):
        idx = str(idx)
        
        num_classes = self.cfg.data.num_selected_view
        
        json_dir = self.data_dict[idx]["json_dir"]
        with open(json_dir) as json_file:
            label = json.load(json_file)

        image_dir = self.data_dict[idx]["input_dir"]
        
        # cts -> (b,n, 1,2)
        #train
        frame_num = self.data_dict[idx]["frame"] + 1
        
        #test
        # # frame_num = self.data_dict[idx]["frame"]
        patient_name = self.data_dict[idx]["patient_number"]

        input = cv2.imread(image_dir)[:, :, :3]
        input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        
        h, w = input.shape

        mask = np.zeros((h, w), dtype=np.uint8)

        for i, class_label in enumerate(self.cfg.data.view[:num_classes]):
            if class_label in label["Frame"][str(frame_num)].keys():
                if label["Frame"][str(frame_num)][class_label] != None:
                    class_contour = label["Frame"][str(frame_num)][class_label]
                    if len(class_contour) != 1:
                        for indv_class_contour in class_contour:
                            cv2.drawContours(mask, [np.array(indv_class_contour)], -1, (i ), -1)
                    else:
                        cv2.drawContours(mask, np.array(class_contour), -1, (i ), -1)
        
        # define a sample 2D array
        search_values = list(range(num_classes))
        # create a list to store the results
        random_pixels = []
        
        if self.cfg.trainer.prompt == "default":
            # loop through the search values and retrieve a random pixel location for each
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
            
            point_coords = [[c_y[0], c_x[0]], [c_y[1], c_x[1]],[c_y[2], c_x[2]]]
        
        else:
            x = random.randint(0, h)
            y = random.randint(0, w)
            x2 = random.randint(0, h)
            y2 = random.randint(0, w)
            x3 = random.randint(0, h)
            y3 = random.randint(0, w)
            x4 = random.randint(0, h)
            y4 = random.randint(0, w)

            point_coords = [[y4, x4], [y3, x3],[y2, x2]]
            
            
        point_labels = [0,1,2]
                
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int)
        coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
                
        # # define a sample 2D array
        # search_values = [0, 1, 2, 3]
                
        # bboxes = []
        # masks = []     
        
        # for r_idx in [1, 2, 3]:
        #     binary_mask = (mask == r_idx)
        #     # generate bounding box prompt
        #     box = get_bbox_from_mask(binary_mask)
            
        #     box_torch = torch.as_tensor(box, dtype=torch.float)
            
        #     box_torch = box_torch.reshape(1, 4)[0].numpy().astype(int)
            
        #     bboxes.append(box_torch)
        #     masks.append(binary_mask)            
            
        #     # # # check bounding box
        #     # mask[int(box_torch[1])-10 :int(box_torch[1]) +10, int(box_torch[0])-10:int(box_torch[0])+10] = 1
        #     # mask[int(box_torch[3])-10 :int(box_torch[3] )+10, int(box_torch[2])-10:int(box_torch[2])+10] = 1
        
        # bboxes = np.stack(bboxes, axis=0)
        # masks = np.stack(masks, axis=0)
        
        data  = {}
        data["input"] = input
        data["label"] = mask

        data = self.transform(data)       
        
        data = {
            # resized image
            "image": data["input"].unsqueeze(0).expand(3, self.cfg["data"]["augmentations"]["resize_h_w"][0], self.cfg["data"]["augmentations"]["resize_h_w"][0]),
            # original input gt size
            # "mask": torch.tensor(masks).long(), # segment any chamber.
            "mask": data["label"],
            "patient_name": patient_name,
            "frame_num" : frame_num,
            "original_size": tuple(np.shape(input)),
            # "boxes":torch.tensor(bboxes),# segment any chamber.
            "point_coords":coords_torch, 
            "point_labels":labels_torch,
            }
        return data

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