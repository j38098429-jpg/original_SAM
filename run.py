# Sekeun Kim 
# cleaned code @ 05102023
# --------------------------------------------------------
import numpy as np
import os
import time
from tqdm import tqdm

from typing import List, Tuple
import argparse
import datetime
import torch
import torch.backends.cudnn as cudnn
import sys
import cv2 

# sys.path.append(".")

from utils.Echo.metric_hepler import compute_metrics, save_report_mia

os.environ['CUDA_VISIBLE_DEVICES']='0'

from dataset.data import build_data_for_MIA
from utils.save_utils import *
from utils.config_util import Config
from utils.util import set_random_seed, str2bool

import copy
import json

import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import numpy as np
def get_args_parser():
    parser = argparse.ArgumentParser('SAM fine-tuning', add_help=True)
    parser.add_argument(
        "--config", help="Path to the training config file.", default="configs/segmentation/sam/config.yaml",
    )
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=10_000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--data_path', default="/mount/mnt/raid/AorticStenosis/EchoNet/data_cleaned/image/", type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./results/segmentation/Echo_CAMUS/sam', help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='/mount/home/local/PARTNERS/sk1064/workspace/medical_sam/models/segmentation/segment_anything/model/ckpt/sam_vit_b_01ec64.pth', help='resume from checkpoint')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    
    # no need to change    
    parser.add_argument('--train_mode', default=False, type=str2bool)
    

    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    
    parser.add_argument('--view_type', default="A4CH", type = str)
    
    parser.add_argument('--Usage_percent', default=1.0, type = float)
    
    parser.add_argument('--print_freq', default=5, type = int)
    parser.add_argument('--save_interval', default=5, type = int)
    
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    
    parser.add_argument('--img_size', type=int,
                    default=5, help='input patch size of network input')
    
    parser.add_argument('--vit_name', type=str,
                    default='vit_h', help='select one vit model')

    parser.add_argument('--num_classes', type=int,
                    default=2, help='num classes')  
    
    parser.add_argument('--rank', type=int,default=4, help='num classes')    
    
    # training_camus2d_data_path = "/mount/mnt/raid/AorticStenosis/CAMUS/data_cleaned/"
    # training_camus2dt_data_path = '/mount/home/local/PARTNERS/sk1064/workspace/samus/dataset/Echo/camus/data_list/train_dict.json'
    # test_json_mgh = '/mount/home/local/PARTNERS/sk1064/workspace/samus/dataset/Echo/mgh/data_list/0/test_dict.json'
    # test_json_camus = '/mount/home/local/PARTNERS/sk1064/workspace/samus/dataset/Echo/camus/seq_data_list/test_dict.json'
    # output_dir = './results/segmentation/sam/CAMUS/'
    parser.add_argument('--base_data_path', type=str, default='/mount/mnt/raid/AorticStenosis')

    parser.add_argument('--use_PEFT', default=False, type=str2bool)
    parser.add_argument('--test_type', type=str, default='mgh')    
    parser.add_argument('--loss_type', type=str, default='CE')    
    parser.add_argument('--model_type', type=str, default='sam_vit_b_basic') #sam_vit_h  sam_vit_h_ST_full_spatio  unet3D sam_vit_h_ST_full_spatio  sam_vit_h
    parser.add_argument('--input_type', type=str, default='2D') #2D 3D 2DT 
    parser.add_argument('--multi', type=bool, default=False)
    parser.add_argument('--aug', type=str, default='randaugment') #randaugment #vanilla
    
    return parser

def custom_collate_fn(instances: List[Tuple]):
    return instances

trained_model_dict = {
    "2D_sam_vit_h": "/mount/home/local/PARTNERS/sk1064/workspace/medical_sam/results/sam_vit_h_data_usage_1.0_PEFT_True_LOSS_CE/checkpoint-latest.pth",
    # "2DT_sam_vit_h_ST_full_spatio": "/mount/home/local/PARTNERS/sk1064/workspace/samus/results/sam_vit_h_ST_full_spatio_data_usage_1.0_v0_3D_ST_3D_PEFT_single_w_adapter_full_spatio_False_LOSS_CE/checkpoint-latest.pth"
}

def run(args, cfg):
    cfg.trainer.model_type = args.model_type 
    cfg.data.input_type = args.input_type
    cfg.data.augmentations.aug = args.aug
    cfg.trainer.multi = args.multi
    cfg.trainer.base_data_path = args.base_data_path

    seed = args.seed
    set_random_seed(seed)

    device = torch.device(args.device)

    cudnn.benchmark = True
    
    """Load Data"""
    dataset_train, dataset_valid = build_data_for_MIA(cfg, args)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""INERENCE"""""""""""""""""""""""""""""""""""""""
        
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=dataset_valid.collate_fn
    )

        
    # MIA EVALUATION
    with torch.no_grad(): 
        
        # 2D + TIME : CAMUS SET
        report_dict = {"p_id" : [], "ef_error": [], "ef_pred": [], "ef_gt" : [], "c_x" : [], "c_y": [], "thres_endo" : [], "thres_epi" : [], "avg_endo": [], "avg_epi" : [], "ef_info": [], "dsc": [], "hd": [], "asd": [], "n_pi_x_pred": [], "n_pi_y_pred": [], "n_pi_x_gt": [], "n_pi_y_gt": [],"num_px_mask": [],"num_px_pred": []}
    
        with torch.cuda.amp.autocast():
            sam_checkpoint = "/mount/home/local/PARTNERS/sk1064/workspace/medical_sam/models/segmentation/segment_anything/model/ckpt/sam_vit_h_4b8939.pth"
            model_type = "vit_h"

            device = "cuda"

            sam = sam_model_registry[model_type]( checkpoint=sam_checkpoint)
            sam.to(device=device)
            model = sam
        
        
        mode = ["mode3"]
        
        for indv_mode in mode:
            dest_path =  args.output_dir + indv_mode
            for data_iter_step, batch in tqdm(enumerate(data_loader_valid)):
                copied_batch = copy.deepcopy(batch)

                # batch[0]["image"][0,0,:,:]
                p_id = batch[0]["patient_name"]
                
                area_dict = {}
                area_dict[p_id] = []
                if cfg.data.input_type == "2D":
                    if args.test_type == "camus":
                        copied_batch[0]["input"] = batch[0]["image"]
                        copied_batch[0]["mask"] = batch[0]["mask"]
                    else:
                        copied_batch[0]["input"] = batch[0]["image"]
                        copied_batch[0]["mask"] = torch.tensor(batch[0]["original_mask"])
                                    
                if cfg.trainer.model_type in ["sam_vit_h", "sam_vit_b_basic"]:
                    new_dict = {}
                    copied_batch[0]["input"] = torch.stack([ x["image"] for x in copied_batch], axis = 0)
                    copied_batch[0]["mask"] = torch.stack([ x["mask"] for x in copied_batch], axis = 0)
                    copied_batch[0]["original_size"] = batch[0]["original_size"]
                    input_tensor = copied_batch[0]["input"].squeeze(0)

                    new_dict = [ {"original_size": copied_batch[0]["original_size"], "image" : input_tensor[i].float().cuda()} for i in range(input_tensor.shape[0])]
                    
                    if cfg.trainer.model_type== "sam_vit_b_basic": 
                        _, fr, ch, h, w = copied_batch[0]["input"].size()
                        predictor = SamPredictor(model)
                        pred = np.zeros((fr, 1, 1024,1024))
                        
                        if indv_mode == "mode1":
                            for indv_fr in tqdm(range(fr)):
                                img = copied_batch[0]["input"][0,indv_fr,:,:,:]
                                img = np.array(img.transpose(0, 1).transpose(1, 2)).astype(np.uint8)
                                
                                # img = img.astype(np.round)
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                
                                predictor.set_image(img)
                                
                                input_point = np.array([copied_batch[0]["point_prompt"][0]])
                                # input_point = np.array([[500, 375], [350, 625]])
                                input_label = np.array([1])
                                                            
                                masks, scores, logits = predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                multimask_output=True,
                                )
                                                                
                                # for i, (mask, score) in enumerate(zip(masks, scores)):
                                #     plt.figure(figsize=(10,10))
                                #     plt.imshow(img)
                                #     show_mask(mask, plt.gca())
                                #     show_points(input_point, input_label, plt.gca())
                                #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                                #     plt.axis('off')
                                #     plt.show()
                                #     plt.savefig(f"Mask {i+1}.jpg")
                                    
                                mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
                                
                                masks, _, _ = predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                mask_input=mask_input[None, :, :],
                                multimask_output=False,
                                )                        
                                
                                # plt.figure(figsize=(10,10))
                                # plt.imshow(img)
                                # show_mask(masks, plt.gca())
                                # show_points(input_point, input_label, plt.gca())
                                # plt.axis('off')
                                # plt.show() 
                                # plt.savefig('test_.jpg')

                                pred[indv_fr, 0, :, :] = masks[0]
                                
                        elif indv_mode == "mode2":
                            
                            for indv_fr in tqdm(range(fr)):
                                img = copied_batch[0]["input"][0,indv_fr,:,:,:]
                                img = np.array(img.transpose(0, 1).transpose(1, 2)).astype(np.uint8)
                                
                                # img = img.astype(np.round)
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                
                                predictor.set_image(img)
                                
                                input_point = np.array([copied_batch[0]["point_prompt"][0], copied_batch[0]["point_prompt"][2]])
                                # input_point = np.array([[500, 375], [350, 625]])
                                input_label = np.array([1, 0])
                                                            
                                masks, scores, logits = predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                multimask_output=True,
                                )
                                                                
                                for i, (mask, score) in enumerate(zip(masks, scores)):
                                    plt.figure(figsize=(10,10))
                                    plt.imshow(img)
                                    show_mask(mask, plt.gca())
                                    show_points(input_point, input_label, plt.gca())
                                    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                                    plt.axis('off')
                                    plt.show()
                                    plt.savefig(f"Mask {i+1}.jpg")
                                    
                                mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
                                
                                masks, _, _ = predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                mask_input=mask_input[None, :, :],
                                multimask_output=False,
                                )                        
                                
                                pred[indv_fr, 0, :, :] = masks[0]
                                
                                plt.figure(figsize=(10,10))
                                plt.imshow(img)
                                show_mask(masks, plt.gca())
                                show_points(input_point, input_label, plt.gca())
                                plt.axis('off')
                                plt.show() 
                                plt.savefig('test_.jpg')
                                print ('')
                                        
                        elif indv_mode == "mode3":
                            for indv_fr in tqdm(range(fr)):
                                img = copied_batch[0]["input"][0,indv_fr,:,:,:]
                                img = np.array(img.transpose(0, 1).transpose(1, 2)).astype(np.uint8)
                                
                                # img = img.astype(np.round)
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                
                                predictor.set_image(img)
                                
                                input_point = np.array([copied_batch[0]["point_prompt"][0], copied_batch[0]["point_prompt"][2]])
                                # input_point = np.array([[500, 375], [350, 625]])
                                input_label = np.array([1, 0])
                                                            
                                masks, scores, logits = predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                multimask_output=True,
                                )
                                                                
                                # for i, (mask, score) in enumerate(zip(masks, scores)):
                                #     plt.figure(figsize=(10,10))
                                #     plt.imshow(img)
                                #     show_mask(mask, plt.gca())
                                #     show_points(input_point, input_label, plt.gca())
                                #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                                #     plt.axis('off')
                                #     plt.show()
                                #     plt.savefig(f"Mask {i+1}.jpg")
                                    
                                mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
                                # mask_input = masks[np.argmax(scores), :, :]  # Choose the model's best mask

                                masks, _, _ = predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                mask_input=mask_input[None, :, :],
                                multimask_output=False,
                                )                      
                                  
                                pred[indv_fr, 0, :, :] = masks
                                
                                if indv_fr in [int(float(copied_batch[0]["ED"]))-1, int(float(copied_batch[0]["ES"]))-1]:
                                    
                                    
                                    plt.figure(figsize=(10,10))
                                    plt.imshow(img)
                                    show_mask(masks, plt.gca())
                                    show_points(input_point, input_label, plt.gca())
                                    plt.axis('off')
                                    plt.show() 
                                    dir_name = './results_point_prompts/' + indv_mode + '/' + copied_batch[0]["patient_name"] +'/'
                                    
                                    os.makedirs(dir_name, exist_ok=True)
                                    plt.savefig(dir_name + str(indv_fr) + '.jpg')
                        
                        elif indv_mode == "mode4":
                            
                            for indv_fr in tqdm(range(fr)):
                                img = copied_batch[0]["input"][0,indv_fr,:,:,:]
                                img = np.array(img.transpose(0, 1).transpose(1, 2)).astype(np.uint8)
                                
                                # img = img.astype(np.round)
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                
                                predictor.set_image(img)
                                
                                input_box = np.array(copied_batch[0]["box_prompt"])
                                
                                masks, _, _ = predictor.predict(
                                    point_coords=None,
                                    point_labels=None,
                                    box=input_box[None, :],
                                    multimask_output=False,
                                )
                                                                
                                pred[indv_fr, 0, :, :] = masks[0]

                                # plt.figure(figsize=(10, 10))
                                # plt.imshow(img)
                                # show_mask(masks[0], plt.gca())
                                # show_box(input_box, plt.gca())
                                # plt.axis('off')
                                # plt.show()
                                # plt.savefig('test_bb.')
                                
                                if indv_fr in [int(float(copied_batch[0]["ED"]))-1, int(float(copied_batch[0]["ES"]))-1]:                                    
                                    plt.figure(figsize=(10,10))
                                    plt.imshow(img)
                                    show_mask(masks[0], plt.gca())
                                    show_box(input_box, plt.gca())
                                    plt.axis('off')
                                    plt.show() 
                                    dir_name = './results_point_prompts/' + indv_mode + '/' + copied_batch[0]["patient_name"] +'/'
                                    ()
                                    os.makedirs(dir_name, exist_ok=True)
                                    plt.savefig(dir_name + str(indv_fr) + '.jpg')

                                print ('')    
                                
                        gt = copied_batch[0]["resized_label_1024"]
                        
                        # args.output_dir = args.output_dir + "_" + indv_mode
                        
                        dc, hd, asd, c_x, c_y, num_px_mask, num_px_pred, ef_pred, ef_gt, ef_error , avg_endo, avg_epi, thres_endo, thres_epi= compute_metrics(gt, pred, 3, (copied_batch[0]["original_size"][1], copied_batch[0]["original_size"][2]), copied_batch[0]["voxel_space"], [copied_batch[0]["ED"], copied_batch[0]["ES"]], only_ED_ES= True)
                
                        if ef_pred != 0:
                            report_dict["p_id"].append(p_id)
                            report_dict["dsc"].append(dc)
                            report_dict["hd"].append(hd)
                            report_dict["asd"].append(asd)
                            
                            report_dict["c_x"].append( [c_x["1"]])
                            report_dict["c_y"].append( [c_y["1"]])
                            
                            report_dict["num_px_mask"].append(num_px_mask)
                            report_dict["num_px_pred"].append(num_px_pred)
                            
                            report_dict['avg_endo'].append(avg_endo)
                            # report_dict['avg_epi'].append(avg_epi)
                            
                            report_dict['thres_endo'].append(thres_endo)
                            # report_dict['thres_epi'].append(thres_epi)
                            
                            report_dict["ef_pred"].append(ef_pred)
                            report_dict["ef_gt"].append(ef_gt)
                            report_dict["ef_error"].append(ef_error)
                        
                            print(len(report_dict["p_id"]))
                            print (report_dict["dsc"])
            

            avg_dsc, std_dsc = save_report_mia(report_dict["p_id"], report_dict["dsc"], title_text = "DSC", dest= dest_path )
            avg_hd, std_hd = save_report_mia(report_dict["p_id"], report_dict["hd"], title_text = "HD", dest= dest_path)
            avg_asd, std_asd = save_report_mia(report_dict["p_id"], report_dict["asd"],title_text = "ASD", dest= dest_path)
            
            avg_n_px_mask, std_n_px_mask = save_report_mia(report_dict["p_id"], report_dict["num_px_mask"],title_text = "N_pixel_mask", dest= dest_path)
            
            avg_n_px_pred, std_n_px_pred = save_report_mia(report_dict["p_id"], report_dict["num_px_pred"],title_text = "N_pixel_pred", dest= dest_path)
            
            result_dict = {"dsc": [avg_dsc, std_dsc], "avg_hd": [avg_hd, std_hd], "avg_asd": [avg_asd,std_asd], "avg_n_px_mask": [avg_n_px_mask,std_n_px_mask],"avg_n_px_pred": [avg_n_px_pred,std_n_px_pred], "TC_endo": np.mean(report_dict['avg_endo']), "TC_inconsistency_endo": 0, "EF_pred" : report_dict["ef_pred"] , "EF_gt" : report_dict["ef_gt"], "Ef_error" : report_dict["ef_error"]}
                        
            with open( os.path.join(dest_path,'report.json'), 'w') as outfile:
                json.dump(result_dict, outfile, indent=4)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    cfg = Config(args.config)
    run(args, cfg)
    
    