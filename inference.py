import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet import Model
# from unet2 import Model
# from unet_att import Model

import time
parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="hubert")
parser.add_argument('--dataset', type=str, default="")  
parser.add_argument('--audio_feat', type=str, default="")
parser.add_argument('--save_path', type=str, default="")     # end with .mp4 please
parser.add_argument('--checkpoint', type=str, default="")
args = parser.parse_args()

checkpoint = args.checkpoint
save_path = args.save_path
dataset_dir = args.dataset
audio_feat_path = args.audio_feat
mode = args.asr

def get_audio_features(features, index):
    left = index - 8
    right = index + 8
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
    return auds

start_time = time.time()
print("Start loading models features...")

audio_feats = np.load(audio_feat_path)
img_dir = os.path.join(dataset_dir, "full_body_img/")
lms_dir = os.path.join(dataset_dir, "landmarks/")
len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(img_dir+"0.jpg")
h, w = exm_img.shape[:2]

if mode=="hubert":
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P', 'G'), 25, (w, h))
if mode=="wenet":
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P', 'G'), 20, (w, h))
step_stride = 0
img_idx = 0

net = Model(6, mode).cuda()
net.load_state_dict(torch.load(checkpoint))
net.eval()

end_time = time.time()  # Capture the end time
execution_time = end_time - start_time  # Calculate the execution time
print(f"Models are loaded in {execution_time} seconds ")

start_time = time.time()
print("Creating video...")

for i in range(audio_feats.shape[0]):

    #this will do the rewind when needed
    if img_idx>len_img - 1:
        step_stride = -1
    if img_idx<1:
        step_stride = 1
    img_idx += step_stride
    img_path = img_dir + str(img_idx)+'.jpg'
    lms_path = lms_dir + str(img_idx)+'.lms'
    
    img = cv2.imread(img_path)
    lms_list = []

    # reading landmarks
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)
    
    #determining width, height and positions
    xmin = lms[1][0]
    ymin = lms[52][1]

    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width
    height = ymax - ymin

    #get a crop from original image and determine its size
    crop_img = img[ymin:ymax, xmin:xmax]
    h, w = crop_img.shape[:2] # isnt this same with the width and height above ?

    #get resized version of crop_img
    crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
    crop_img_ori = crop_img.copy()

    img_real_ex = crop_img[4:164, 4:164].copy()
    img_real_ex_ori = img_real_ex.copy()

    #prepare masked image from the img_real_ex_ori
    img_masked = cv2.rectangle(img_real_ex_ori,(5,5,150,145),(0,0,0),-1)
    
    # turn/flip the images
    img_masked = img_masked.transpose(2,0,1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
    
    # merge masked and get tensor version of it
    img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
    img_masked_T = torch.from_numpy(img_masked / 255.0)
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
    
    # get the audio features tensor
    audio_feat = get_audio_features(audio_feats, i)
    if mode=="hubert":
        audio_feat = audio_feat.reshape(32,32,32)
    if mode=="wenet":
        audio_feat = audio_feat.reshape(256,16,32)
    audio_feat = audio_feat[None]

    # put those tensors in gpu memory
    audio_feat = audio_feat.cuda()
    img_concat_T = img_concat_T.cuda()
    
    # get prediction by using audio and image tensors
    with torch.no_grad():
        pred = net(img_concat_T, audio_feat)[0]
        
    # take the prediction in cpu and return/flip it    
    pred = pred.cpu().numpy().transpose(1,2,0)*255
    pred = np.array(pred, dtype=np.uint8)

    #merge prediction and original image
    crop_img_ori[4:164, 4:164] = pred

    #resize again ( why )
    crop_img_ori = cv2.resize(crop_img_ori, (w, h))
    
    #merge into the original image (frame)
    img[ymin:ymax, xmin:xmax] = crop_img_ori

    #write as a video frame
    video_writer.write(img)
    
video_writer.release()
end_time = time.time()  # Capture the end time
execution_time = end_time - start_time  # Calculate the execution time
print(f"Video created in {execution_time} seconds ")
# ffmpeg -i test_video.mp4 -i test_audio.pcm -c:v libx264 -c:a aac result_test.mp4