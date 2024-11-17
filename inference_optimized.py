import argparse
import os
import subprocess
import cv2
import ffmpegcv
import ffmpegcv.ffmpeg_writer_noblock
import onnx
import onnxruntime
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_utils.process import get_audio_feature
from unet import Model
# from unet2 import Model
# from unet_att import Model

import time
parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="hubert")
parser.add_argument('--dataset', type=str, default="")  
# parser.add_argument('--audio_feat', type=str, default="")
parser.add_argument('--wav', type=str, default="")  # end with .wav please
parser.add_argument('--save_path', type=str, default="")     # end with .mp4 please
parser.add_argument('--checkpoint', type=str, default="")

args = parser.parse_args()

checkpoint = args.checkpoint
save_path = args.save_path
dataset_dir = args.dataset

mode = args.asr
wav = args.wav
onnx_model = True if args.checkpoint is not None and args.checkpoint.endswith(".onnx") else False



audio_feat_path = wav.replace('.wav', '_hu.npy')

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
if(not os.path.exists(audio_feat_path)):
    if mode == "wenet":
        subprocess.run("python ./data_utils/wenet_infer.py "+wav,shell=True)
    if mode == "hubert":
        subprocess.run("python ./data_utils/hubert.py --wav "+wav, shell=True)
audio_feats = np.load(audio_feat_path)
img_dir = os.path.join(dataset_dir, "full_body_img/")
lms_dir = os.path.join(dataset_dir, "landmarks/")
# len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(img_dir+"0.jpg")
h, w = exm_img.shape[:2]

if mode=="hubert":
    # video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
    video_writer = ffmpegcv.noblock(ffmpegcv.VideoWriterNV,save_path, 'h264', fps=25, bitrate='2000k')
if mode=="wenet":
    # video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
    video_writer = ffmpegcv.noblock(ffmpegcv.VideoWriterNV,save_path, 'h264', fps=20, bitrate='2000k')

step_stride = 0
img_idx = 0

if(onnx_model):
    onnx_weights = onnx.load(checkpoint)
    onnx.checker.check_model(onnx_weights)
    providers = ["CUDAExecutionProvider"]
    net = onnxruntime.InferenceSession(checkpoint, providers=providers)
else:
    net = Model(6, mode).cuda()
    net.load_state_dict(torch.load(checkpoint))
    net.eval()

end_time = time.time()  # Capture the end time
execution_time = end_time - start_time  # Calculate the execution time
print(f"Models are loaded in {execution_time} seconds ")


start_time = time.time()
print("Start caching images and landmarks features...")

# lets take images into gpu memory first and we expect 2 returns which means we are going to divide the length with 4
min_audio_length = 30*25
audio_length = audio_feats.shape[0]
frames_needed = audio_length if audio_length <= min_audio_length else  (audio_length // 4) + (audio_length % 4)
print(str(frames_needed)+" frames will be used, which means "+str(frames_needed/25)+" seconds of video, total video length is "+str(audio_length/25)+" seconds")
cached_images= []
cached_landmarks= []
for i in range(frames_needed):
    cached_images.append(cv2.imread(img_dir+str(i)+".jpg"))
    # reading landmarks
    lms_path = lms_dir + str(i)+'.lms'
    
    with open(lms_path, "r") as f:
        lms_list = []
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    cached_landmarks.append(np.array(lms_list, dtype=np.int32))

end_time = time.time()  # Capture the end time
execution_time = end_time - start_time  # Calculate the execution time
print(f"Images and landmarks cached in {execution_time} seconds ")

start_time = time.time()
print("Creating video...")
# set total images wi are going to work with
len_img = cached_images.__len__() -1

cached_image_tensors = []
for i in range(audio_feats.shape[0]):
    start_process_time = time.time()
    
    #this will do the rewind when needed
    if img_idx==len_img-1:
        step_stride = -1
    if img_idx == 0:
        step_stride = 1

    #reading image
    img = cached_images[img_idx]
    
    try :
        if(onnx_model):
            img_concat_T = cached_image_tensors[img_idx][0].cpu()
        else:
            img_concat_T = cached_image_tensors[img_idx][0].cuda()
    except:
        # reading landmarks
        lms = cached_landmarks[img_idx]
        
        #determining width, height and positions
        xmin = lms[1][0]
        ymin = lms[52][1]

        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width
        height = ymax - ymin

        #get a crop from original image and determine its size
        crop_img = img[ymin:ymax, xmin:xmax]
        # h, w = crop_img.shape[:2] # isnt this same with the width and height above ?

        #get resized version of crop_img
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)

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
        cache=[img_concat_T,crop_img,xmin,xmax,ymin,ymax,width,height]
        cached_image_tensors.append(cache)
        if(onnx_model):
            img_concat_T= img_concat_T.cpu()
        else:
            img_concat_T= img_concat_T.cuda()
    

    # get the audio features tensor
    audio_feat = get_audio_features(audio_feats, i)
    if mode=="hubert":
        audio_feat = audio_feat.reshape(32,32,32)
    if mode=="wenet":
        audio_feat = audio_feat.reshape(256,16,32)
    audio_feat = audio_feat[None]
    
    # put those tensors in gpu memory
    if(onnx_model):
        audio_feat = audio_feat.cpu()
    else:
        audio_feat = audio_feat.cuda()

    # if(i==1):
        # print(f"audio shape : {audio_feat.device}")
        # print(f"image shape : {img_concat_T.device}")
    # print(f"{i+1}. Image tensor ready to process {time.time() - start_process_time}  in second(s) ")
    # pred_start_time = time.time()
    # get prediction by using audio and image tensors
    if(onnx_model):
        pred=net.run(None, {"input":img_concat_T.numpy(),"audio":audio_feat.numpy()})[0][0]
    else:
        with torch.no_grad():
            pred = net(img_concat_T, audio_feat)[0]

    # print(f"{i+1}. Got prediction {time.time() - pred_start_time}  in second(s) ")
        
    # take the prediction in cpu and return/flip it  
    if(onnx_model):
        pred = pred.transpose(1,2,0)*255
    else:
        pred = pred.cpu().numpy().transpose(1,2,0)*255

    pred = np.array(pred, dtype=np.uint8)

    #merge prediction and original image
    crop_img_ori = cached_image_tensors[img_idx][1]
    crop_img_ori[4:164, 4:164] = pred

    #resize again ( why )
    crop_img_ori = cv2.resize(crop_img_ori, (cached_image_tensors[img_idx][6], cached_image_tensors[img_idx][7]))
    
    #merge into the original image (frame)
    img[cached_image_tensors[img_idx][4]:cached_image_tensors[img_idx][5], cached_image_tensors[img_idx][2]:cached_image_tensors[img_idx][3]] = crop_img_ori

    # print(f"{i+1}. Predicted image ready to write into video {time.time() - start_process_time}  in second(s) ")
    #write as a video frame
    video_writer.write(img)
    # print(f"{i+1}. Video frame write {time.time() - start_process_time}  in second(s) ")
    img_idx = img_idx+step_stride
video_writer.release()
cmd = f'ffmpeg -y -loglevel quiet -i {save_path} -i {wav} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {save_path.replace(".mp4","_result.mp4")}'
os.system(cmd)

end_time = time.time()  # Capture the end time
execution_time = end_time - start_time  # Calculate the execution time
print(f"Video created in {execution_time} seconds ")